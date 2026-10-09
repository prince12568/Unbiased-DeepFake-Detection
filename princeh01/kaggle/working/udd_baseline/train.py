"""Baseline training (original branch only): CLIP ViT-B/16 frozen + LoRA + FC, cross-entropy.

Paper hyper-parameters (Table 5) are the defaults. Kaggle-friendly features:
fp16 autocast, grad accumulation, optional grad checkpointing, DataParallel on 2xT4,
trainable-only checkpoints with auto-resume, and a wall-clock budget guard.

Example (Kaggle):
  !python train.py --epochs 20 --warmup_epochs 1
Resume after a session reset: just rerun the same command (picks up last.pt).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (FrameDataset, load_split_table, normalize_on_device,
                  sanity_check_files, select_frames)
from model import UDDViT


def get_args(argv=None):
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data_root", default="/kaggle/input/datasets/princeh01/udd-dataset/UDD_Dataset")
    p.add_argument("--split_json", default=None, help="default: <data_root>/dataset_split.json")
    p.add_argument("--out_dir", default="/kaggle/working/udd_baseline")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--train_frames_per_video", type=int, default=8)
    p.add_argument("--frame_sampling", choices=["even", "random"], default="even")
    p.add_argument("--val_frames_per_video", type=int, default=8)
    p.add_argument("--test_frames_per_video", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    # model
    p.add_argument("--arch", default="vit_base_patch16_clip_224.openai")
    p.add_argument("--weights_file", default=None, help="local weights if Kaggle internet is off")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--lora_rank", type=int, default=4, help="0 = frozen backbone + FC only")
    p.add_argument("--lora_alpha", type=float, default=None, help="default = rank (scale 1)")
    p.add_argument("--grad_ckpt", action="store_true")
    # optimisation (paper Table 5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64, help="effective batch size")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--warmup_epochs", type=float, default=5)
    p.add_argument("--clip_grad", type=float, default=0.0)
    p.add_argument("--class_weight", action="store_true", help="inverse-frequency CE weights")
    # system
    p.add_argument("--no_dp", action="store_true", help="disable DataParallel")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--time_budget_hours", type=float, default=11.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true", help="evaluate best.pt on test and exit")
    p.add_argument("--bad_list", default=None, help="text file of bad image paths (from scan_images.py) to drop")
    p.add_argument("--debug_videos", type=int, default=0, help="limit #videos per split (smoke test)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def predict(model, table, args, device, use_amp):
    ds = FrameDataset(table, args.data_root, args.img_size, train=False,
                      bad_log=os.path.join(args.out_dir, "bad_files_runtime.txt"))
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=args.num_workers,
                    pin_memory=device.type == "cuda")
    model.eval()
    probs = np.full(len(ds), np.nan, dtype=np.float32)  # NaN = row skipped (unreadable image)
    for x, _, idx in tqdm(dl, desc="eval", leave=False):
        x = normalize_on_device(x, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(x)
        probs[idx.numpy()] = F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
    return probs


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def compute_metrics(table, probs):
    t = table.copy()
    t["p"] = probs
    t = t[~np.isnan(probs)]
    out = {}
    for name, sub in [("all", t), ("ffpp", t[t.ds == "ffpp"]), ("celebdf", t[t.ds == "celebdf"])]:
        if len(sub) == 0:
            continue
        v = sub.groupby("video").agg(p=("p", "mean"), y=("label", "first"))
        out[name] = {
            "frame_auc": _auc(sub.label.values, sub.p.values),
            "video_auc": _auc(v.y.values, v.p.values),
            "frame_acc": float(((sub.p.values > 0.5) == sub.label.values).mean()),
            "n_frames": int(len(sub)),
            "n_videos": int(len(v)),
        }
    return out


def evaluate(model, split_table, k, args, device, use_amp):
    table = select_frames(split_table, k, mode="even")
    return compute_metrics(table, predict(model, table, args, device, use_amp))


def fmt(m):
    return " | ".join(f"{k}: vAUC {v['video_auc']*100:.2f} fAUC {v['frame_auc']*100:.2f}" for k, v in m.items())


# ----------------------------------------------------------------------------- main
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    split_json = args.split_json or os.path.join(args.data_root, "dataset_split.json")
    if not os.path.isfile(split_json):  # fallback: look one level down (e.g. an extra UDD_Dataset folder)
        for sub in sorted(os.listdir(args.data_root)) if os.path.isdir(args.data_root) else []:
            cand = os.path.join(args.data_root, sub, "dataset_split.json")
            if os.path.isfile(cand):
                print(f"dataset_split.json found in {os.path.dirname(cand)} - using it as data_root")
                args.data_root, split_json = os.path.dirname(cand), cand
                break
        else:
            raise FileNotFoundError(
                f"dataset_split.json not found under {args.data_root}. "
                "Run: !find /kaggle/input -name dataset_split.json  and pass the folder as --data_root")

    tables = {s: load_split_table(split_json, s) for s in ("train", "val", "test")}
    if args.bad_list:
        bad = {l.split("\t")[0].strip() for l in open(args.bad_list) if l.strip()}
        for s in tables:
            tables[s] = tables[s][~tables[s].path.isin(bad)].reset_index(drop=True)
        print(f"dropped {len(bad)} bad paths listed in {args.bad_list}")
    if args.debug_videos:
        for s in tables:
            keep = tables[s]["video"].drop_duplicates().sample(
                min(args.debug_videos, tables[s]["video"].nunique()), random_state=0)
            tables[s] = tables[s][tables[s].video.isin(keep)].reset_index(drop=True)
    sanity_check_files(tables["train"], args.data_root)
    for s, t in tables.items():
        print(f"{s}: {len(t)} frames, {t.video.nunique()} videos, "
              f"fake ratio {t.label.mean():.3f}, datasets {t.ds.value_counts().to_dict()}")

    model = UDDViT(args.arch, not args.no_pretrained, args.weights_file, args.lora_rank,
                   args.lora_alpha, 2, args.grad_ckpt).to(device)
    tr, tot = model.param_summary()
    print(f"trainable params: {tr/1e6:.3f}M / {tot/1e6:.2f}M total")
    net = model
    if device.type == "cuda" and torch.cuda.device_count() > 1 and not args.no_dp:
        net = nn.DataParallel(model)
        print(f"DataParallel on {torch.cuda.device_count()} GPUs")

    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    if args.eval_only:
        model.load_trainable(torch.load(best_path, map_location="cpu")["model"])
        res = evaluate(net, tables["test"], args.test_frames_per_video, args, device, use_amp)
        print("TEST", fmt(res))
        json.dump(res, open(os.path.join(args.out_dir, "test_results.json"), "w"), indent=2)
        return

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=tuple(args.betas), eps=args.eps,
                            weight_decay=args.weight_decay)
    micro_bs = args.batch_size // args.accum_steps
    n_train = len(select_frames(tables["train"], args.train_frames_per_video, args.frame_sampling))
    steps_per_epoch = (n_train // micro_bs) // args.accum_steps
    total_steps = steps_per_epoch * args.epochs
    warm = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):  # linear warmup, then cosine annealing to 0
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    cw = None
    if args.class_weight:
        freq = np.bincount(tables["train"].label.values, minlength=2).astype(np.float64)
        cw = torch.tensor(freq.sum() / (2 * freq), dtype=torch.float32, device=device)
        print("class weights", cw.tolist())

    start_epoch, best_auc, history = 0, -1.0, []
    if os.path.isfile(last_path):
        ck = torch.load(last_path, map_location="cpu")
        model.load_trainable(ck["model"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_auc, history = ck["epoch"] + 1, ck["best_auc"], ck["history"]
        print(f"Resumed from epoch {start_epoch} (best val video AUC {best_auc:.4f})")

    t_start, epoch_times = time.time(), []
    for epoch in range(start_epoch, args.epochs):
        if epoch_times and (time.time() - t_start + max(epoch_times)) / 3600 > args.time_budget_hours:
            print("Time budget reached - stopping cleanly. Re-run the same command to resume.")
            break
        t0 = time.time()
        table = select_frames(tables["train"], args.train_frames_per_video, args.frame_sampling,
                              seed=args.seed, epoch=epoch)
        dl = DataLoader(FrameDataset(table, args.data_root, args.img_size, train=True,
                                     bad_log=os.path.join(args.out_dir, "bad_files_runtime.txt")),
                        batch_size=micro_bs, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
        net.train()
        run_loss = run_acc = n_seen = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for it, (x, y, _) in enumerate(pbar):
            x = normalize_on_device(x, device); y = y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = net(x)
            loss = F.cross_entropy(logits.float(), y, weight=cw)
            scaler.scale(loss / args.accum_steps).backward()
            if (it + 1) % args.accum_steps == 0:
                if args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += loss.item() * len(y); n_seen += len(y)
            run_acc += (logits.argmax(1) == y).sum().item()
            if it % 20 == 0:
                pbar.set_postfix(loss=f"{run_loss/n_seen:.4f}", acc=f"{run_acc/n_seen:.3f}",
                                 lr=f"{sched.get_last_lr()[0]:.2e}",
                                 ips=f"{n_seen/(time.time()-t0):.0f}")
        train_time = time.time() - t0

        val = evaluate(net, tables["val"], args.val_frames_per_video, args, device, use_amp)
        val_auc = val["all"]["video_auc"]
        epoch_times.append(time.time() - t0)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / n_seen, "train_acc": run_acc / n_seen,
               "img_per_s": n_seen / train_time, "epoch_min": epoch_times[-1] / 60, "val": val}
        history.append(rec)
        print(f"[ep {epoch+1}] loss {rec['train_loss']:.4f} acc {rec['train_acc']:.3f} "
              f"{rec['img_per_s']:.0f} img/s {rec['epoch_min']:.1f} min\n   VAL {fmt(val)}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.trainable_state_dict(), "epoch": epoch, "val": val, "args": vars(args)}, best_path)
        torch.save({"model": model.trainable_state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_auc": best_auc, "history": history}, last_path)
        json.dump(history, open(os.path.join(args.out_dir, "history.json"), "w"), indent=1)

    if os.path.isfile(best_path):
        model.load_trainable(torch.load(best_path, map_location="cpu")["model"])
        res = evaluate(net, tables["test"], args.test_frames_per_video, args, device, use_amp)
        print("TEST (best val ckpt):", fmt(res))
        json.dump(res, open(os.path.join(args.out_dir, "test_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main(get_args())
