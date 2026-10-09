"""train_udd.py -- Full UDD training, corrected."""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from data        import FrameDataset, load_split_table, normalize_on_device, sanity_check_files, select_frames
from model       import UDDViT
from shuf_branch import shuf_forward
from mix_branch  import mix_forward
from losses      import UDDLoss


class UDDStep(nn.Module):
    """Runs the original, shuffling and mixing branches on one chunk of the batch.
    Wrapped in nn.DataParallel so each GPU processes its own slice; every branch shares model's weights."""
    def __init__(self, model, mix_ratio, layer_range, pos_crop):
        super().__init__()
        self.model, self.mix_ratio, self.layer_range, self.pos_crop = model, mix_ratio, layer_range, pos_crop

    def forward(self, x, y):
        logits_base, feat_orig = self.model(x)
        logits_shuf, feat_shuf = shuf_forward(self.model, x, pos_crop=self.pos_crop)
        logits_mix, feat_mix, _ = mix_forward(self.model, x, y, ratio=self.mix_ratio, layer_range=self.layer_range)
        return logits_base, feat_orig, logits_shuf, feat_shuf, logits_mix, feat_mix


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    default="/kaggle/input/datasets/shlokdivyam/udd-dataset")
    p.add_argument("--split_json",   default="/kaggle/input/datasets/shlokdivyam/udd-dataset/dataset_split_relative.json")
    p.add_argument("--out_dir",      default="/kaggle/working/udd_full")
    p.add_argument("--img_size",     type=int,   default=224)
    p.add_argument("--train_frames_per_video", type=int, default=8)
    p.add_argument("--frame_sampling", choices=["even","random"], default="even")
    p.add_argument("--val_frames_per_video",   type=int, default=8)
    p.add_argument("--test_frames_per_video",  type=int, default=32)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--bad_list",     default="/kaggle/input/datasets/shlokdivyam/princeh01/kaggle/working/bad_files.txt")
    p.add_argument("--debug_videos", type=int,   default=0)
    p.add_argument("--arch",         default="vit_base_patch16_clip_224.openai")
    p.add_argument("--weights_file", default=None)
    p.add_argument("--no_pretrained",action="store_true")
    p.add_argument("--lora_rank",    type=int,   default=4)
    p.add_argument("--lora_alpha",   type=float, default=None)
    p.add_argument("--grad_ckpt",    action="store_true", default=True)
    p.add_argument("--mix_ratio",    type=float, default=0.30)
    p.add_argument("--mix_layer_min", type=int,  default=5, help="mix after a random block in [min,max] (1-indexed; paper: mid stage)")
    p.add_argument("--mix_layer_max", type=int,  default=8)
    p.add_argument("--no_pos_crop",  action="store_true", help="ablation: disable the random position-embedding crop")
    p.add_argument("--proj_dim",     type=int,   default=128)
    p.add_argument("--temperature",  type=float, default=0.10)
    p.add_argument("--lambda1",      type=float, default=0.10)
    p.add_argument("--lambda2",      type=float, default=0.10)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64, help="EFFECTIVE batch size (paper: 64)")
    p.add_argument("--per_gpu_batch", type=int,  default=32, help="images per GPU per forward pass (memory knob)")
    p.add_argument("--accum_steps",  type=int,   default=0, help="0 = auto: batch_size // (per_gpu_batch * n_gpus)")
    p.add_argument("--no_dp",        action="store_true", help="disable multi-GPU DataParallel")
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--betas",        type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--eps",          type=float, default=1e-3)
    p.add_argument("--warmup_epochs",type=float, default=5)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--no_amp",       action="store_true")
    p.add_argument("--time_budget_hours", type=float, default=11.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval_only",    action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def predict(model, table, args, device, use_amp):
    ds = FrameDataset(table, args.data_root, args.img_size, train=False,
                      bad_log=os.path.join(args.out_dir, "bad_files_runtime.txt"))
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=args.num_workers,
                    pin_memory=device.type=="cuda")
    model.eval()
    probs = np.full(len(ds), np.nan, dtype=np.float32)
    for x, _, idx in tqdm(dl, desc="eval", leave=False):
        x = normalize_on_device(x, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(x)
        probs[idx.numpy()] = F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
    return probs


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def compute_metrics(table, probs):
    t = table.copy(); t["p"] = probs; t = t[~np.isnan(probs)]
    out = {}
    for name, sub in [("all",t),("ffpp",t[t.ds=="ffpp"]),("celebdf",t[t.ds=="celebdf"])]:
        if not len(sub): continue
        v = sub.groupby("video").agg(p=("p","mean"), y=("label","first"))
        out[name] = {
            "frame_auc": _auc(sub.label.values, sub.p.values),
            "video_auc": _auc(v.y.values, v.p.values),
            "frame_acc": float(((sub.p.values>0.5)==sub.label.values).mean()),
            "n_frames": int(len(sub)), "n_videos": int(len(v)),
        }
    return out


def fmt(m):
    return " | ".join(
        f"{k}: vAUC {v['video_auc']*100:.2f} fAUC {v['frame_auc']*100:.2f}"
        for k,v in m.items()
    )


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type=="cuda" and not args.no_amp
    n_gpu   = torch.cuda.device_count() if (device.type == "cuda" and not args.no_dp) else 1
    micro   = args.per_gpu_batch * max(1, n_gpu)                  # images per optimizer micro-step
    accum   = args.accum_steps if args.accum_steps > 0 else max(1, args.batch_size // micro)
    eff_bs  = micro * accum
    print(f"GPUs: {n_gpu} | per-GPU batch {args.per_gpu_batch} | micro-batch {micro} | accum {accum} | "
          f"effective batch {eff_bs} | grad_ckpt: {args.grad_ckpt}")
    print(f"mix_ratio {args.mix_ratio} | mix after block {args.mix_layer_min}-{args.mix_layer_max} | pos_crop {not args.no_pos_crop}")

    if not os.path.isfile(args.split_json):
        raise FileNotFoundError(args.split_json)
    tables = {s: load_split_table(args.split_json, s) for s in ("train","val","test")}
    if args.bad_list and os.path.isfile(args.bad_list):
        bad = {l.split("\t")[0].strip() for l in open(args.bad_list) if l.strip()}
        for s in tables:
            tables[s] = tables[s][~tables[s].path.isin(bad)].reset_index(drop=True)
        print(f"Dropped {len(bad)} bad paths")
    if args.debug_videos:
        for s in tables:
            keep = tables[s]["video"].drop_duplicates().sample(
                min(args.debug_videos, tables[s]["video"].nunique()), random_state=0)
            tables[s] = tables[s][tables[s].video.isin(keep)].reset_index(drop=True)
    sanity_check_files(tables["train"], args.data_root)
    for s,t in tables.items():
        print(f"{s}: {len(t)} frames  {t.video.nunique()} videos  fake {t.label.mean():.3f}")

    model = UDDViT(args.arch, not args.no_pretrained, args.weights_file,
                   args.lora_rank, args.lora_alpha, 2, args.grad_ckpt).to(device)
    tr, tot = model.param_summary()
    embed_dim = model.vit.embed_dim
    print(f"Trainable: {tr/1e6:.3f}M / {tot/1e6:.2f}M  embed_dim: {embed_dim}")

    cw = None
    if args.class_weight:
        freq = np.bincount(tables["train"].label.values, minlength=2).astype(np.float64)
        cw   = torch.tensor(freq.sum()/(2*freq), dtype=torch.float32, device=device)
    criterion = UDDLoss(
        embed_dim=embed_dim, proj_dim=args.proj_dim, head_fc=model.head_fc,
        temperature=args.temperature, lambda1=args.lambda1, lambda2=args.lambda2,
        class_weight=cw,
    ).to(device)

    params = (
        [p for p in model.parameters() if p.requires_grad] +
        list(criterion.projector.parameters())
    )
    opt = torch.optim.AdamW(params, lr=args.lr, betas=tuple(args.betas),
                             eps=args.eps, weight_decay=args.weight_decay)

    step_mod = UDDStep(model, args.mix_ratio, (args.mix_layer_min, args.mix_layer_max), not args.no_pos_crop).to(device)
    net      = nn.DataParallel(step_mod) if n_gpu > 1 else step_mod
    eval_net = nn.DataParallel(model)    if n_gpu > 1 else model

    n_train         = len(select_frames(tables["train"], args.train_frames_per_video, args.frame_sampling))
    steps_per_epoch = (n_train // micro) // accum
    total_steps     = steps_per_epoch * args.epochs
    warm            = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warm: return (step+1)/max(1,warm)
        prog = (step-warm)/max(1,total_steps-warm)
        return 0.5*(1+math.cos(math.pi*min(1.0,prog)))

    sched  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    if args.eval_only:
        ck = torch.load(best_path, map_location="cpu")
        model.load_trainable(ck["model"])
        t  = select_frames(tables["test"], args.test_frames_per_video, mode="even")
        res = compute_metrics(t, predict(eval_net, t, args, device, use_amp))
        print("TEST", fmt(res))
        json.dump(res, open(os.path.join(args.out_dir,"test_results.json"),"w"), indent=2)
        return

    start_epoch, best_auc, history = 0, -1.0, []
    if os.path.isfile(last_path):
        ck = torch.load(last_path, map_location="cpu")
        model.load_trainable(ck["model"])
        criterion.projector.load_state_dict(ck["projector"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_auc, history = ck["epoch"]+1, ck["best_auc"], ck["history"]
        print(f"Resumed from epoch {start_epoch}  (best val AUC {best_auc:.4f})")

    t_start, epoch_times = time.time(), []
    for epoch in range(start_epoch, args.epochs):
        if epoch_times and (time.time()-t_start+max(epoch_times))/3600 > args.time_budget_hours:
            print("Time budget reached -- rerun same command to resume."); break

        t0    = time.time()
        table = select_frames(tables["train"], args.train_frames_per_video,
                              args.frame_sampling, seed=args.seed, epoch=epoch)
        dl    = DataLoader(
            FrameDataset(table, args.data_root, args.img_size, train=True,
                         bad_log=os.path.join(args.out_dir,"bad_files_runtime.txt")),
            batch_size=micro, shuffle=True, drop_last=True,
            num_workers=args.num_workers, pin_memory=device.type=="cuda")

        net.train(); criterion.train()
        run = {"total":0.0,"ce_b":0.0,"ce_s":0.0,"ce_m":0.0,"con":0.0,"js":0.0,"acc":0.0}
        n_seen = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")

        for it, (x, y, _) in enumerate(pbar):
            x = normalize_on_device(x, device); y = y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits_base, feat_orig, logits_shuf, feat_shuf, logits_mix, feat_mix = net(x, y)
                losses = criterion(logits_base, logits_shuf, logits_mix,
                                   feat_orig, feat_shuf, feat_mix, y)
            scaler.scale(losses["loss"] / accum).backward()
            if (it+1) % accum == 0:
                if args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            B = y.size(0)
            run["total"] += losses["loss"].item()    * B
            run["ce_b"]  += losses["ce_base"].item() * B
            run["ce_s"]  += losses["ce_shuf"].item() * B
            run["ce_m"]  += losses["ce_mix"].item()  * B
            run["con"]   += losses["l_con"].item()   * B
            run["js"]    += losses["l_js"].item()    * B
            run["acc"]   += (logits_base.argmax(1)==y).sum().item()
            n_seen += B
            if it % 20 == 0:
                pbar.set_postfix(
                    total=f'{losses["loss"].item():.3f}',
                    ce_b =f'{losses["ce_base"].item():.3f}',
                    ce_s =f'{losses["ce_shuf"].item():.3f}',
                    con  =f'{losses["l_con"].item():.3f}',
                    js   =f'{losses["l_js"].item():.3f}',
                    acc  =f'{run["acc"]/n_seen:.3f}',
                    lr   =f'{sched.get_last_lr()[0]:.2e}')

        train_time = time.time()-t0
        val_t = select_frames(tables["val"], args.val_frames_per_video, mode="even")
        val   = compute_metrics(val_t, predict(eval_net, val_t, args, device, use_amp))
        val_auc = val["all"]["video_auc"]
        epoch_times.append(time.time()-t0)
        rec = {"epoch":epoch+1, "train_total":run["total"]/n_seen,
               "train_ce_base":run["ce_b"]/n_seen, "train_ce_shuf":run["ce_s"]/n_seen,
               "train_ce_mix":run["ce_m"]/n_seen,  "train_con":run["con"]/n_seen,
               "train_js":run["js"]/n_seen, "train_acc":run["acc"]/n_seen,
               "img_per_s":n_seen/train_time, "epoch_min":epoch_times[-1]/60, "val":val}
        history.append(rec)
        print(f"[ep {epoch+1}] total {rec['train_total']:.3f}  "
              f"ce_b {rec['train_ce_base']:.3f}  ce_s {rec['train_ce_shuf']:.3f}  "
              f"con {rec['train_con']:.3f}  js {rec['train_js']:.3f}  "
              f"acc {rec['train_acc']:.3f}  {rec['img_per_s']:.0f} img/s  {rec['epoch_min']:.1f} min")
        print("   VAL", fmt(val))

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.trainable_state_dict(),
                        "projector": criterion.projector.state_dict(),
                        "epoch":epoch, "val":val, "args":vars(args)}, best_path)
        torch.save({"model": model.trainable_state_dict(),
                    "projector": criterion.projector.state_dict(),
                    "opt":opt.state_dict(), "sched":sched.state_dict(),
                    "scaler":scaler.state_dict(), "epoch":epoch,
                    "best_auc":best_auc, "history":history}, last_path)
        json.dump(history, open(os.path.join(args.out_dir,"history.json"),"w"), indent=1)

    if os.path.isfile(best_path):
        ck = torch.load(best_path, map_location="cpu")
        model.load_trainable(ck["model"])
        test_t = select_frames(tables["test"], args.test_frames_per_video, mode="even")
        res = compute_metrics(test_t, predict(eval_net, test_t, args, device, use_amp))
        print("TEST (best val ckpt):", fmt(res))
        json.dump(res, open(os.path.join(args.out_dir,"test_results.json"),"w"), indent=2)


if __name__ == "__main__":
    main(get_args())
