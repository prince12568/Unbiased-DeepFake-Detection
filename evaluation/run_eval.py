"""run_eval.py - evaluate one or more trained models (Baseline, UDD, ablations...) in the same way.

Stages (pick any, or 'all'):
  metrics   test-set metrics (frame/video AUC, acc, balanced acc, F1, EER, confusion counts, per-method AUC)
  cutout    centre-region cutout test           (paper Table 11 / Fig. 7)
  robust    5 perturbations x 5 levels          (paper Fig. 4)   - resumable
  tsne      feature-space t-SNE on FF++ test    (paper Fig. 11)
  attention [CLS] attention maps + centre-mass  (paper Fig. 5 / 9)
  gradcam   GradCAM++ on the first block        (paper Fig. 8)

Example (Kaggle):
  !python run_eval.py --stage metrics --models Baseline=/path/best.pt UDD=/path/best.pt
  !python run_eval.py --stage all
Then:  !python make_plots.py
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_core import (LEVELS, PERTURBATIONS, PRETTY, auc_safe, build_model, center_mass, cls_attention,
                       cutout, evaluate_df, gradcam_pp, load_test_table, predict_multi, select_frames,
                       video_table, wrap_dp, normalize_on_device)
from PIL import Image

STAGES = ["metrics", "cutout", "robust", "tsne", "attention", "gradcam"]
DEF_BASE = "/kaggle/input/datasets/shlokdivyam/princeh01/kaggle/working/udd_baseline/best.pt"
DEF_UDD = "/kaggle/working/udd_full/best.pt"


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", nargs="+", default=["metrics"], choices=STAGES + ["all"])
    p.add_argument("--models", nargs="+", default=[f"Baseline={DEF_BASE}", f"UDD={DEF_UDD}"],
                   help="name=path/to/best.pt (any number of models)")
    p.add_argument("--data_root", default="/kaggle/input/datasets/shlokdivyam/udd-dataset")
    p.add_argument("--split_json", default="/kaggle/input/datasets/shlokdivyam/udd-dataset/dataset_split_relative.json")
    p.add_argument("--bad_list", default="/kaggle/input/datasets/shlokdivyam/princeh01/kaggle/working/bad_files.txt")
    p.add_argument("--out_dir", default="/kaggle/working/udd_eval_out")
    p.add_argument("--weights_file", default=None, help="local CLIP weights if Kaggle internet is off")
    p.add_argument("--no_pretrained", action="store_true", help="(testing only) random frozen backbone")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--test_frames_per_video", type=int, default=32)
    p.add_argument("--cutout_sizes", nargs="+", type=int, default=[0, 16, 32, 48, 64])
    p.add_argument("--cutout_frames_per_video", type=int, default=8)
    p.add_argument("--robust_frames_per_video", type=int, default=8, help="paper uses 32; 8 keeps Kaggle time down")
    p.add_argument("--robust_perts", nargs="+", default=PERTURBATIONS, choices=PERTURBATIONS)
    p.add_argument("--robust_levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--tsne_n", type=int, default=2500, help="total FF++ test frames used for t-SNE")
    p.add_argument("--n_vis", type=int, default=4, help="samples per dataset in attention / GradCAM figures")
    p.add_argument("--attn_stat_n", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


class Ctx:
    pass


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def jload(path, default):
    return json.load(open(path)) if os.path.isfile(path) else default


def pct(x):
    return "  nan" if x != x else f"{100 * x:5.2f}"


def predict(ctx, table, cond=("clean", 0), feats=False, desc="eval"):
    return predict_multi(ctx.dp_models, table, ctx.args.data_root, ctx.args.img_size, ctx.device, cond,
                         ctx.args.batch_size, ctx.args.num_workers, ctx.use_amp, feats, desc,
                         bad_log=os.path.join(ctx.args.out_dir, "bad_files_runtime.txt"))


def scored(table, prob):
    df = table.copy()
    df["prob"] = prob
    return df.dropna(subset=["prob"])


def is_readable(root, path):
    try:
        Image.open(os.path.join(root, path)).convert("RGB")
        return True
    except Exception:
        return False


def load_rgb(root, path, size):
    return Image.open(os.path.join(root, path)).convert("RGB").resize((size, size), Image.BICUBIC)


# ----------------------------------------------------------------------------- stages
def stage_metrics(ctx):
    a = ctx.args
    table = select_frames(ctx.test, a.test_frames_per_video, "even")
    print(f"[metrics] {len(table)} test frames, {table.video.nunique()} videos")
    probs, _ = predict(ctx, table, desc="metrics")
    print(f"\n{'model':<12}{'group':<9}{'fAUC':>7}{'vAUC':>7}{'f-acc':>7}{'f-bAcc':>8}{'f-F1':>7}{'v-EER':>7}")
    for name in ctx.models:
        df = scored(table, probs[name])
        df[["path", "video", "ds", "method", "label", "prob"]].to_csv(os.path.join(a.out_dir, f"preds_{name}.csv"), index=False)
        res = evaluate_df(df)
        jdump(res, os.path.join(a.out_dir, f"metrics_{name}.json"))
        for g in ("all", "ffpp", "celebdf"):
            if g in res:
                f, v = res[g]["frame"], res[g]["video"]
                print(f"{name:<12}{g:<9}{pct(f['auc']):>7}{pct(v['auc']):>7}{pct(f['acc']):>7}{pct(f['bal_acc']):>8}"
                      f"{pct(f['f1']):>7}{pct(v['eer']):>7}")
        print(f"{name:<12}per-method video AUC: " + ", ".join(f"{m} {pct(r['video_auc']).strip()}" for m, r in res["per_method"].items()))
    print("saved preds_*.csv and metrics_*.json")


def eval_condition(ctx, table, cond):
    probs, _ = predict(ctx, table, cond=cond, desc=f"{cond[0]}{cond[1]}")
    out = {}
    for name in ctx.models:
        df = scored(table, probs[name])
        v = video_table(df)
        out[name] = {"frame_auc": auc_safe(df.label, df.prob), "video_auc": auc_safe(v.label, v.prob),
                     "n_frames": int(len(df))}
    return out


def stage_cutout(ctx):
    a = ctx.args
    ff = ctx.test[ctx.test.ds == "ffpp"]
    table = select_frames(ff, a.cutout_frames_per_video, "even")
    print(f"[cutout] FF++ test: {len(table)} frames, sizes {a.cutout_sizes}")
    path = os.path.join(a.out_dir, "cutout_results.json")
    res = jload(path, {})
    for s in a.cutout_sizes:
        if str(s) in res:
            continue
        res[str(s)] = eval_condition(ctx, table, ("cutout", s))
        jdump(res, path)
        print(f"  s={s:<3} " + "  ".join(f"{n}: fAUC {pct(r['frame_auc'])}" for n, r in res[str(s)].items()))
    # example strip (paper Fig. 7)
    row = table[table.label == 1].iloc[0]
    img = np.array(load_rgb(a.data_root, row.path, a.img_size))
    fig, axs = plt.subplots(1, len(a.cutout_sizes), figsize=(2.1 * len(a.cutout_sizes), 2.4))
    for ax, s in zip(np.atleast_1d(axs), a.cutout_sizes):
        ax.imshow(cutout(img, s)); ax.set_title(f"s={s}", fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "cutout_examples.png"), dpi=130); plt.close(fig)


def stage_robust(ctx):
    a = ctx.args
    ff = ctx.test[ctx.test.ds == "ffpp"]
    table = select_frames(ff, a.robust_frames_per_video, "even")
    n_cond = len(a.robust_perts) * len(a.robust_levels)
    print(f"[robust] FF++ test: {len(table)} frames x {n_cond} conditions (+clean) x {len(ctx.models)} models")
    path = os.path.join(a.out_dir, "robust_results.json")
    res = jload(path, {"clean": None, "perts": {}})
    if res["clean"] is None:
        res["clean"] = eval_condition(ctx, table, ("clean", 0)); jdump(res, path)
    t0, done = time.time(), 0
    for kind in a.robust_perts:
        for lvl in a.robust_levels:
            d = res["perts"].setdefault(kind, {})
            if str(lvl) in d:
                continue
            d[str(lvl)] = eval_condition(ctx, table, (kind, lvl))
            jdump(res, path); done += 1
            print(f"  {PRETTY[kind]:<18} level {lvl} (param {LEVELS[kind][lvl-1]}): " +
                  "  ".join(f"{n}: vAUC {pct(r['video_auc'])}" for n, r in d[str(lvl)].items()) +
                  f"   [{(time.time()-t0)/60:.1f} min]")


def stage_tsne(ctx):
    from sklearn.manifold import TSNE
    a = ctx.args
    ff = ctx.test[ctx.test.ds == "ffpp"]
    methods = sorted(ff.method.unique())
    per = max(1, a.tsne_n // len(methods))
    rng = np.random.default_rng(a.seed)
    parts = [g.iloc[rng.permutation(len(g))[:per]] for _, g in ff.groupby("method")]
    table = pd.concat(parts).reset_index(drop=True)
    print(f"[tsne] {len(table)} FF++ test frames ({per} per class: {methods})")
    _, feats = predict(ctx, table, feats=True, desc="features")
    out = {"method": np.array(table.method.tolist(), dtype="U32"), "label": table.label.values.astype(int)}
    ok = ~np.isnan(feats[next(iter(ctx.models))]).any(1)
    for name in ctx.models:
        Z = feats[name]
        perp = float(min(30, max(2, ok.sum() // 4)))
        emb = np.full((len(Z), 2), np.nan, np.float32)
        emb[ok] = TSNE(n_components=2, perplexity=perp, init="pca", random_state=a.seed).fit_transform(Z[ok])
        out[f"coords_{name}"] = emb
        print(f"  t-SNE done for {name}")
    np.savez(os.path.join(a.out_dir, "tsne_data.npz"), **out)


def pick_samples(ctx):
    """n_vis fake frames per dataset + n_vis/2 real + n_vis/2 fake FF++ frames for real-vs-fake figure."""
    a = ctx.args
    rng = np.random.default_rng(a.seed)
    t = ctx.test

    def take(df, n):  # random rows, skipping unreadable images
        keep = []
        for i in rng.permutation(len(df)):
            if len(keep) == n:
                break
            if is_readable(a.data_root, df.path.iloc[i]):
                keep.append(i)
        return df.iloc[keep]
    fake = pd.concat([take(t[(t.ds == d) & (t.label == 1)], a.n_vis) for d in ("ffpp", "celebdf")])
    ffr = take(t[(t.ds == "ffpp") & (t.label == 0)], max(1, a.n_vis // 2))
    fff = take(t[(t.ds == "ffpp") & (t.label == 1)], max(1, a.n_vis // 2))
    return fake.reset_index(drop=True), pd.concat([ffr, fff]).reset_index(drop=True)


def batch_tensor(ctx, rows):
    imgs = [np.array(load_rgb(ctx.args.data_root, p, ctx.args.img_size)) for p in rows.path]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)
    return imgs, normalize_on_device(x, ctx.device)


def stage_attention(ctx):
    a = ctx.args
    fake, pair = pick_samples(ctx)
    # --- figure A: fake samples, mean map + 3 sharpest heads per model (paper Fig. 5)
    imgs, x = batch_tensor(ctx, fake)
    maps = {n: cls_attention(m, x).float().cpu() for n, m in ctx.models.items()}
    M = len(ctx.models)
    ncol = 1 + 4 * M
    fig, axs = plt.subplots(len(fake), ncol, figsize=(1.45 * ncol, 1.5 * len(fake)))
    for i in range(len(fake)):
        axs[i, 0].imshow(imgs[i]); axs[i, 0].axis("off")
        if i == 0: axs[i, 0].set_title("image", fontsize=8)
        for k, (n, mp) in enumerate(maps.items()):
            heads = mp[i]                                   # (H, g, g)
            top = heads.flatten(1).max(1)[0].argsort(descending=True)[:3]
            cols = [heads.mean(0)] + [heads[h] for h in top]
            names = ["mean"] + [f"head {int(h)}" for h in top]
            for j, (c, nm) in enumerate(zip(cols, names)):
                ax = axs[i, 1 + 4 * k + j]
                ax.imshow(c.numpy(), cmap="viridis"); ax.axis("off")
                if i == 0: ax.set_title(f"{n}\n{nm}", fontsize=7)
    fig.suptitle("Last-layer [CLS] attention (fake frames; heads shown = 3 with highest peak)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "attention_maps_fake.png"), dpi=130); plt.close(fig)
    # --- figure B: real vs fake (paper Fig. 9)
    imgs, x = batch_tensor(ctx, pair)
    maps = {n: cls_attention(m, x).float().cpu().mean(1) for n, m in ctx.models.items()}
    fig, axs = plt.subplots(len(pair), 1 + M, figsize=(1.7 * (1 + M), 1.7 * len(pair)))
    for i in range(len(pair)):
        axs[i, 0].imshow(imgs[i]); axs[i, 0].axis("off")
        axs[i, 0].set_ylabel("real" if pair.label[i] == 0 else "fake")
        axs[i, 0].set_title("real" if pair.label[i] == 0 else "fake", fontsize=8)
        for k, (n, mp) in enumerate(maps.items()):
            axs[i, 1 + k].imshow(mp[i].numpy(), cmap="viridis"); axs[i, 1 + k].axis("off")
            if i == 0: axs[i, 1 + k].set_title(n, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "attention_maps_real_vs_fake.png"), dpi=130); plt.close(fig)
    # --- quantitative position-bias proxy (extra, not in the paper)
    t = ctx.test[ctx.test.label == 1]
    rng = np.random.default_rng(a.seed + 1)
    stat = t.iloc[rng.permutation(len(t))[:int(a.attn_stat_n * 1.2) + 8]]
    res = {n: [] for n in ctx.models}
    kept = 0
    for s in range(0, len(stat), 64):
        chunk = stat.iloc[s:s + 64]
        chunk = chunk[[is_readable(a.data_root, p) for p in chunk.path]].iloc[:max(0, a.attn_stat_n - kept)]
        if not len(chunk):
            continue
        kept += len(chunk)
        _, x = batch_tensor(ctx, chunk)
        for n, m in ctx.models.items():
            res[n].extend(center_mass(cls_attention(m, x).float().mean(1)).tolist())
    g = cls_attention(next(iter(ctx.models.values())), x[:1]).shape[-1]
    out = {n: {"center_mass_mean": float(np.mean(v)), "n": len(v), "uniform_expected": 36 / g ** 2}
           for n, v in res.items()}
    jdump(out, os.path.join(a.out_dir, "attention_stats.json"))
    print("[attention] share of [CLS] attention inside the central 6x6 patches (uniform would be "
          f"{100*36/g**2:.1f}%):")
    for n, r in out.items():
        print(f"   {n:<12}{100*r['center_mass_mean']:.1f}%")


def stage_gradcam(ctx):
    a = ctx.args
    fake, pair = pick_samples(ctx)
    rows = pd.concat([fake, pair[pair.label == 0]]).reset_index(drop=True)
    imgs, x = batch_tensor(ctx, rows)
    cams = {n: gradcam_pp(m, x.float()) for n, m in ctx.models.items()}
    M = len(ctx.models)
    fig, axs = plt.subplots(len(rows), 1 + M, figsize=(1.9 * (1 + M), 1.9 * len(rows)))
    for i in range(len(rows)):
        axs[i, 0].imshow(imgs[i]); axs[i, 0].axis("off")
        axs[i, 0].set_title(("fake" if rows.label[i] else "real") + f" ({rows.ds[i]})", fontsize=7)
        for k, (n, cm) in enumerate(cams.items()):
            up = torch.nn.functional.interpolate(cm[i][None, None], size=imgs[i].shape[:2], mode="bilinear",
                                                 align_corners=False)[0, 0].numpy()
            axs[i, 1 + k].imshow(imgs[i]); axs[i, 1 + k].imshow(up, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axs[i, 1 + k].axis("off")
            if i == 0: axs[i, 1 + k].set_title(n, fontsize=8)
    fig.suptitle("GradCAM++ on the first block (target = 'fake' logit)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "gradcam_heatmaps.png"), dpi=130); plt.close(fig)
    print("[gradcam] saved gradcam_heatmaps.png")


# ----------------------------------------------------------------------------- main
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ctx = Ctx()
    ctx.args = args
    ctx.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx.use_amp = ctx.device.type == "cuda" and not args.no_amp
    specs = [m.split("=", 1) for m in args.models]
    bad = [p for _, p in specs if not os.path.isfile(p)]
    if bad:
        raise FileNotFoundError("checkpoint(s) not found: " + ", ".join(bad) +
                                "\nPass the right paths with  --models Baseline=<path> UDD=<path>")
    ctx.models, ctx.dp_models = {}, {}
    for name, path in specs:
        m, ck = build_model(path, ctx.device, args.weights_file, not args.no_pretrained)
        ctx.models[name] = m
        ctx.dp_models[name] = wrap_dp(m, ctx.device)
        print(f"loaded {name}: {path}  (epoch {ck.get('epoch', '?') + 1 if isinstance(ck.get('epoch'), int) else '?'}, "
              f"lora_rank {(ck.get('args') or {}).get('lora_rank', '?')})")
    ctx.test = load_test_table(args.split_json, args.bad_list)
    print(f"test table: {len(ctx.test)} frames, {ctx.test.video.nunique()} videos, "
          f"datasets {ctx.test.ds.value_counts().to_dict()}")
    stages = STAGES if "all" in args.stage else args.stage
    for s in stages:
        print(f"\n===== {s} =====")
        globals()[f"stage_{s}"](ctx)
    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main(get_args())
