"""make_plots.py - build every chart from the files written by run_eval.py (no GPU needed).

  !python make_plots.py                                   # everything that has data
  !python make_plots.py --histories Baseline=<history.json> UDD=<history.json>
  !python make_plots.py --write_templates                 # empty ablation/sweep JSON files to fill in

Each plot is skipped (with a message) if its input file is missing.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

PALETTE = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]
METHOD_COLORS = {"original": "#16a34a", "Deepfakes": "#2563eb", "Face2Face": "#ea580c",
                 "FaceSwap": "#9333ea", "NeuralTextures": "#dc2626", "CelebDF": "#0891b2"}
PRETTY = {"blur": "Gaussian blur", "block": "Block-wise noise", "contrast": "Change contrast",
          "saturation": "Change saturation", "jpeg": "JPEG compression"}
GROUPS = [("all", "All test data"), ("ffpp", "FF++"), ("celebdf", "Celeb-DF")]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/kaggle/working/udd_eval_out", help="where run_eval.py wrote its files")
    p.add_argument("--plot_dir", default=None, help="default: <out_dir>/plots")
    p.add_argument("--histories", nargs="*", default=[
        "Baseline=/kaggle/input/datasets/shlokdivyam/princeh01/kaggle/working/udd_baseline/history.json",
        "UDD=/kaggle/working/udd_full/history.json"], help="name=history.json")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--warmup_epochs", type=float, default=1)
    p.add_argument("--ablation_json", default=None, help='{"row": {"metric": value}} (see --write_templates)')
    p.add_argument("--sweeps_json", default=None, help='{"gamma": {"0.1": v, ...}} (see --write_templates)')
    p.add_argument("--write_templates", action="store_true")
    return p.parse_args(argv)


def color_of(names):
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def save(fig, args, name):
    path = os.path.join(args.plot_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


def video_df(df):
    return df.groupby("video").agg(prob=("prob", "mean"), label=("label", "first"), ds=("ds", "first")).reset_index()


def sub_of(df, g):
    return df if g == "all" else df[df.ds == g]


def load_preds(out_dir):
    preds = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "preds_*.csv"))):
        preds[os.path.basename(f)[6:-4]] = pd.read_csv(f)
    return preds


def load_metrics(out_dir):
    return {os.path.basename(f)[8:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, "metrics_*.json")))}


# ----------------------------------------------------------------------------- charts from preds / metrics
def plot_roc(preds, args):
    col = color_of(preds)
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
    for r, level in enumerate(["frame", "video"]):
        for c, (g, title) in enumerate(GROUPS):
            ax = axs[r, c]
            for n, df in preds.items():
                d = sub_of(df, g)
                d = d if level == "frame" else video_df(d)
                if len(d) and d.label.nunique() == 2:
                    fpr, tpr, _ = roc_curve(d.label, d.prob)
                    ax.plot(fpr, tpr, color=col[n], label=f"{n} (AUC {100*sk_auc(fpr, tpr):.2f})")
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_title(f"{title} - {level}-level"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right", fontsize=8)
    save(fig, args, "01_roc_curves.png")
    # zoomed (log-x) version: near-perfect curves are indistinguishable on linear axes
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for c, (g, title) in enumerate(GROUPS):
        for n, df in preds.items():
            d = video_df(sub_of(df, g))
            if len(d) and d.label.nunique() == 2:
                fpr, tpr, _ = roc_curve(d.label, d.prob)
                axs[c].plot(np.maximum(fpr, 1e-4), tpr, color=col[n], label=n)
        axs[c].set_xscale("log"); axs[c].set_title(f"{title} - video-level (log FPR)")
        axs[c].set_xlabel("False positive rate (log)"); axs[c].set_ylabel("True positive rate"); axs[c].legend(fontsize=8)
    save(fig, args, "01b_roc_curves_logx.png")


def plot_confusion(preds, args):
    names = list(preds)
    fig, axs = plt.subplots(2, len(names), figsize=(3.6 * len(names), 6.4), squeeze=False)
    for c, n in enumerate(names):
        for r, level in enumerate(["frame", "video"]):
            d = preds[n] if level == "frame" else video_df(preds[n])
            pred = (d.prob > 0.5).astype(int)
            cm = np.array([[((d.label == i) & (pred == j)).sum() for j in (0, 1)] for i in (0, 1)])
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax = axs[r, c]
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]}\n{100*norm[i, j]:.1f}%", ha="center", va="center",
                            color="white" if norm[i, j] > 0.5 else "black")
            ax.set_xticks([0, 1], ["pred real", "pred fake"]); ax.set_yticks([0, 1], ["real", "fake"])
            ax.set_title(f"{n} - {level}-level (thr 0.5)")
    save(fig, args, "02_confusion_matrices.png")


def plot_histograms(preds, args):
    names = list(preds)
    fig, axs = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.6), squeeze=False)
    for c, n in enumerate(names):
        v = video_df(preds[n]); ax = axs[0, c]
        bins = np.linspace(0, 1, 26)
        ax.hist(v[v.label == 0].prob, bins=bins, alpha=0.65, color="#16a34a", label="real videos")
        ax.hist(v[v.label == 1].prob, bins=bins, alpha=0.65, color="#dc2626", label="fake videos")
        ax.set_yscale("log"); ax.axvline(0.5, color="k", ls=":", lw=0.8)
        ax.set_title(f"{n}: video-level P(fake)"); ax.set_xlabel("mean predicted P(fake)"); ax.set_ylabel("# videos (log)")
        ax.legend(fontsize=8)
    save(fig, args, "03_score_histograms.png")


def plot_pr_calibration(preds, args):
    col = color_of(preds)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for n, df in preds.items():
        v = video_df(df)
        p, r, _ = precision_recall_curve(v.label, v.prob)
        axs[0].plot(r, p, color=col[n], label=f"{n} (AP {100*np.trapezoid(p[::-1], r[::-1]):.2f})")
        bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(df.prob, bins) - 1, 0, 9)
        g = pd.DataFrame({"i": idx, "p": df.prob, "y": df.label}).groupby("i").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        axs[1].plot(g.p, g.y, "o-", color=col[n], label=n, ms=4)
    axs[0].set_title("Precision-recall (video-level)"); axs[0].set_xlabel("Recall"); axs[0].set_ylabel("Precision"); axs[0].legend(fontsize=8)
    axs[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    axs[1].set_title("Calibration (frame-level)"); axs[1].set_xlabel("mean predicted P(fake)"); axs[1].set_ylabel("fraction actually fake"); axs[1].legend(fontsize=8)
    save(fig, args, "04_precision_recall_calibration.png")


def plot_per_method(metrics, args):
    names = [n for n in metrics if metrics[n].get("per_method")]
    if not names:
        print("skip per-method plot (no per-method AUC in metrics)"); return
    col = color_of(names)
    methods = list(metrics[names[0]]["per_method"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in [(axs[0], "frame_auc", "frame-level"), (axs[1], "video_auc", "video-level")]:
        w = 0.8 / len(names); lo = 1.0
        for k, n in enumerate(names):
            vals = [100 * metrics[n]["per_method"][m][key] for m in methods]; lo = min(lo, min(vals))
            bars = ax.bar(np.arange(len(methods)) + k * w - 0.4 + w / 2, vals, w, color=col[n], label=n)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(methods)), methods); ax.set_ylim(max(50, lo - 3), 100.5)
        ax.set_ylabel("AUC (%)"); ax.set_title(f"FF++ per-forgery-method AUC ({title}, real vs method)"); ax.legend(fontsize=8)
    save(fig, args, "05_per_method_auc.png")


def plot_summary(metrics, args):
    names = list(metrics)
    col = color_of(names)
    items = [("Frame AUC", "frame", "auc"), ("Video AUC", "video", "auc"), ("Frame bal. acc", "frame", "bal_acc"),
             ("Frame F1", "frame", "f1")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (g, title) in zip(axs, GROUPS):
        w = 0.8 / len(names); lo = 1.0
        for k, n in enumerate(names):
            if g not in metrics[n]:
                continue
            vals = [100 * metrics[n][g][lvl][key] for _, lvl, key in items]; lo = min(lo, min(vals))
            bars = ax.bar(np.arange(len(items)) + k * w - 0.4 + w / 2, vals, w, color=col[n], label=n)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(items)), [i[0] for i in items], fontsize=8); ax.set_ylim(max(50, lo - 3), 100.5)
        ax.set_title(title); ax.set_ylabel("%"); ax.legend(fontsize=8, loc="lower right")
    save(fig, args, "06_summary_bars.png")


def write_summary_table(metrics, args):
    rows = []
    for n, m in metrics.items():
        for g, title in GROUPS:
            if g not in m:
                continue
            f, v = m[g]["frame"], m[g]["video"]
            rows.append({"Model": n, "Data": title, "Frame AUC": 100 * f["auc"], "Video AUC": 100 * v["auc"],
                         "Frame acc": 100 * f["acc"], "Frame bal.acc": 100 * f["bal_acc"], "Frame F1": 100 * f["f1"],
                         "Frame prec": 100 * f["precision"], "Frame recall": 100 * f["recall"],
                         "Frame EER": 100 * f["eer"], "Video EER": 100 * v["eer"], "Video acc": 100 * v["acc"],
                         "Video F1": 100 * v["f1"], "#frames": f["n"], "#videos": v["n"]})
    if not rows:
        return
    df = pd.DataFrame(rows).round(2)
    df.to_csv(os.path.join(args.plot_dir, "summary_table.csv"), index=False)
    with open(os.path.join(args.plot_dir, "summary_table.md"), "w") as fh:
        fh.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for _, r in df.iterrows():
            fh.write("| " + " | ".join(str(x) for x in r.values) + " |\n")
    print("saved summary_table.csv / .md")


# ----------------------------------------------------------------------------- charts from other run_eval outputs
def plot_robust(args):
    path = os.path.join(args.out_dir, "robust_results.json")
    if not os.path.isfile(path):
        print("skip robustness plot (robust_results.json missing)"); return
    res = json.load(open(path))
    names = list(res["clean"]); col = color_of(names)
    perts = [p for p in PRETTY if p in res["perts"]]
    levels = [1, 2, 3, 4, 5]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5)); axs = axs.ravel()
    avg = {n: [[] for _ in levels] for n in names}
    for ax, p in zip(axs, perts):
        for n in names:
            ys = [100 * res["perts"][p][str(l)][n]["video_auc"] if str(l) in res["perts"][p] else np.nan for l in levels]
            ax.plot(levels, ys, "o-", color=col[n], label=n, ms=4)
            ax.axhline(100 * res["clean"][n]["video_auc"], color=col[n], ls=":", lw=0.8)
            for i, y in enumerate(ys):
                avg[n][i].append(y)
        ax.set_title(PRETTY[p]); ax.set_xlabel("Level (5 = most severe)"); ax.set_ylabel("Video AUC (%)"); ax.set_xticks(levels)
    ax = axs[len(perts)]
    for n in names:
        ax.plot(levels, [float(np.mean([z for z in v if z == z])) if any(z == z for z in v) else np.nan for v in avg[n]], "o-", color=col[n], label=n, ms=4)
    ax.set_title(f"Average over {len(perts)} perturbations"); ax.set_xlabel("Level"); ax.set_ylabel("Video AUC (%)"); ax.set_xticks(levels)
    axs[0].legend(fontsize=8)
    for ax in axs[len(perts) + 1:]:
        ax.axis("off")
    fig.suptitle("Robustness on FF++ test (dotted = clean, same frames)", fontsize=10)
    save(fig, args, "07_robustness_curves.png")


def plot_cutout(args):
    path = os.path.join(args.out_dir, "cutout_results.json")
    if not os.path.isfile(path):
        print("skip cutout plot (cutout_results.json missing)"); return
    res = json.load(open(path)); sizes = sorted(res, key=int)
    names = list(res[sizes[0]]); col = color_of(names)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, key, t in [(axs[0], "frame_auc", "frame-level"), (axs[1], "video_auc", "video-level")]:
        for n in names:
            ys = [100 * res[s][n][key] for s in sizes]
            ax.plot([int(s) for s in sizes], ys, "o-", color=col[n], label=n, ms=4)
            for s, y in zip(sizes, ys):
                ax.annotate(f"{y:.1f}", (int(s), y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6.5, color=col[n])
        ax.set_xticks([int(s) for s in sizes]); ax.set_xlabel("Centre cutout size s (px)"); ax.set_ylabel("AUC (%)")
        ax.set_title(f"Centre-cutout test, FF++ ({t})"); ax.legend(fontsize=8)
    save(fig, args, "08_cutout_curve.png")


def plot_tsne(args):
    path = os.path.join(args.out_dir, "tsne_data.npz")
    if not os.path.isfile(path):
        print("skip t-SNE plot (tsne_data.npz missing)"); return
    d = np.load(path); names = [k[7:] for k in d.files if k.startswith("coords_")]
    meth = d["method"]
    fig, axs = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.6), squeeze=False)
    for ax, n in zip(axs[0], names):
        xy = d[f"coords_{n}"]
        for m in sorted(set(meth), key=lambda z: (z != "original", z)):
            k = (meth == m) & ~np.isnan(xy[:, 0])
            ax.scatter(xy[k, 0], xy[k, 1], s=5, alpha=0.6, color=METHOD_COLORS.get(m, "gray"), label=m)
        ax.set_title(f"t-SNE of [CLS] features - {n}"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axs[0, 0].legend(markerscale=3, fontsize=8)
    save(fig, args, "09_tsne.png")


def plot_histories(args):
    hist = {}
    for s in args.histories or []:
        if "=" not in s:
            print(f"skip --histories entry '{s}' (expected name=path/to/history.json)")
            continue
        n, p = s.split("=", 1)
        if os.path.isfile(p):
            hist[n] = json.load(open(p))
        else:
            print(f"skip history '{n}' (not found: {p})")
    if not hist:
        return
    col = color_of(hist)
    # loss curves
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for n, h in hist.items():
        ep = [r["epoch"] for r in h]
        tot = [r.get("train_total", r.get("train_loss")) for r in h]
        axs[0].plot(ep, tot, "o-", color=col[n], label=f"{n} total", ms=3)
    axs[0].set_yscale("log"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Train loss (log)"); axs[0].set_title("Total training loss"); axs[0].legend(fontsize=8)
    comp = [("train_ce_base", "CE original"), ("train_ce_shuf", "CE shuffle"), ("train_ce_mix", "CE mix"),
            ("train_con", "L_con"), ("train_js", "L_align (JS)")]
    for n, h in hist.items():
        if "train_con" not in h[0]:
            continue
        ep = [r["epoch"] for r in h]
        for (k, lab), c in zip(comp, PALETTE):
            axs[1].plot(ep, [max(r[k], 1e-6) for r in h], color=c, label=f"{n}: {lab}")
    axs[1].set_yscale("log"); axs[1].set_xlabel("Epoch"); axs[1].set_title("UDD loss components"); axs[1].legend(fontsize=7)
    save(fig, args, "10_loss_curves.png")
    # accuracy / val AUC
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for n, h in hist.items():
        ep = [r["epoch"] for r in h]
        axs[0].plot(ep, [100 * r["train_acc"] for r in h], "o-", color=col[n], label=n, ms=3)
        axs[1].plot(ep, [100 * r["val"]["all"]["video_auc"] for r in h], "o-", color=col[n], label=n, ms=3)
        axs[2].plot(ep, [100 * r["val"]["all"]["frame_auc"] for r in h], "o-", color=col[n], label=n, ms=3)
    for ax, t in zip(axs, ["Train accuracy (%)", "Validation video AUC (%)", "Validation frame AUC (%)"]):
        ax.set_xlabel("Epoch"); ax.set_title(t); ax.legend(fontsize=8)
    save(fig, args, "11_accuracy_and_val_auc.png")
    # LR schedule + throughput
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    e = np.linspace(0, args.epochs, 400)
    w = args.warmup_epochs
    lr = np.where(e < w, args.lr * (e + 1e-3) / max(w, 1e-9),
                  0.5 * args.lr * (1 + np.cos(np.pi * np.clip((e - w) / max(args.epochs - w, 1e-9), 0, 1))))
    axs[0].plot(e, lr, color="#2563eb"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Learning rate")
    axs[0].set_title(f"LR schedule (warm-up {w:g} ep, cosine; peak {args.lr:g})")
    for n, h in hist.items():
        axs[1].plot([r["epoch"] for r in h], [r["img_per_s"] for r in h], "o-", color=col[n], label=n, ms=3)
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Training images / s"); axs[1].set_title("Training throughput"); axs[1].legend(fontsize=8)
    save(fig, args, "12_lr_and_throughput.png")


# ----------------------------------------------------------------------------- optional: ablations / sweeps
def plot_ablation(args):
    if not args.ablation_json:
        return
    d = {k: v for k, v in json.load(open(args.ablation_json)).items() if not k.startswith("_")}
    d = {k: {m: x for m, x in v.items() if x is not None} for k, v in d.items()}
    d = {k: v for k, v in d.items() if v}
    if not d:
        print("skip ablation plot (no values filled in)"); return
    metrics = list(next(iter(d.values())))
    scale = 100 if max(x for v in d.values() for x in v.values()) <= 1.0 else 1
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(d)), 4))
    w = 0.8 / len(metrics); lo = 100
    for k, m in enumerate(metrics):
        vals = [scale * d[r].get(m, np.nan) for r in d]; lo = min(lo, np.nanmin(vals))
        bars = ax.bar(np.arange(len(d)) + k * w - 0.4 + w / 2, vals, w, label=m, color=PALETTE[k % len(PALETTE)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(d)), list(d), rotation=20, ha="right"); ax.set_ylim(max(50, lo - 3), 100.5)
    ax.set_ylabel("AUC (%)"); ax.set_title("Ablation (each bar = its own training run)"); ax.legend(fontsize=8)
    save(fig, args, "13_ablation_bars.png")


def plot_sweeps(args):
    if not args.sweeps_json:
        return
    d = {k: {x: y for x, y in v.items() if y is not None} for k, v in json.load(open(args.sweeps_json)).items() if not k.startswith("_")}
    d = {k: v for k, v in d.items() if v}
    if not d:
        print("skip sweep plot (no values filled in)"); return
    fig, axs = plt.subplots(1, len(d), figsize=(3.6 * len(d), 3.4), squeeze=False)
    for ax, (k, v) in zip(axs[0], d.items()):
        xs = list(v)
        try:
            xv = [float(x) for x in xs]; order = np.argsort(xv); xs = [xs[i] for i in order]
        except ValueError:
            pass
        ys = [(100 if max(v.values()) <= 1 else 1) * v[x] for x in xs]
        ax.plot(range(len(xs)), ys, "o-", color="#2563eb")
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.2f}", (i, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
        ax.set_xticks(range(len(xs)), xs); ax.set_title(k); ax.set_ylabel("AUC (%)")
    save(fig, args, "14_hyperparameter_sweeps.png")


def write_templates(args):
    os.makedirs(args.plot_dir, exist_ok=True)
    abl = {"_note": "fill in video-level AUC (0-1 or %) from each separate training run; null = skip",
           "Baseline (frozen CLIP + FC)": {"FF++ video AUC": None, "Celeb-DF video AUC": None},
           "+ LoRA only": {"FF++ video AUC": None, "Celeb-DF video AUC": None},
           "w/o shuffling branch": {"FF++ video AUC": None, "Celeb-DF video AUC": None},
           "w/o mixing branch": {"FF++ video AUC": None, "Celeb-DF video AUC": None},
           "w/o L_con": {"FF++ video AUC": None, "Celeb-DF video AUC": None},
           "w/o L_align": {"FF++ video AUC": None, "Celeb-DF video AUC": None},
           "Full UDD": {"FF++ video AUC": None, "Celeb-DF video AUC": None}}
    swp = {"_note": "x-value -> AUC from separate runs; null = skip",
           "mix stage (Table 4)": {"before first": None, "early": None, "mid": None, "late": None},
           "gamma (mixing ratio)": {"0.1": None, "0.3": None, "0.5": None, "0.7": None},
           "lambda1 (L_con)": {"0": None, "0.1": None, "0.5": None, "1": None},
           "lambda2 (L_align)": {"0": None, "0.1": None, "0.5": None, "1": None},
           "tau (temperature)": {"0.05": None, "0.1": None, "0.15": None, "0.2": None},
           "shuffle blocks s": {"1": None, "2": None, "7": None, "14": None}}
    json.dump(abl, open(os.path.join(args.plot_dir, "ablation_template.json"), "w"), indent=2)
    json.dump(swp, open(os.path.join(args.plot_dir, "sweeps_template.json"), "w"), indent=2)
    print("wrote ablation_template.json and sweeps_template.json in", args.plot_dir)


def main(args):
    args.plot_dir = args.plot_dir or os.path.join(args.out_dir, "plots")
    os.makedirs(args.plot_dir, exist_ok=True)
    if args.write_templates:
        write_templates(args)
        return
    preds, metrics = load_preds(args.out_dir), load_metrics(args.out_dir)
    if preds:
        plot_roc(preds, args); plot_confusion(preds, args); plot_histograms(preds, args); plot_pr_calibration(preds, args)
    else:
        print("no preds_*.csv found - run  run_eval.py --stage metrics  first")
    if metrics:
        plot_per_method(metrics, args); plot_summary(metrics, args); write_summary_table(metrics, args)
    plot_robust(args); plot_cutout(args); plot_tsne(args); plot_histories(args)
    plot_ablation(args); plot_sweeps(args)
    print("\nDone. Plots in", args.plot_dir)


if __name__ == "__main__":
    main(get_args())
