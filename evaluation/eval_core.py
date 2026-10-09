"""eval_core.py - shared helpers for the UDD evaluation (metrics, model loading,
perturbations, one-pass multi-model prediction, attention maps, GradCAM++).

Works with ANY checkpoint produced by train.py / train_udd.py (they all store the
trainable LoRA + FC weights and the training args), so it evaluates the baseline,
the current UDD model and any retrained / ablation model in exactly the same way.
Needs data.py and model.py in the same folder.
"""
import io
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import FrameDataset, load_split_table, normalize_on_device, select_frames  # noqa: F401
from model import UDDViT

# ----------------------------------------------------------------------------- perturbations
# The paper follows DeeperForensics-1.0 (Jiang et al. 2020) for 5 perturbation types x 5 levels.
# The exact per-level values are NOT in the UDD paper; the numbers below are my best recollection
# of the public implementations and are NOT verified. Edit them here if your course gives others.
PERTURBATIONS = ["blur", "block", "contrast", "saturation", "jpeg"]
LEVELS = {
    "saturation": [0.4, 0.3, 0.2, 0.1, 0.0],       # colour factor (1 = unchanged, 0 = grey)
    "contrast": [0.85, 0.725, 0.6, 0.475, 0.35],   # contrast factor
    "block": [16, 32, 48, 64, 80],                 # number of random coloured blocks
    "blur": [7, 9, 13, 17, 21],                    # Gaussian kernel size
    "jpeg": [60, 45, 30, 20, 10],                  # JPEG quality factor
}
PRETTY = {"blur": "Gaussian blur", "block": "Block-wise noise", "contrast": "Change contrast",
          "saturation": "Change saturation", "jpeg": "JPEG compression"}


def cutout(arr, s):
    """Black s x s square at the image centre (paper Table 11 / Fig. 7)."""
    if s <= 0:
        return arr
    a = arr.copy()
    H, W = a.shape[:2]
    y, x = (H - s) // 2, (W - s) // 2
    a[y:y + s, x:x + s] = 0
    return a


def perturb_image(arr, kind, level, rng):
    """arr: HWC uint8. level in 1..5."""
    p = LEVELS[kind][level - 1]
    img = Image.fromarray(arr)
    if kind == "saturation":
        img = ImageEnhance.Color(img).enhance(p)
    elif kind == "contrast":
        img = ImageEnhance.Contrast(img).enhance(p)
    elif kind == "blur":
        sigma = 0.3 * ((p - 1) * 0.5 - 1) + 0.8  # sigma OpenCV derives from a kernel size
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif kind == "block":
        a = np.array(img)
        H, W = a.shape[:2]
        side = max(4, H // 14)  # one ViT patch (16 px at 224)
        for _ in range(int(p)):
            y, x = rng.integers(0, H - side + 1), rng.integers(0, W - side + 1)
            a[y:y + side, x:x + side] = rng.integers(0, 256, 3, dtype=np.uint8)
        return a
    elif kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(p))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    else:
        raise ValueError(kind)
    return np.array(img)


class CondDataset(FrameDataset):
    """FrameDataset that applies one condition: ("clean",0) | ("cutout",size) | (perturbation, level).
    The random block pattern depends only on (level, row index), so every model sees identical pixels."""

    def __init__(self, table, root, img_size=224, cond=("clean", 0), bad_log=None):
        super().__init__(table, root, img_size, train=False, bad_log=bad_log)
        self.cond = cond

    def _load(self, j):
        arr = super()._load(j)
        kind, lvl = self.cond
        if kind == "clean":
            return arr
        if kind == "cutout":
            return cutout(arr, lvl)
        return perturb_image(arr, kind, lvl, np.random.default_rng(7919 * (lvl + 1) + j))


# ----------------------------------------------------------------------------- data
def add_method(df):
    """method = FF++ sub-folder (original / Deepfakes / Face2Face / FaceSwap / NeuralTextures) or 'CelebDF'."""
    df = df.copy()
    df["method"] = np.where(df["ds"] == "ffpp", df["path"].str.split("/").str[1], "CelebDF")
    return df


def load_test_table(split_json, bad_list=None, split="test"):
    t = load_split_table(split_json, split)
    if bad_list and os.path.isfile(bad_list):
        bad = {l.split("\t")[0].strip() for l in open(bad_list) if l.strip()}
        t = t[~t.path.isin(bad)].reset_index(drop=True)
    return add_method(t)


# ----------------------------------------------------------------------------- models
def build_model(ckpt_path, device, weights_file=None, pretrained=True):
    """Rebuild UDDViT from the args stored in the checkpoint and load its trainable weights."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    m = UDDViT(a.get("arch", "vit_base_patch16_clip_224.openai"), pretrained, weights_file,
               a.get("lora_rank", 4), a.get("lora_alpha"), 2, False)
    sd = ck["model"]
    need = {n for n, p in m.named_parameters() if p.requires_grad}
    missing = need - set(sd)
    assert not missing, f"checkpoint is missing {len(missing)} trainable tensors, e.g. {sorted(missing)[:3]}"
    m.load_trainable(sd)
    return m.to(device).eval(), ck


def wrap_dp(m, device):
    return nn.DataParallel(m) if device.type == "cuda" and torch.cuda.device_count() > 1 else m


@torch.no_grad()
def predict_multi(models, table, root, img_size, device, cond=("clean", 0), batch_size=128,
                  num_workers=4, use_amp=True, want_feats=False, desc="eval", bad_log=None):
    """One pass over the data for ALL models (images are decoded/perturbed once).
    Returns probs[name] (P(fake) per row, NaN = unreadable image) and feats[name] if requested."""
    ds = CondDataset(table, root, img_size, cond, bad_log)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=device.type == "cuda")
    probs = {k: np.full(len(ds), np.nan, np.float32) for k in models}
    feats = {k: None for k in models}
    for x, _, idx in tqdm(dl, desc=desc, leave=False):
        x = normalize_on_device(x, device)
        idx = idx.numpy()
        for k, m in models.items():
            m.eval()
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits, f = m(x)
            probs[k][idx] = F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            if want_feats:
                if feats[k] is None:
                    feats[k] = np.zeros((len(ds), f.shape[1]), np.float32)
                feats[k][idx] = f.float().cpu().numpy()
    return probs, feats


# ----------------------------------------------------------------------------- metrics
_KEYS = ["auc", "ap", "acc", "bal_acc", "precision", "recall", "specificity", "f1", "eer",
         "tp", "tn", "fp", "fn"]


def metrics_from_scores(y, s, thr=0.5):
    """AUC, average precision, accuracy, balanced accuracy, precision, recall, specificity, F1, EER
    and the confusion matrix counts at threshold `thr`."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    out = {"n": int(len(y)), "n_fake": int(y.sum())}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {**out, **{k: float("nan") for k in _KEYS}}
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    pred = (s > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    out.update(auc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)),
               acc=(tp + tn) / len(y), bal_acc=(rec + spec) / 2, precision=prec, recall=rec,
               specificity=spec, f1=f1, eer=float((fpr[i] + fnr[i]) / 2), tp=tp, tn=tn, fp=fp, fn=fn)
    return out


def video_table(df):
    return df.groupby("video").agg(prob=("prob", "mean"), label=("label", "first"),
                                   ds=("ds", "first")).reset_index()


def auc_safe(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(y) and len(np.unique(y)) == 2 else float("nan")


def evaluate_df(df):
    """df columns: video, ds, method, label, prob. Video score = mean frame probability."""
    df = df.dropna(subset=["prob"])
    res = {}
    for name, sub in [("all", df), ("ffpp", df[df.ds == "ffpp"]), ("celebdf", df[df.ds == "celebdf"])]:
        if not len(sub):
            continue
        v = video_table(sub)
        res[name] = {"frame": metrics_from_scores(sub.label, sub.prob),
                     "video": metrics_from_scores(v.label, v.prob)}
    ff = df[df.ds == "ffpp"]
    per = {}
    if (ff.method == "original").any():
        for m in sorted(set(ff.method) - {"original"}):
            sub = ff[ff.method.isin(["original", m])]
            v = video_table(sub)
            per[m] = {"frame_auc": auc_safe(sub.label, sub.prob), "video_auc": auc_safe(v.label, v.prob)}
    res["per_method"] = per
    return res


# ----------------------------------------------------------------------------- attention / GradCAM++
@torch.no_grad()
def cls_attention(model, x):
    """[CLS]-as-query attention of the LAST block, per head: (B, heads, g, g) (paper Fig. 5 / 9)."""
    t = model.run_blocks(model.embed(x), 0, model.num_blocks - 1)
    blk = model.vit.blocks[-1]
    at = blk.attn
    xn = blk.norm1(t)
    B, N, C = xn.shape
    qkv = at.qkv(xn).reshape(B, N, 3, at.num_heads, C // at.num_heads).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    q, k = at.q_norm(q), at.k_norm(k)
    att = ((q * at.scale) @ k.transpose(-2, -1)).softmax(-1)
    g = int(round((N - 1) ** 0.5))
    return att[:, :, 0, 1:].reshape(B, at.num_heads, g, g)


def center_mass(att_map, size=6):
    """Share of attention inside the central size x size patches (uniform attention = size^2 / g^2)."""
    g = att_map.shape[-1]
    a = (g - size) // 2
    return (att_map[..., a:a + size, a:a + size].sum((-2, -1)) / att_map.sum((-2, -1))).cpu().numpy()


def gradcam_pp(model, x, target=1):
    """GradCAM++ on the output of the FIRST transformer block (paper Fig. 8), for class `target`
    (1 = fake). Returns (B, g, g) maps normalised to [0, 1]. Runs in fp32."""
    model.eval()
    with torch.no_grad():
        t_in = model.embed(x)
    t_in = t_in.detach().requires_grad_(True)
    t0 = model.run_blocks(t_in, 0, 1)
    t0.retain_grad()
    logits, _ = model.head(model.run_blocks(t0, 1, None))
    model.zero_grad(set_to_none=True)
    logits[:, target].sum().backward()
    A = t0[:, 1:].detach().transpose(1, 2)        # (B, D, N)
    G = t0.grad[:, 1:].transpose(1, 2)            # (B, D, N)
    g2, g3 = G ** 2, G ** 3
    denom = 2 * g2 + A.sum(2, keepdim=True) * g3
    alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
    w = (alpha * F.relu(G)).sum(2, keepdim=True)
    cam = F.relu((w * A).sum(1))                   # (B, N)
    g = int(round(cam.shape[1] ** 0.5))
    cam = cam.reshape(-1, g, g)
    cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    model.zero_grad(set_to_none=True)
    return cam.detach().cpu()
