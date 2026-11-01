"""Data pipeline for the UDD replication (baseline path).

Reads dataset_split.json ({"train"|"val"|"test": [{"image_path", "label"}, ...]}),
groups frames by video (= parent folder of the image), samples K frames per video
and serves uint8 tensors (normalisation is done on the GPU in train.py to keep
the 4 Kaggle CPU cores free for JPEG decoding).
"""
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ── Augmentation (FIX: was only horizontal flip; now matches standard deepfake
#    detection practice: random resized crop + colour jitter + h-flip) ────────
def _build_train_aug(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0),
                            ratio=(0.9, 1.1),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.05),
    ])

def _build_val_aug(img_size: int = 224) -> T.Compose:
    # Validation / test: deterministic centre crop only
    return T.Compose([
        T.CenterCrop(img_size),
    ])

# Top-level folder name (as it appears in the json paths) -> short dataset tag
# used for the per-dataset metric breakdown in train.py / train_udd.py. Anything
# not listed here falls back to its raw folder name instead of crashing.
_DATASET_DIRS = {
    "FaceForensics++": "ffpp",
    "CelebDF": "celebdf",
    "DFD": "dfd",
    "DFDC": "dfdc",
}

# Datasets whose frame folders do NOT uniquely identify a video (all frames of
# every video sit in one flat class folder, e.g. "DFDC/fake/xxx_0.png",
# "DFDC/fake/xxx_1.png", ...). For these, the video id must be recovered from
# the filename instead of the parent folder.
_FLAT_LAYOUT_DATASETS = {"DFDC"}


def _video_id(path):
    """Return a unique video id for one frame path.

    Default: the parent folder already is the video (FF++, CelebDF, DFD all
    store one video's frames in their own folder). DFDC is the exception -
    its frames are flat inside 'DFDC/fake' / 'DFDC/real', with the video id
    encoded in the filename as '<video_id>_<frame_idx>.<ext>', so we strip
    the trailing '_<digits>' from the filename stem instead.
    """
    parent, fname = path.rsplit("/", 1)
    top = path.split("/", 1)[0]
    if top in _FLAT_LAYOUT_DATASETS:
        stem = fname.rsplit(".", 1)[0]
        vid = re.sub(r"_\d+$", "", stem)
        return f"{parent}/{vid}"
    return parent


def load_split_table(split_json, split):
    """Return a DataFrame [path, label, video, ds] for one split, frames sorted per video."""
    with open(split_json, "r") as f:
        d = json.load(f)
    df = pd.DataFrame(d[split])
    df = df.rename(columns={"image_path": "path"})
    df["label"] = df["label"].astype(int)
    df["video"] = df["path"].apply(_video_id)
    top = df["path"].str.split("/", n=1).str[0]
    df["ds"] = top.map(_DATASET_DIRS).fillna(top)
    df = df.sort_values(["video", "path"]).reset_index(drop=True)
    return df


def select_frames(df, k, mode="even", seed=0, epoch=0):
    """Pick up to k frames per video.

    mode="even"   : k evenly spaced frames (paper: 'evenly sample 8 frames'), same every epoch.
    mode="random" : k random frames, re-drawn every epoch (more diversity at the same cost).
    k<=0          : keep every frame.
    Returns a new DataFrame (row order preserved within video).
    """
    if k is None or k <= 0:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed + 1000 * epoch)
    keep = []
    for _, idx in df.groupby("video", sort=False).indices.items():
        n = len(idx)
        if n <= k:
            keep.append(idx)
        elif mode == "even":
            keep.append(idx[np.unique(np.linspace(0, n - 1, k).round().astype(int))])
        elif mode == "random":
            keep.append(np.sort(rng.choice(idx, size=k, replace=False)))
        else:
            raise ValueError(mode)
    keep = np.concatenate(keep)
    return df.iloc[keep].reset_index(drop=True)


class FrameDataset(Dataset):
    """Returns (uint8 CHW tensor, label, row_index).

    Augmentation (train=True):
        RandomResizedCrop  scale=(0.8,1.0)  ratio=(0.9,1.1)
        RandomHorizontalFlip  p=0.5
        ColorJitter  brightness/contrast/saturation=0.2  hue=0.05

    Augmentation (train=False):
        CenterCrop only (image was already saved at OUT_SIZE so this is a no-op
        unless img_size differs from the saved resolution).

    If an image cannot be read (corrupt / truncated file), the next readable row
    is used instead; bad paths are appended to `bad_log` if given.
    """

    def __init__(self, table, root, img_size=224, train=False, bad_log=None):
        self.table  = table.reset_index(drop=True)
        self.paths  = self.table["path"].tolist()
        self.labels = self.table["label"].to_numpy()
        self.root   = root
        self.img_size = img_size
        self.train  = train
        self.bad_log = bad_log
        self._aug   = (_build_train_aug(img_size) if train
                       else _build_val_aug(img_size))

    def __len__(self):
        return len(self.paths)

    def _load(self, j) -> Image.Image:
        img = Image.open(os.path.join(self.root, self.paths[j])).convert("RGB")
        # Ensure the image is at least img_size on both sides before crop
        if min(img.size) < self.img_size:
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        return img

    def __getitem__(self, i):
        for off in range(10):
            j = (i + off) % len(self.paths)
            try:
                img = self._load(j)
                break
            except Exception as e:
                if self.bad_log:
                    with open(self.bad_log, "a") as f:
                        f.write(f"{self.paths[j]}\t{type(e).__name__}\n")
        else:
            raise RuntimeError(f"10 consecutive unreadable images starting at row {i}")
        img = self._aug(img)                        # PIL → PIL
        arr = np.ascontiguousarray(np.array(img)).transpose(2, 0, 1)
        return torch.from_numpy(arr), int(self.labels[j]), j


def normalize_on_device(x_uint8, device):
    """uint8 (B,3,H,W) -> float CLIP-normalised, on `device`."""
    x = x_uint8.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def sanity_check_files(table, root, n=5):
    """Fail early with a readable message if the root path is wrong."""
    for p in table["path"].iloc[:: max(1, len(table) // n)][:n]:
        fp = os.path.join(root, p)
        if not os.path.isfile(fp):
            raise FileNotFoundError(
                f"Image not found: {fp}\nCheck --data_root (paths in the JSON are relative to it)."
            )
