"""Data pipeline for the UDD replication (baseline path).

Reads dataset_split.json ({"train"|"val"|"test": [{"image_path", "label"}, ...]}),
groups frames by video (= parent folder of the image), samples K frames per video
and serves uint8 tensors (normalisation is done on the GPU in train.py to keep
the 4 Kaggle CPU cores free for JPEG decoding).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_split_table(split_json, split):
    """Return a DataFrame [path, label, video, ds] for one split, frames sorted per video."""
    with open(split_json, "r") as f:
        d = json.load(f)
    df = pd.DataFrame(d[split])
    df = df.rename(columns={"image_path": "path"})
    df["label"] = df["label"].astype(int)
    df["video"] = df["path"].str.rsplit("/", n=1).str[0]
    df["ds"] = np.where(df["path"].str.startswith("CelebDF"), "celebdf", "ffpp")
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

    If an image cannot be read (corrupt / truncated file), the next readable row is used
    instead and the *returned* label/index are those of the row actually loaded, so images
    and labels never get mismatched. Bad paths are appended to `bad_log` if given.
    """

    def __init__(self, table, root, img_size=224, train=False, bad_log=None):
        self.table = table.reset_index(drop=True)
        self.paths = self.table["path"].tolist()
        self.labels = self.table["label"].to_numpy()
        self.root = root
        self.img_size = img_size
        self.train = train
        self.bad_log = bad_log

    def __len__(self):
        return len(self.paths)

    def _load(self, j):
        img = Image.open(os.path.join(self.root, self.paths[j])).convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        return np.array(img)

    def __getitem__(self, i):
        for off in range(10):
            j = (i + off) % len(self.paths)
            try:
                arr = self._load(j)
                break
            except Exception as e:  # corrupt / unreadable image
                if self.bad_log:
                    with open(self.bad_log, "a") as f:
                        f.write(f"{self.paths[j]}\t{type(e).__name__}\n")
        else:
            raise RuntimeError(f"10 consecutive unreadable images starting at row {i}")
        if self.train and np.random.rand() < 0.5:  # horizontal flip is the only augmentation
            arr = arr[:, ::-1]
        arr = np.ascontiguousarray(arr).transpose(2, 0, 1)
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
