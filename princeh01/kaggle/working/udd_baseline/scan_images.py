"""Find unreadable images listed in dataset_split.json (run once, ~5-15 min on Kaggle).

  !python scan_images.py --data_root /kaggle/input/datasets/princeh01/udd-dataset/UDD_Dataset
Writes bad_files.txt; then train with:  --bad_list /kaggle/working/bad_files.txt
"""
import argparse
import json
import os
from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm

ROOT = None


def check(p):
    try:
        fp = os.path.join(ROOT, p)
        if os.path.getsize(fp) == 0:
            return p, "EmptyFile"
        with Image.open(fp) as im:
            im.convert("RGB")  # full decode, catches truncated data too
        return p, None
    except Exception as e:
        return p, type(e).__name__


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/kaggle/input/datasets/princeh01/udd-dataset/UDD_Dataset")
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--out", default="/kaggle/working/bad_files.txt")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ROOT = a.data_root
    d = json.load(open(a.split_json or os.path.join(ROOT, "dataset_split.json")))
    paths = sorted({x["image_path"] for v in d.values() for x in v})
    split_of = {x["image_path"]: s for s, v in d.items() for x in v}
    bad = []
    with Pool(a.workers) as pool:
        for p, err in tqdm(pool.imap_unordered(check, paths, chunksize=256), total=len(paths)):
            if err:
                bad.append((p, err))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for p, err in sorted(bad):
            f.write(f"{p}\t{err}\n")
    print(f"scanned {len(paths)} images, {len(bad)} unreadable -> {a.out}")
    for p, err in sorted(bad)[:20]:
        print(f"  [{split_of[p]}] {p}  ({err})")


if __name__ == "__main__":
    main()
