"""
Dataset preparation script for UDD deepfake detection.

Crawls FaceForensics++ and CelebDF face folders and produces:
  - A flat CSV  (image_path, label)          -> dataset.csv
  - A JSON split (train/val/test lists)      -> dataset_split.json

Label convention:  0 = real,  1 = fake

FaceForensics++ layout
-----------------------
  FaceForensics++/original/<id>/          -> REAL
  FaceForensics++/Deepfakes/<id>/         -> FAKE
  FaceForensics++/Face2Face/<id>/         -> FAKE
  FaceForensics++/FaceSwap/<id>/          -> FAKE
  FaceForensics++/NeuralTextures/<id>/    -> FAKE

CelebDF layout
--------------
  CelebDF/CelebDF_faces/Celeb-real/<id>/      -> REAL
  CelebDF/CelebDF_faces/YouTube-real/<id>/    -> REAL
  CelebDF/CelebDF_faces/Celeb-synthesis/<id>/ -> FAKE

Run from the project root:
  python prepare_dataset.py [--root /path/to/project]

All paths in the CSV/JSON are RELATIVE to --root (e.g.
"FaceForensics++/Deepfakes/001_003/0001.jpg" or
"CelebDF/CelebDF_faces/Celeb-real/id0_0000/0001.jpg"), so the dataset
can be moved or re-mounted (e.g. a different Kaggle input path) without
regenerating the split.
"""

import os
import csv
import json
import random
import argparse
from pathlib import Path

# ── label map ────────────────────────────────────────────────────────────────
FF_REAL = {"original"}
FF_FAKE = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}

CELEB_REAL = {"Celeb-real", "YouTube-real"}
CELEB_FAKE = {"Celeb-synthesis"}

VALID_EXT = {".jpg", ".jpeg", ".png"}

# ── helpers ──────────────────────────────────────────────────────────────────

def iter_images(folder: Path):
    """Yield every image file under *folder* (non-recursive one level)."""
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in VALID_EXT:
            yield p


def rel(img_path: Path, root: Path) -> str:
    """Path of *img_path* relative to *root*, using forward slashes."""
    return img_path.relative_to(root).as_posix()


def collect_ff(ff_root: Path, root: Path):
    """Return list of (relative_path_str, label) for FaceForensics++."""
    records = []

    for split_name in sorted(ff_root.iterdir()):
        if not split_name.is_dir():
            continue

        name = split_name.name
        if name in FF_REAL:
            label = 0
        elif name in FF_FAKE:
            label = 1
        else:
            print(f"  [FF++] Skipping unknown directory: {name}")
            continue

        # Each split has many video-level sub-folders
        for video_dir in sorted(split_name.iterdir()):
            if not video_dir.is_dir():
                continue
            for img_path in iter_images(video_dir):
                records.append((rel(img_path, root), label))

    return records


def collect_celeb(celeb_root: Path, root: Path):
    """Return list of (relative_path_str, label) for CelebDF."""
    records = []
    faces_root = celeb_root / "CelebDF_faces"

    if not faces_root.exists():
        print(f"  [CelebDF] CelebDF_faces not found at {faces_root}")
        return records

    for split_name in sorted(faces_root.iterdir()):
        if not split_name.is_dir():
            continue

        name = split_name.name
        if name in CELEB_REAL:
            label = 0
        elif name in CELEB_FAKE:
            label = 1
        else:
            print(f"  [CelebDF] Skipping unknown directory: {name}")
            continue

        for video_dir in sorted(split_name.iterdir()):
            if not video_dir.is_dir():
                continue
            for img_path in iter_images(video_dir):
                records.append((rel(img_path, root), label))

    return records


def split_records(records, train=0.7, val=0.15, test=0.15, seed=42):
    """Stratified random split preserving real/fake ratio."""
    assert abs(train + val + test - 1.0) < 1e-9, "Splits must sum to 1.0"

    rng = random.Random(seed)
    real = [r for r in records if r[1] == 0]
    fake = [r for r in records if r[1] == 1]

    def _split(items):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train)
        n_val   = int(n * val)
        return (
            items[:n_train],
            items[n_train:n_train + n_val],
            items[n_train + n_val:],
        )

    r_tr, r_va, r_te = _split(real)
    f_tr, f_va, f_te = _split(fake)

    return (
        r_tr + f_tr,
        r_va + f_va,
        r_te + f_te,
    )


def write_csv(records, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(records)


def write_split_json(train, val, test, path: Path):
    data = {
        "train": [{"image_path": p, "label": l} for p, l in train],
        "val":   [{"image_path": p, "label": l} for p, l in val],
        "test":  [{"image_path": p, "label": l} for p, l in test],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_stats(name, records):
    real = sum(1 for _, l in records if l == 0)
    fake = sum(1 for _, l in records if l == 1)
    print(f"  {name:10s}  total={len(records):7,}  real={real:6,}  fake={fake:6,}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build a labelled image list from FF++ and CelebDF face folders."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root containing FaceForensics++ and CelebDF (default: cwd). "
             "All output paths are stored relative to this folder.",
    )
    parser.add_argument(
        "--out",
        default=".",
        help="Output directory for CSV and JSON (default: cwd)",
    )
    parser.add_argument(
        "--train", type=float, default=0.70, help="Train fraction (default: 0.70)"
    )
    parser.add_argument(
        "--val",   type=float, default=0.15, help="Val fraction   (default: 0.15)"
    )
    parser.add_argument(
        "--test",  type=float, default=0.15, help="Test fraction  (default: 0.15)"
    )
    parser.add_argument(
        "--seed",  type=int,   default=42,   help="Random seed    (default: 42)"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out  = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nProject root : {root}")
    print(f"Output dir   : {out}\n")

    # ── collect ──────────────────────────────────────────────────────────────
    ff_root    = root / "FaceForensics++"
    celeb_root = root / "CelebDF"

    all_records = []

    if ff_root.exists():
        print("Scanning FaceForensics++ …")
        ff_records = collect_ff(ff_root, root)
        print_stats("FF++", ff_records)
        all_records.extend(ff_records)
    else:
        print(f"[WARN] FaceForensics++ not found at {ff_root}")

    if celeb_root.exists():
        print("Scanning CelebDF …")
        celeb_records = collect_celeb(celeb_root, root)
        print_stats("CelebDF", celeb_records)
        all_records.extend(celeb_records)
    else:
        print(f"[WARN] CelebDF not found at {celeb_root}")

    if not all_records:
        print("\n[ERROR] No images found. Check --root path.")
        return

    print()
    print_stats("TOTAL", all_records)

    # ── split ────────────────────────────────────────────────────────────────
    print(f"\nSplitting  train={args.train:.0%}  val={args.val:.0%}  test={args.test:.0%}  seed={args.seed}")
    train_recs, val_recs, test_recs = split_records(
        all_records, args.train, args.val, args.test, args.seed
    )
    print_stats("train", train_recs)
    print_stats("val",   val_recs)
    print_stats("test",  test_recs)

    # ── write outputs ─────────────────────────────────────────────────────────
    csv_path  = out / "dataset.csv"
    json_path = out / "dataset_split.json"

    write_csv(all_records, csv_path)
    write_split_json(train_recs, val_recs, test_recs, json_path)

    print(f"\n✓ dataset.csv        → {csv_path}  ({len(all_records):,} rows)")
    print(f"✓ dataset_split.json → {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
