#!/usr/bin/env python3
"""
preprocess_all_datasets.py — local-video, integrated two-stage pipeline
=======================================================================

STAGE 1 — CPU: video decoding + face detection + alignment
----------------------------------------------------------
Reads each selected source video and writes aligned face crops DIRECTLY into
the final dataset structure. There is no intermediate raw-frame directory.

Targets per video:
  - FaceForensics++: 8 successful face crops, with the whole video assigned
    to either train OR val before processing (video-level split).
  - DFD, DFDC, CelebDF: 32 successful face crops in test.

Important: the script does not merely sample 8/32 frames and discard frames
without faces. It tries evenly distributed frames first, then checks additional
frames from that same video until the target number of valid aligned crops is
reached or all frames have been examined. If fewer than the target number of
frames contain detectable faces, it reports the actual count.

Output:
  dataset_processed/
  ├── train/FaceForensics++/<class>/<video>/<frame>.jpg
  ├── val/FaceForensics++/<class>/<video>/<frame>.jpg
  └── test/<DFD|DFDC|CelebDF>/<class>/<video>/<frame>.jpg

Dlib face detection, landmark prediction and alignment run on the CPU.
Original videos are never moved, modified or deleted.

STAGE 2 — CPU: manifest generation
-----------------------------------
Scans the aligned face-crop folders and generates:
  - dataset_split.json: {"train": [...], "val": [...], "test": [...]}
  - dataset.csv: flat image_path,label table

Labels: 0 = real, 1 = fake.
The manifest stage can also be run independently on an already processed
dataset with the expected folder structure.

Usage:
  # Process only DFDC videos and create its test manifest.
  python preprocess_all_datasets.py all \\
      --only DFDC \\
      --dfdc_root ./dataset/DFDC \\
      --dataset_root ./dataset_processed \\
      --workers 4

  # Process all datasets.
  python preprocess_all_datasets.py all \\
      --ffpp_root ./dataset/FaceForensics++ \\
      --dfd_root ./dataset/DFD \\
      --dfdc_root ./dataset/DFDC \\
      --celebdf_root ./dataset/Celeb-DF-v2 \\
      --dataset_root ./dataset_processed \\
      --val_frac 0.15 --seed 42 --workers 4

  # Stage 1 only: videos -> aligned face crops.
  python preprocess_all_datasets.py stage1 --only DFDC \\
      --dfdc_root ./dataset/DFDC \\
      --dataset_root ./dataset_processed

  # Stage 2 only: generate JSON + CSV from existing aligned face crops.
  python preprocess_all_datasets.py stage2 --only DFDC \\
      --dataset_root ./dataset_processed
"""
import argparse
import csv
import json
import random
import re
import shutil
import sys
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Frames sampled per video
FF_FRAMES        = 8    # FF++ — each video assigned to train OR val
TEST_FRAMES      = 32   # DFD, DFDC, CelebDF — test

# Output image quality
JPEG_QUALITY     = 95

# FF++ dataset sub-paths
FF_DATASETS = {
    "original":       "original_sequences/youtube",
    "Deepfakes":      "manipulated_sequences/Deepfakes",
    "Face2Face":      "manipulated_sequences/Face2Face",
    "FaceSwap":       "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}
FF_REAL = {"original"}

MAX_VIDEO_DEPTH = 4

# Dlib face alignment settings (matches DeepfakeBench exactly)
DLIB_PREDICTOR   = "./dlib_tools/shape_predictor_81_face_landmarks.dat"
ALIGN_OUT_SIZE   = 256    # save at 256×256 (paper standard); dataloader resizes to 224
ALIGN_SCALE      = 1.3    # crop scale around the aligned face

_done_log_lock   = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# Dlib face alignment  (DeepfakeBench preprocess.py — exact replica)
# ──────────────────────────────────────────────────────────────────────────────

def _build_dlib_tools():
    """Load dlib detector + predictor. Call once per worker process."""
    try:
        import dlib
        from imutils import face_utils
        from skimage import transform as trans
    except ImportError:
        sys.exit(
            "\nERROR: Stage 2 requires dlib, imutils, scikit-image.\n"
            "  pip install dlib imutils scikit-image\n"
        )
    if not Path(DLIB_PREDICTOR).exists():
        sys.exit(
            f"\nERROR: Dlib shape predictor not found at '{DLIB_PREDICTOR}'.\n"
            "Download shape_predictor_81_face_landmarks.dat from:\n"
            "  https://github.com/SCLBD/DeepfakeBench/releases\n"
            f"and place it at: {Path(DLIB_PREDICTOR).resolve()}\n"
        )
    import dlib
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_PREDICTOR)
    return detector, predictor


# 5 canonical landmark positions used by DeepfakeBench (ArcFace standard)
_ARCFACE_DST = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041],
], dtype=np.float32)


def _get_5pts(image_rgb, face, predictor):
    """Extract 5 key points (leye, reye, nose, lmouth, rmouth) via Dlib."""
    import dlib
    shape   = predictor(image_rgb, face)
    leye    = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye    = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose    = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth  = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth  = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)


def _align_crop(image_rgb, landmarks, out_size=ALIGN_OUT_SIZE,
                scale=ALIGN_SCALE):
    """
    Similarity-transform align + crop — identical to DeepfakeBench's
    img_align_crop() with scale=1.3.
    """
    from skimage import transform as trans

    dst = _ARCFACE_DST.copy()
    # Adjust canonical points for output size (they are defined for 112×112)
    if out_size != 112:
        dst[:, 0] += 8.0               # standard ArcFace offset for 112+16
        dst[:, 0] = dst[:, 0] * out_size / 112
        dst[:, 1] = dst[:, 1] * out_size / 112

    margin_rate = scale - 1
    x_margin    = out_size * margin_rate / 2.0
    y_margin    = out_size * margin_rate / 2.0
    dst[:, 0]  += x_margin
    dst[:, 1]  += y_margin
    dst[:, 0]  *= out_size / (out_size + 2 * x_margin)
    dst[:, 1]  *= out_size / (out_size + 2 * y_margin)

    src    = landmarks.astype(np.float32)
    tform  = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M      = tform.params[0:2, :]
    warped = cv2.warpAffine(image_rgb, M, (out_size, out_size))
    return warped


def align_face(image_bgr, detector, predictor):
    """
    Detect the largest face in image_bgr, align + crop it.
    Returns aligned BGR image (ALIGN_OUT_SIZE × ALIGN_OUT_SIZE) or None.
    """
    rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(rgb, 1)
    if not faces:
        return None
    face      = max(faces, key=lambda r: r.width() * r.height())
    landmarks = _get_5pts(rgb, face, predictor)
    aligned   = _align_crop(rgb, landmarks)
    return cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _reporthook(count, block, total):
    pct = min(100, int(count * block * 100 / max(total, 1)))
    sys.stdout.write(f"\r  {pct:3d}%  {count*block/1e6:.1f} MB")
    sys.stdout.flush()


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp), reporthook=_reporthook)
        sys.stdout.write("\n")
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _done_set(log: Path) -> set:
    return set(log.read_text().splitlines()) if log.exists() else set()


def _mark_done(log: Path, key: str):
    with _done_log_lock:
        with open(log, "a") as f:
            f.write(key + "\n")


def _find_subdir(root: Path, name: str) -> Path | None:
    """Case-insensitive sub-directory lookup."""
    direct = root / name
    if direct.exists():
        return direct
    for p in root.iterdir():
        if p.name.lower() == name.lower():
            return p
    return None


def _sample_indices(total: int, n: int, seed: int | None = None) -> list[int]:
    """Pick n evenly-spaced frame indices then randomly shuffle selection."""
    even = np.unique(np.linspace(0, total - 1, min(n, total),
                                 endpoint=True, dtype=int)).tolist()
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(even)
        even = sorted(even[:n])
    return even

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — CPU: video decoding + face detection/alignment in one pass
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}


def _videos_at_depth(root: Path, max_depth: int = MAX_VIDEO_DEPTH) -> list[Path]:
    """Find videos no more than max_depth directories below root."""
    import os
    root = root.resolve()
    found = []
    if not root.is_dir():
        print(f"  [warn] Dataset directory does not exist: {root}")
        return found
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        if depth >= max_depth:
            dirs[:] = []
        for filename in files:
            path = current_path / filename
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                found.append(path)
    return sorted(found)


def _ffpp_category(video_path: Path, ffpp_root: Path) -> str | None:
    """Infer the FF++ manipulation category from its directory path."""
    parts = [part.lower() for part in video_path.relative_to(ffpp_root).parts[:-1]]
    joined = "/".join(parts)
    if "original" in joined or "youtube" in joined:
        return "original"
    for category in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
        if category.lower() in parts or category.lower() in joined:
            return category
    for category in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
        if category.lower() in video_path.parent.name.lower():
            return category
    return None


# Dlib instances are created once in each worker process.
_proc_detector = None
_proc_predictor = None


def _worker_init(predictor_path: str):
    global _proc_detector, _proc_predictor
    import dlib
    _proc_detector = dlib.get_frontal_face_detector()
    _proc_predictor = dlib.shape_predictor(predictor_path)


def _stable_video_seed(video_path: str, seed: int) -> int:
    """Stable per-video seed (unlike Python's process-randomized hash())."""
    import zlib
    return (zlib.crc32(video_path.encode("utf-8")) + int(seed)) & 0xFFFFFFFF


def _candidate_frame_order(total_frames: int, target: int, seed: int) -> list[int]:
    """
    Try evenly distributed frames first, then every remaining frame.
    This lets the worker replace no-face frames with other frames from the
    SAME video until it obtains the requested number of valid face crops.
    """
    if total_frames <= 0:
        return []
    first_count = min(total_frames, max(target * 4, target))
    first = np.linspace(0, total_frames - 1, first_count, dtype=int).tolist()
    seen = set(first)
    remaining = [idx for idx in range(total_frames) if idx not in seen]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    return first + remaining


def _direct_video_worker(args):
    """
    Decode a video and save up to target valid aligned face crops directly
    into the final train/val/test folder. No intermediate raw-frame folder.
    """
    video_path_str, output_dir_str, target, seed = args
    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p for p in output_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(existing) >= target:
        return str(video_path), len(existing), 0, "already_complete"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return str(video_path), len(existing), 0, "cannot_open"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return str(video_path), len(existing), 0, "empty_video"

    # Existing filenames are frame-index based, so resume without overwriting.
    existing_names = {p.name for p in existing}
    saved = len(existing)
    attempted = 0
    per_video_seed = _stable_video_seed(str(video_path), seed)
    candidates = _candidate_frame_order(total_frames, target - saved, per_video_seed)

    try:
        for frame_idx in candidates:
            if saved >= target:
                break
            out_name = f"{frame_idx:06d}.jpg"
            if out_name in existing_names:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            attempted += 1
            aligned = align_face(frame, _proc_detector, _proc_predictor)
            if aligned is None:
                continue

            out_path = output_dir / out_name
            if cv2.imwrite(
                str(out_path), aligned,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            ):
                saved += 1
                existing_names.add(out_name)
    finally:
        cap.release()

    status = "ok" if saved >= target else "insufficient_faces"
    return str(video_path), saved, attempted, status


def _run_video_face_processing(jobs, workers: int, predictor_path: str,
                               description: str):
    """Run integrated video decoding + Dlib alignment with one Dlib instance per process."""
    from concurrent.futures import ProcessPoolExecutor

    if not jobs:
        print(f"  [warn] No videos found for {description}")
        return

    completed = failed = short = total_crops = 0
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=_worker_init,
        initargs=(predictor_path,),
    ) as pool:
        futures = [pool.submit(_direct_video_worker, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=description, leave=True):
            video, saved, attempted, status = future.result()
            total_crops += saved
            if status == "ok":
                completed += 1
            elif status == "already_complete":
                completed += 1
            elif status == "insufficient_faces":
                short += 1
                print(f"  [warn] {video}: only {saved} face crops found "
                      f"after checking {attempted} frames")
            else:
                failed += 1
                print(f"  [warn] {video}: {status}")
    print(f"  Results: videos complete={completed}, below target={short}, "
          f"unreadable/errors={failed}, face crops present={total_crops:,}")


def _ffpp_video_jobs(ffpp_root: Path, dataset_root: Path,
                     val_frac: float, seed: int):
    """Discover FF++ videos and assign whole videos to train OR val."""
    videos = _videos_at_depth(ffpp_root)
    grouped = defaultdict(list)
    for video in videos:
        category = _ffpp_category(video, ffpp_root)
        if category is None:
            print(f"  [skip] Cannot identify FF++ category: {video}")
            continue
        grouped[category].append(video)

    jobs = []
    for category, category_videos in sorted(grouped.items()):
        category_videos = sorted(category_videos)
        rng = random.Random(seed)
        rng.shuffle(category_videos)
        n_val = max(1, int(len(category_videos) * val_frac)) if len(category_videos) > 1 else 0
        val_videos = set(category_videos[:n_val])

        for video in sorted(category_videos):
            split_name = "val" if video in val_videos else "train"
            rel_parent = video.relative_to(ffpp_root).parent
            ignored = {
                "original_sequences", "manipulated_sequences", "youtube",
                "videos", "c23", "c40", "raw", "deepfakes", "face2face",
                "faceswap", "neuraltextures",
            }
            safe_parent = Path(*[
                part for part in rel_parent.parts if part.lower() not in ignored
            ])
            output_dir = (
                dataset_root / split_name / "FaceForensics++" /
                category / safe_parent / video.stem
            )
            jobs.append((str(video), str(output_dir), FF_FRAMES, seed))

        print(f"  FF++/{category}: {len(category_videos) - n_val} train videos, "
              f"{n_val} val videos")
    return jobs


def _test_video_jobs(dataset_name: str, source_root: Path,
                     dataset_root: Path, seed: int):
    """Discover test videos and retain their class/nested directory structure."""
    videos = _videos_at_depth(source_root)
    print(f"  {dataset_name}: found {len(videos)} videos "
          f"(maximum directory depth: {MAX_VIDEO_DEPTH})")
    jobs = []
    for video in videos:
        rel_parent = video.relative_to(source_root).parent
        output_dir = dataset_root / "test" / dataset_name / rel_parent / video.stem
        jobs.append((str(video), str(output_dir), TEST_FRAMES, seed))
    return jobs


def stage1_process_videos(ffpp_root: Path | None, dfd_root: Path | None,
                          dfdc_root: Path | None, celebdf_root: Path | None,
                          dataset_root: Path, workers: int = 2, seed: int = 42,
                          val_frac: float = 0.15,
                          predictor_path: str = DLIB_PREDICTOR,
                          only: list[str] | None = None):
    """
    Integrated Stage 1: decode videos, detect faces, align and save final crops.

    FF++ videos are assigned to train/val BEFORE processing, at video level.
    Each FF++ video produces up to 8 valid face crops; each test video
    produces up to 32 valid face crops. If a sampled frame has no face,
    additional frames from that same video are tried until the target is met
    or every frame has been checked.
    """
    valid = ["FaceForensics++", "DFD", "DFDC", "CelebDF"]
    selected = set(only) if only else set(valid)
    unknown = selected - set(valid)
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(sorted(unknown))}")

    print(f"\n{'=' * 72}")
    print("STAGE 1 — Integrated video extraction + Dlib face alignment (CPU)")
    print(f"Target: FF++={FF_FRAMES} aligned crops/video; test={TEST_FRAMES} aligned crops/video")
    print("No intermediate raw frames are written.")
    print(f"Selected: {', '.join(name for name in valid if name in selected)}")
    print(f"Workers: {workers} | seed: {seed} | validation fraction: {val_frac:.1%}")
    print("=" * 72)

    # Validate Dlib dependencies/model before starting any jobs.
    _build_dlib_tools()
    dataset_root.mkdir(parents=True, exist_ok=True)

    if "FaceForensics++" in selected:
        if ffpp_root is None:
            print("  [skip] FF++ selected, but --ffpp_root was not supplied")
        elif not ffpp_root.is_dir():
            print(f"  [skip] FF++ source folder not found: {ffpp_root}")
        else:
            jobs = _ffpp_video_jobs(ffpp_root, dataset_root, val_frac, seed)
            _run_video_face_processing(
                jobs, workers, predictor_path, "FF++ video → aligned face crops"
            )

    test_roots = {
        "DFD": dfd_root,
        "DFDC": dfdc_root,
        "CelebDF": celebdf_root,
    }
    for dataset_name in ("DFD", "DFDC", "CelebDF"):
        if dataset_name not in selected:
            print(f"  [skip] {dataset_name}: not selected")
            continue
        source_root = test_roots[dataset_name]
        if source_root is None:
            print(f"  [skip] {dataset_name}: source root argument not supplied")
            continue
        if not source_root.is_dir():
            print(f"  [skip] {dataset_name}: source folder not found: {source_root}")
            continue
        jobs = _test_video_jobs(dataset_name, source_root, dataset_root, seed)
        _run_video_face_processing(
            jobs, workers, predictor_path,
            f"{dataset_name} video → aligned face crops"
        )

    print(f"\nIntegrated Stage 1 complete. Aligned crops saved under {dataset_root}/")


# ══════════════════════════════════════════════════════════════════════════════

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Immediate class-folder name -> binary label (0=real, 1=fake).
STAGE3_LABEL_MAPS = {
    "FaceForensics++": {
        "original": 0,
        "Deepfakes": 1,
        "Face2Face": 1,
        "FaceSwap": 1,
        "NeuralTextures": 1,
        "FaceShifter": 1,
        "DeepFakeDetection": 1,
        "DeepFakeDetection_original": 0,
    },
    "DFD": {
        "DFD_original_sequences": 0,
        "DFD_manipulated_sequences": 1,
    },
    "DFDC": {"real": 0, "fake": 1},
    "CelebDF": {
        "Celeb-real": 0,
        "YouTube-real": 0,
        "Celeb-synthesis": 1,
    },
}


def _iter_manifest_images(folder: Path):
    if not folder.is_dir():
        return
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            yield path


def _collect_manifest_split(split_dir: Path, dataset_root: Path,
                            dataset_name: str, label_map: dict,
                            max_frames_per_video: int | None = None) -> list[dict]:
    """Collect {image_path, label} records; optionally cap frames per video folder."""
    ds_root = split_dir / dataset_name
    records = []
    if not ds_root.is_dir():
        print(f"  [warn] Not found: {ds_root} — skipping")
        return records

    for class_dir in sorted(ds_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name not in label_map:
            print(f"  [{dataset_name}] Unknown class folder '{class_dir.name}' — skipping")
            continue

        label = label_map[class_dir.name]
        before = len(records)
        if max_frames_per_video is None:
            for image_path in _iter_manifest_images(class_dir):
                records.append({
                    "image_path": image_path.relative_to(dataset_root).as_posix(),
                    "label": label,
                })
        else:
            video_buckets = defaultdict(list)
            for image_path in _iter_manifest_images(class_dir):
                # Stage 2 stores every video's frames in its own leaf directory.
                video_buckets[image_path.parent].append(image_path)
            for video_dir in sorted(video_buckets):
                for image_path in video_buckets[video_dir][:max_frames_per_video]:
                    records.append({
                        "image_path": image_path.relative_to(dataset_root).as_posix(),
                        "label": label,
                    })

        count = len(records) - before
        cap_note = f" (max {max_frames_per_video}/video)" if max_frames_per_video else ""
        print(f"    {class_dir.name:40s} label={label} images={count:,}{cap_note}")

    return records


def _manifest_video_key(record: dict) -> str:
    # Drop train/val so the same relative video path is comparable across splits.
    parts = Path(record["image_path"]).parts
    return str(Path(*parts[1:-1]))


def _check_manifest_leakage(train: list[dict], val: list[dict]):
    train_videos = {_manifest_video_key(r) for r in train}
    val_videos = {_manifest_video_key(r) for r in val}
    overlap = train_videos & val_videos
    if overlap:
        print(f"  [WARNING] {len(overlap)} video folder(s) occur in both train and val:")
        for item in sorted(overlap)[:10]:
            print(f"    {item}")
    else:
        print(f"  No train/val video leakage detected "
              f"(train={len(train_videos):,} videos, val={len(val_videos):,} videos).")


def _print_manifest_stats(name: str, records: list[dict]):
    real = sum(record["label"] == 0 for record in records)
    fake = sum(record["label"] == 1 for record in records)
    print(f"  {name:12s} total={len(records):>10,}  real={real:>9,}  fake={fake:>9,}")


def stage2(dataset_root: Path, output_dir: Path | None = None,
           train_frames: int = 8, val_frames: int = 8, test_frames: int = 32,
           exclude_dfd: bool = False, exclude_dfdc: bool = False,
           exclude_celebdf: bool = False,
           only: list[str] | None = None):
    """
    Build dataset_split.json and dataset.csv from a train/val/test directory tree.

    By default, process all datasets. If `only` is supplied, scan only the
    requested dataset(s). FaceForensics++ contributes train and val records;
    DFD, DFDC and CelebDF contribute test records. A test-only selection does
    not require train/val directories to exist.

    Train and validation default to 8 frames/video; cross-dataset test sets
    default to 32 frames/video. Set a cap to 0 to keep all available frames.
    """
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = (output_dir or dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"ERROR: dataset root not found: {dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cap = train_frames or None
    val_cap = val_frames or None
    test_cap = test_frames or None

    valid_datasets = ["FaceForensics++", "DFD", "DFDC", "CelebDF"]
    selected = set(only) if only else set(valid_datasets)
    unknown = selected - set(valid_datasets)
    if unknown:
        raise SystemExit(
            f"ERROR: Unknown dataset(s): {', '.join(sorted(unknown))}. "
            f"Choose from: {', '.join(valid_datasets)}"
        )

    excluded = {
        "DFD": exclude_dfd,
        "DFDC": exclude_dfdc,
        "CelebDF": exclude_celebdf,
    }
    conflicts = selected.intersection(name for name, is_excluded in excluded.items()
                                       if is_excluded)
    if conflicts:
        raise SystemExit(
            "ERROR: Dataset(s) cannot be both selected with --only and excluded: "
            + ", ".join(sorted(conflicts))
        )

    print(f"\n{'=' * 60}\nSTAGE 2 — Build dataset manifests\n{'=' * 60}")
    print(f"  Dataset root: {dataset_root}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Selection   : {', '.join(name for name in valid_datasets if name in selected)}")

    train, val, test = [], [], []

    if "FaceForensics++" in selected:
        print("\nScanning train/FaceForensics++ …")
        train = _collect_manifest_split(
            dataset_root / "train", dataset_root, "FaceForensics++",
            STAGE3_LABEL_MAPS["FaceForensics++"], train_cap)
        if not train:
            raise SystemExit(
                "ERROR: No training images found under train/FaceForensics++/. "
                "Check the dataset directory structure."
            )

        print("\nScanning val/FaceForensics++ …")
        val = _collect_manifest_split(
            dataset_root / "val", dataset_root, "FaceForensics++",
            STAGE3_LABEL_MAPS["FaceForensics++"], val_cap)

    for dataset_name in ("DFD", "DFDC", "CelebDF"):
        if dataset_name not in selected:
            continue
        print(f"\nScanning test/{dataset_name} …")
        test.extend(_collect_manifest_split(
            dataset_root / "test", dataset_root, dataset_name,
            STAGE3_LABEL_MAPS[dataset_name], test_cap))

    if not train and not val and not test:
        raise SystemExit(
            "ERROR: No images found for the selected dataset(s). "
            "Check the folder structure and class-folder names."
        )

    print("\nDataset statistics:")
    _print_manifest_stats("train", train)
    _print_manifest_stats("val", val)
    _print_manifest_stats("test", test)

    if "FaceForensics++" in selected:
        print("\nChecking train/val video-level separation …")
        _check_manifest_leakage(train, val)

    json_path = output_dir / "dataset_split.json"
    csv_path = output_dir / "dataset.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"train": train, "val": val, "test": test}, file, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(train + val + test)

    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote CSV : {csv_path}")
    print("Stage 2 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage deepfake preprocessing (Stage 1: integrated CPU video-to-face-crops, "
                    "Stage 2: CSV/JSON manifests)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("stage", choices=["stage1", "stage2", "all"],
                   help="stage1=video-to-aligned-crops; stage2=manifest generation; all=both.")

    # Shared
    p.add_argument("--frames_root",  default="./frames",
                   help="Legacy unused option; integrated Stage 1 writes directly to --dataset_root.")
    p.add_argument("--dataset_root", default="./dataset_processed",
                   help="Root for aligned face crops split into train/val/test.")
    p.add_argument("--workers",      type=int, default=2)
    p.add_argument("--seed",         type=int, default=42)

    # Stage 1 — local dataset folders; nothing is downloaded or deleted.
    p.add_argument("--ffpp_root", required=False,
                   help="Local FF++ folder containing the video files.")
    p.add_argument("--celebdf_root", default=None,
                   help="Local CelebDF folder containing video files.")
    p.add_argument("--dfdc_root", default=None,
                   help="Local DFDC folder containing video files.")
    p.add_argument("--dfd_root", default=None,
                   help="Local DFD folder containing video files.")

    # Stage 2 — split
    p.add_argument("--val_frac", type=float, default=0.15,
                   help="Fraction of FF++ videos for validation (video-level).")
    p.add_argument("--dlib_predictor", default=DLIB_PREDICTOR,
                   help="Path to shape_predictor_81_face_landmarks.dat.")

    # Stage 2 — manifest generation
    p.add_argument("--manifest_out", default=None,
                   help="Output directory for dataset_split.json and dataset.csv (default: dataset_root).")
    p.add_argument("--train_frames", type=int, default=8,
                   help="Maximum frames per video in training (0 = no cap).")
    p.add_argument("--val_frames", type=int, default=8,
                   help="Maximum frames per video in validation (0 = no cap).")
    p.add_argument("--test_frames", type=int, default=32,
                   help="Maximum frames per video in test (0 = no cap).")
    p.add_argument("--exclude_dfd", action="store_true",
                   help="Exclude DFD from the test manifest.")
    p.add_argument("--exclude_dfdc", action="store_true",
                   help="Exclude DFDC from the test manifest.")
    p.add_argument("--exclude_celebdf", action="store_true",
                   help="Exclude CelebDF from the test manifest.")
    p.add_argument("--only", nargs="+",
                   choices=["FaceForensics++", "DFD", "DFDC", "CelebDF"],
                   default=None, metavar="DATASET",
                   help="Restrict EVERY selected stage (including 'all') to these dataset(s). "
                        "FaceForensics++ maps to train/val; DFD, DFDC and CelebDF map to test. "
                        "Omit to process all datasets.")
    return p.parse_args()


def main():
    args = parse_args()

    global DLIB_PREDICTOR
    DLIB_PREDICTOR = args.dlib_predictor

    dataset_root = Path(args.dataset_root).expanduser()
    selected = args.only

    if args.stage in ("stage1", "all"):
        stage1_process_videos(
            ffpp_root=Path(args.ffpp_root).expanduser() if args.ffpp_root else None,
            dfd_root=Path(args.dfd_root).expanduser() if args.dfd_root else None,
            dfdc_root=Path(args.dfdc_root).expanduser() if args.dfdc_root else None,
            celebdf_root=Path(args.celebdf_root).expanduser() if args.celebdf_root else None,
            dataset_root=dataset_root,
            workers=args.workers,
            seed=args.seed,
            val_frac=args.val_frac,
            predictor_path=args.dlib_predictor,
            only=selected,
        )

    if args.stage in ("stage2", "all"):
        stage2(
            dataset_root=dataset_root,
            output_dir=Path(args.manifest_out).expanduser() if args.manifest_out else None,
            train_frames=args.train_frames,
            val_frames=args.val_frames,
            test_frames=args.test_frames,
            exclude_dfd=args.exclude_dfd,
            exclude_dfdc=args.exclude_dfdc,
            exclude_celebdf=args.exclude_celebdf,
            only=selected,
        )



if __name__ == "__main__":
    main()
