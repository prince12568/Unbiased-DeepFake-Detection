#!/usr/bin/env python3
"""
FaceForensics++ Chunked Download + Face Crop Pipeline
======================================================
Downloads FF++ videos in chunks, extracts cropped face frames,
saves them to disk, then deletes the raw videos before the next chunk.

Usage:
    python preprocess_ff_chunked.py \
        --base_url  <URL from FF++ form email> \
        --output_root /home/shlok/Project/ML_Project \
        --datasets Deepfakes Face2Face FaceSwap NeuralTextures original \
        --compression c23 \
        --chunk_size 10 \
        --frames_per_video 32 \
        --face_size 224 \
        --server EU

Dependencies (see requirements_preprocess.txt):
    pip install opencv-python-headless tqdm retina-face deepface
    # OR: pip install insightface onnxruntime   (faster RetinaFace)

Face detector priority (auto-fallback):
    1. insightface  (RetinaFace via ONNX – fastest)
    2. retinaface   (deepface's bundled RetinaFace)
    3. cv2 Haar cascade (last resort, least accurate)
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Dataset constants (mirrors the original download script)
# ──────────────────────────────────────────────────────────────────────────────
DATASETS = {
    "original": "original_sequences/youtube",
    "DeepFakeDetection_original": "original_sequences/actors",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "DeepFakeDetection": "manipulated_sequences/DeepFakeDetection",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceShifter": "manipulated_sequences/FaceShifter",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}
FILELIST_URL = "misc/filelist.json"
DEEPFAKES_DETECTION_URL = "misc/deepfake_detection_filenames.json"

# ──────────────────────────────────────────────────────────────────────────────
# Face detector loader (tries best available, falls back gracefully)
# ──────────────────────────────────────────────────────────────────────────────

def _load_detector():
    """Returns (detector_name, detector_object)."""
    # 1. insightface (RetinaFace + ONNX – best speed/accuracy)
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[detector] Using insightface RetinaFace (ONNX)")
        return "insightface", app
    except ImportError:
        pass

    # 2. retinaface (pip install retina-face)
    try:
        from retinaface import RetinaFace  # noqa
        print("[detector] Using retinaface")
        return "retinaface", None          # stateless API
    except ImportError:
        pass

    # 3. OpenCV Haar cascade (always available)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("[detector] Falling back to OpenCV Haar cascade (least accurate)")
    return "haar", face_cascade


DETECTOR_NAME, DETECTOR_OBJ = _load_detector()


def detect_face(frame_bgr: np.ndarray):
    """
    Returns (x1, y1, x2, y2) of the largest face, or None.
    frame_bgr: H×W×3 BGR uint8 numpy array.
    """
    h, w = frame_bgr.shape[:2]

    if DETECTOR_NAME == "insightface":
        rgb = frame_bgr[:, :, ::-1]
        faces = DETECTOR_OBJ.get(rgb)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        return x1, y1, x2, y2

    elif DETECTOR_NAME == "retinaface":
        from retinaface import RetinaFace
        rgb = frame_bgr[:, :, ::-1]
        faces = RetinaFace.detect_faces(rgb)
        if not faces or isinstance(faces, tuple):
            return None
        best = max(faces.values(),
                   key=lambda f: (f["facial_area"][2] - f["facial_area"][0])
                               * (f["facial_area"][3] - f["facial_area"][1]))
        x1, y1, x2, y2 = best["facial_area"]
        return x1, y1, x2, y2

    else:  # haar
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = DETECTOR_OBJ.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, bw, bh = max(faces, key=lambda r: r[2] * r[3])
        return x, y, x + bw, y + bh


def crop_and_save(frame_bgr: np.ndarray, bbox, out_path: Path, face_size: int, margin: float = 0.3):
    """Crops face with margin, resizes to face_size×face_size, saves as JPEG."""
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    face_crop = frame_bgr[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (face_size, face_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ──────────────────────────────────────────────────────────────────────────────
# Video → face frames
# ──────────────────────────────────────────────────────────────────────────────

def extract_faces_from_video(video_path: Path, out_dir: Path,
                             frames_per_video: int, face_size: int) -> int:
    """
    Samples `frames_per_video` evenly-spaced frames from the video,
    detects + crops the face in each, and saves to out_dir.
    Returns number of frames successfully saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] Cannot open {video_path.name}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    sample_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
    saved = 0
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        bbox = detect_face(frame)
        if bbox is None:
            continue
        out_path = out_dir / f"{int(idx):06d}.jpg"
        if out_path.exists():  # skip already-done frames
            saved += 1
            continue
        crop_and_save(frame, bbox, out_path, face_size)
        saved += 1

    cap.release()
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers (adapted from the original FF++ script)
# ──────────────────────────────────────────────────────────────────────────────

def _reporthook(count, block_size, total_size):
    pct = min(100, int(count * block_size * 100 / max(total_size, 1)))
    mb = count * block_size / (1024 ** 2)
    sys.stdout.write(f"\r  Downloading … {pct:3d}%  {mb:.1f} MB")
    sys.stdout.flush()


def download_video(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return  # already downloaded
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp), reporthook=_reporthook)
        sys.stdout.write("\n")
        tmp.rename(dest)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise e


def get_filelist(base_url: str, dataset_path: str) -> list:
    """Returns list of bare filenames (no extension) for the dataset."""
    if "DeepFakeDetection" in dataset_path or "actors" in dataset_path:
        data = json.loads(urllib.request.urlopen(base_url + DEEPFAKES_DETECTION_URL).read())
        if "actors" in dataset_path:
            return data["actors"]
        return data["DeepFakesDetection"]
    elif "original" in dataset_path:
        pairs = json.loads(urllib.request.urlopen(base_url + FILELIST_URL).read())
        filelist = []
        for pair in pairs:
            filelist += pair
        return filelist
    else:
        pairs = json.loads(urllib.request.urlopen(base_url + FILELIST_URL).read())
        filelist = []
        for pair in pairs:
            filelist.append("_".join(pair))
            filelist.append("_".join(pair[::-1]))
        return filelist


# ──────────────────────────────────────────────────────────────────────────────
# Chunk-wise pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_dataset_chunked(dataset_name: str, base_url: str,
                             output_root: Path, compression: str,
                             chunk_size: int, frames_per_video: int,
                             face_size: int):
    dataset_path = DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}  ({dataset_path})")
    print(f"{'='*60}")

    # Fetch filelist from server
    print("Fetching filelist from server …")
    filelist = get_filelist(base_url, dataset_path)
    print(f"  → {len(filelist)} videos found")

    # Paths
    video_url_base = f"{base_url}{dataset_path}/{compression}/videos/"
    video_tmp_dir  = output_root / "tmp_videos" / dataset_name
    faces_dir      = output_root / "faces" / dataset_name
    done_log       = output_root / "faces" / f".done_{dataset_name}.txt"

    # Load already-done videos to allow resuming
    already_done: set = set()
    if done_log.exists():
        already_done = set(done_log.read_text().splitlines())
    remaining = [f for f in filelist if f not in already_done]
    print(f"  → {len(already_done)} already processed, {len(remaining)} remaining")

    # Split into chunks
    chunks = [remaining[i:i + chunk_size] for i in range(0, len(remaining), chunk_size)]
    print(f"  → {len(chunks)} chunks of ≤{chunk_size} videos each")

    for chunk_idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} videos) ---")

        # 1. Download chunk
        downloaded = []
        for fname in chunk:
            url  = video_url_base + fname + ".mp4"
            dest = video_tmp_dir / (fname + ".mp4")
            print(f"  ↓ {fname}.mp4")
            try:
                download_video(url, dest)
                downloaded.append((fname, dest))
            except Exception as e:
                print(f"  [error] Download failed for {fname}: {e}")

        # 2. Extract faces
        for fname, video_path in tqdm(downloaded, desc="  Extracting faces", leave=False):
            out_dir = faces_dir / fname
            n = extract_faces_from_video(video_path, out_dir, frames_per_video, face_size)
            if n > 0:
                with open(done_log, "a") as f:
                    f.write(fname + "\n")
            else:
                print(f"  [warn] No faces extracted from {fname}")

        # 3. Delete downloaded videos to free disk
        print(f"  🗑  Deleting {len(downloaded)} videos to free disk …")
        for _, video_path in downloaded:
            try:
                video_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"  [warn] Could not delete {video_path}: {e}")

        # Clean up empty tmp dir
        try:
            shutil.rmtree(video_tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Chunked FF++ download + face-crop pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_url", required=True,
                   help="Base URL from the FF++ access email (ends with /v3/)")
    p.add_argument("--output_root", default="/home/shlok/Project/ML_Project",
                   help="Root directory for all output")
    p.add_argument("--datasets", nargs="+",
                   default=["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
                   choices=list(DATASETS.keys()),
                   help="Which datasets to process")
    p.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"],
                   help="Video compression level")
    p.add_argument("--chunk_size", type=int, default=10,
                   help="Number of videos to download per chunk before deleting")
    p.add_argument("--frames_per_video", type=int, default=32,
                   help="Number of evenly-spaced frames to sample per video")
    p.add_argument("--face_size", type=int, default=224,
                   help="Output face crop size (pixels, square)")
    return p.parse_args()


def main():
    args = parse_args()

    # Normalise base_url
    base_url = args.base_url
    if not base_url.endswith("/"):
        base_url += "/"

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Output root : {output_root}")
    print(f"Detector    : {DETECTOR_NAME}")
    print(f"Datasets    : {args.datasets}")
    print(f"Compression : {args.compression}")
    print(f"Chunk size  : {args.chunk_size} videos")
    print(f"Frames/video: {args.frames_per_video}")
    print(f"Face size   : {args.face_size}×{args.face_size}")

    for dataset in args.datasets:
        process_dataset_chunked(
            dataset_name=dataset,
            base_url=base_url,
            output_root=output_root,
            compression=args.compression,
            chunk_size=args.chunk_size,
            frames_per_video=args.frames_per_video,
            face_size=args.face_size,
        )

    print("\n✅ All datasets processed.")
    print(f"   Face images saved to: {output_root / 'faces'}/")


if __name__ == "__main__":
    main()
