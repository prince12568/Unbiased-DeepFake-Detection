
#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

LABELS = {
    "Celeb-real": "real",
    "YouTube-real": "real",
    "Celeb-synthesis": "fake",
}

OUT_SIZE = 224
CROP_MARGIN = 0.30
MIN_CONF = 0.95
MIN_FACE = 40
DET_MAX_SIDE = 640
BATCH = 16
JPEG_QUALITY = 95
PAD_MODE = cv2.BORDER_CONSTANT


# ============================================================
# FACE DETECTOR
# ============================================================

class FacePreprocessor:
    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Device: {self.device}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            thresholds=[0.6, 0.7, 0.7],
            min_face_size=MIN_FACE,
        )

    def detect_faces(self, frames_bgr):
        """
        Detect the largest sufficiently confident face in each frame.
        Returns bounding boxes in original-frame coordinates, or None.
        """
        pils = []
        scales = []

        for frame in frames_bgr:
            h, w = frame.shape[:2]
            scale = min(1.0, DET_MAX_SIDE / max(h, w))

            if scale < 1.0:
                small = cv2.resize(
                    frame,
                    (
                        max(1, int(w * scale)),
                        max(1, int(h * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small = frame

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pils.append(Image.fromarray(rgb))
            scales.append(scale)

        boxes, probs = self.mtcnn.detect(pils)

        results = []

        for frame_boxes, frame_probs, scale in zip(
            boxes, probs, scales
        ):
            if frame_boxes is None or frame_probs is None:
                results.append(None)
                continue

            best_box = None
            best_area = 0

            for box, confidence in zip(frame_boxes, frame_probs):
                if confidence is None or confidence < MIN_CONF:
                    continue

                x1, y1, x2, y2 = box
                fw = x2 - x1
                fh = y2 - y1

                if min(fw, fh) < MIN_FACE * 0.8:
                    continue

                area = fw * fh

                if area > best_area:
                    best_box = box
                    best_area = area

            if best_box is None:
                results.append(None)
            else:
                results.append(np.asarray(best_box) / scale)

        return results

    @staticmethod
    def crop_and_pad(frame, box):
        """
        Create a square crop around the detected face, add a margin,
        pad out-of-frame regions with black, and resize to OUT_SIZE.
        """
        h, w = frame.shape[:2]

        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        side = max(x2 - x1, y2 - y1) * (
            1 + 2 * CROP_MARGIN
        )

        sx1 = int(round(cx - side / 2))
        sy1 = int(round(cy - side / 2))
        sx2 = int(round(cx + side / 2))
        sy2 = int(round(cy + side / 2))

        pad_left = max(0, -sx1)
        pad_top = max(0, -sy1)
        pad_right = max(0, sx2 - w)
        pad_bottom = max(0, sy2 - h)

        if pad_left or pad_top or pad_right or pad_bottom:
            frame = cv2.copyMakeBorder(
                frame,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                PAD_MODE,
                value=(0, 0, 0),
            )

            sx1 += pad_left
            sx2 += pad_left
            sy1 += pad_top
            sy2 += pad_top

        face = frame[sy1:sy2, sx1:sx2]

        if face.size == 0:
            raise ValueError("Empty face crop")

        return cv2.resize(
            face,
            (OUT_SIZE, OUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def read_frame(cap, idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        return frame if success else None

    def extract_faces(self, video_path, output_dir, n=64):
        """
        Extract up to n face images from a video.

        Frames are sampled evenly first. If fewer than n faces are
        found, remaining frames are tried in a deterministic random
        order until n faces are found or all frames are exhausted.
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return 0

        # Evenly spaced primary samples.
        base = np.unique(
            np.linspace(
                0,
                total - 1,
                min(n, total),
            ).astype(int)
        )

        # Backup frames, in reproducible random order.
        rest = np.setdiff1d(
            np.arange(total),
            base,
        )

        np.random.RandomState(0).shuffle(rest)
        order = np.concatenate([base, rest])

        # frame_idx -> cropped face
        good = {}
        position = 0

        while len(good) < n and position < len(order):
            indices = order[position:position + BATCH]
            position += BATCH

            frames = []
            valid_indices = []

            for idx in indices:
                frame = self.read_frame(cap, idx)

                if frame is not None:
                    frames.append(frame)
                    valid_indices.append(idx)

            if not frames:
                continue

            detected = self.detect_faces(frames)

            for idx, frame, box in zip(
                valid_indices, frames, detected
            ):
                if box is None:
                    continue

                if len(good) >= n:
                    break

                try:
                    good[idx] = self.crop_and_pad(frame, box)
                except (ValueError, cv2.error):
                    continue

        cap.release()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = 0

        for k, idx in enumerate(sorted(good)):
            image_path = output_dir / f"{k:02d}_f{idx:04d}.jpg"

            success = cv2.imwrite(
                str(image_path),
                good[idx],
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            if success:
                saved += 1

        return saved


# ============================================================
# ZIP DATASET PROCESSING
# ============================================================

def find_videos(zip_file, categories):
    """Return sorted MP4 paths inside the archive for each category."""
    videos = {category: [] for category in categories}

    with zipfile.ZipFile(zip_file, "r") as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            parts = name.split("/")

            if len(parts) < 2 or not name.lower().endswith(".mp4"):
                continue

            category = parts[0]

            if category in videos:
                videos[category].append(name)

    for category in videos:
        videos[category].sort()

    return videos


def already_done(folder, frames_per_video):
    """
    Skip completed videos when enough readable JPEG files already
    exist. Incomplete folders are reprocessed.
    """
    if not folder.is_dir():
        return False

    images = list(folder.glob("*.jpg"))

    if len(images) < frames_per_video:
        return False

    # Check image readability instead of trusting file count alone.
    for path in images:
        image = cv2.imread(str(path))

        if image is None:
            return False

    return True


def process_video(
    archive,
    video_name,
    output_root,
    preprocessor,
    frames_per_video,
    temp_dir,
):
    category = video_name.split("/")[0]
    stem = Path(video_name).stem

    drive_folder = output_root / category / stem

    if already_done(drive_folder, frames_per_video):
        return "skipped", frames_per_video

    # Use a temporary local file so OpenCV can decode the video.
    temp_video = temp_dir / f"{category}_{stem}.mp4"
    local_folder = temp_dir / "faces" / category / stem

    try:
        temp_video.parent.mkdir(parents=True, exist_ok=True)

        with archive.open(video_name, "r") as src:
            with open(temp_video, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

        if local_folder.exists():
            shutil.rmtree(local_folder)

        count = preprocessor.extract_faces(
            temp_video,
            local_folder,
            n=frames_per_video,
        )

        drive_folder.parent.mkdir(parents=True, exist_ok=True)

        if drive_folder.exists():
            shutil.rmtree(drive_folder)

        # Save the extracted faces directly to the output directory.
        shutil.copytree(local_folder, drive_folder)

        return (
            "processed",
            count,
        )

    finally:
        if temp_video.exists():
            temp_video.unlink()

        if local_folder.exists():
            shutil.rmtree(local_folder, ignore_errors=True)


# ============================================================
# MANIFEST GENERATION
# ============================================================

def generate_manifest(output_root):
    manifest_path = output_root / "manifest.csv"
    rows = 0

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "path",
            "video",
            "category",
            "label",
        ])

        for category, label in LABELS.items():
            category_dir = output_root / category

            if not category_dir.is_dir():
                continue

            for video_dir in sorted(category_dir.iterdir()):
                if not video_dir.is_dir():
                    continue

                for image_path in sorted(video_dir.glob("*.jpg")):
                    relative_path = image_path.relative_to(
                        output_root
                    ).as_posix()

                    writer.writerow([
                        relative_path,
                        video_dir.name,
                        category,
                        label,
                    ])

                    rows += 1

    return manifest_path, rows


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract face crops from Celeb-DF-v2 videos "
            "in chunks and save them locally."
        )
    )

    parser.add_argument(
        "--zip_path",
        type=Path,
        default=None,
        help=(
            "Path to Celeb-DF-v2.zip. If omitted, search the "
            "output directory and current working directory."
        ),
    )

    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Directory where extracted faces and manifest are saved.",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of videos processed per chunk.",
    )

    parser.add_argument(
        "--frames_per_video",
        type=int,
        default=64,
        help="Maximum number of face images to extract per video.",
    )

    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional limit per category for testing.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU processing even if CUDA is available.",
    )

    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error("--chunk_size must be at least 1")

    if args.frames_per_video < 1:
        parser.error("--frames_per_video must be at least 1")

    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max_videos must be at least 1")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.zip_path is not None:
        zip_path = args.zip_path.expanduser().resolve()
    else:
        candidates = [
            output_root / "Celeb-DF-v2.zip",
            Path.cwd() / "Celeb-DF-v2.zip",
        ]

        zip_path = next(
            (path.resolve() for path in candidates if path.is_file()),
            None,
        )

    if zip_path is None or not zip_path.is_file():
        parser.error(
            "Dataset ZIP not found. Supply --zip_path /path/to/"
            "Celeb-DF-v2.zip"
        )

    if zip_path.resolve() == (output_root / "manifest.csv").resolve():
        parser.error("Invalid ZIP path.")

    temp_dir = output_root / ".preprocess_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.cpu else None
    preprocessor = FacePreprocessor(device=device)

    print(f"Dataset ZIP : {zip_path}")
    print(f"Output root : {output_root}")
    print(f"Chunk size  : {args.chunk_size}")
    print(f"Frames/video: {args.frames_per_video}")

    videos = find_videos(zip_path, LABELS.keys())

    all_videos = []

    for category in LABELS:
        category_videos = videos[category]

        if args.max_videos is not None:
            category_videos = category_videos[:args.max_videos]

        print(f"{category}: {len(category_videos)} videos")
        all_videos.extend(category_videos)

    processed = 0
    skipped = 0
    failed = 0
    short_videos = []

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            total_chunks = (
                len(all_videos) + args.chunk_size - 1
            ) // args.chunk_size

            for chunk_idx in range(total_chunks):
                start = chunk_idx * args.chunk_size
                chunk = all_videos[
                    start:start + args.chunk_size
                ]

                print(
                    f"\nChunk {chunk_idx + 1}/{total_chunks} "
                    f"({len(chunk)} videos)"
                )

                for video_name in tqdm(
                    chunk,
                    desc=f"Chunk {chunk_idx + 1}",
                    unit="video",
                ):
                    try:
                        status, count = process_video(
                            archive=archive,
                            video_name=video_name,
                            output_root=output_root,
                            preprocessor=preprocessor,
                            frames_per_video=args.frames_per_video,
                            temp_dir=temp_dir,
                        )

                        if status == "skipped":
                            skipped += 1
                        else:
                            processed += 1

                            if count < args.frames_per_video:
                                short_videos.append(
                                    (video_name, count)
                                )

                    except Exception as exc:
                        failed += 1
                        print(
                            f"\nFAILED: {video_name}\n"
                            f"Reason: {exc}"
                        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    manifest_path, image_count = generate_manifest(output_root)

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Videos processed : {processed}")
    print(f"Videos skipped   : {skipped}")
    print(f"Videos failed    : {failed}")
    print(f"Short videos     : {len(short_videos)}")
    print(f"Face images      : {image_count}")
    print(f"Manifest         : {manifest_path}")

    if short_videos:
        print(
            f"\nVideos with fewer than "
            f"{args.frames_per_video} face images:"
        )

        for name, count in short_videos[:20]:
            print(f"  {name}: {count} images")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()

