#!/usr/bin/env python3
"""
evaluate_all.py

Evaluate every trained UDD model found directly under OUTPUTS_DIR on the
external test split only (CelebDF, DFD and DFDC).

Expected project layout:
    project/
      dataset_processed/
        dataset.csv
        dataset_split.json
        train/
        val/
        test/
      outputs/
        udd_full/
          best.pt
        udd_no_mix/
          best.pt
      udd_full_pipeline_fixed/
        data.py
        model.py
      evaluate_all.py

The test split in the split JSON is authoritative. The dataset CSV is loaded
as the master manifest and used to validate test entries. Training and
validation splits are not loaded or evaluated.

Only CelebDF, DFD and DFDC test entries are evaluated. Results are written to:
    outputs/results.json
    outputs/results.csv

Every frame in the external test split is evaluated. No training-time
per-video frame subsampling is applied.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all UDD checkpoints on the test split only, reporting "
            "CelebDF, DFD and DFDC separately."
        )
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing dataset_processed/, outputs/ and udd_full_pipeline_fixed/.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Processed dataset root. Default: <project_root>/dataset_processed.",
    )
    parser.add_argument(
        "--dataset_csv",
        type=Path,
        default=None,
        help="Master dataset CSV. Default: <data_root>/dataset.csv.",
    )
    parser.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Video-level split JSON. Default: <data_root>/dataset_split.json.",
    )
    parser.add_argument(
        "--outputs_dir",
        type=Path,
        default=None,
        help="Directory containing one subdirectory per model. Default: <project_root>/outputs.",
    )
    parser.add_argument(
        "--pipeline_dir",
        type=Path,
        default=None,
        help="Directory containing data.py and model.py. Default: <project_root>/udd_full_pipeline_fixed.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="Checkpoint filename to evaluate in each model directory.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=None,
                        help="Override checkpoint img_size; otherwise use checkpoint args or 224.")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--disable_cudnn", action="store_true",
                        help="Disable cuDNN (useful if the system cuDNN libraries fail to load).")
    parser.add_argument("--include_root_checkpoint", action="store_true",
                        help="Also evaluate a checkpoint stored directly in outputs_dir.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional model directory names to evaluate. Default: all subdirectories with the selected checkpoint.")
    parser.add_argument("--bad_list", type=Path, default=None,
                        help="Optional text file of bad relative image paths, one per line (or tab-separated with details).")
    parser.add_argument("--output_prefix", default="results",
                        help="Output basename. Default creates results.json and results.csv.")
    parser.add_argument("--allow_manifest_mismatch", action="store_true",
                        help="Do not stop if split JSON paths are absent from dataset.csv; still evaluate split JSON entries.")
    return parser.parse_args()


def resolve_paths(args):
    root = args.project_root.resolve()
    args.data_root = (args.data_root or root / "dataset_processed").resolve()
    args.dataset_csv = (args.dataset_csv or args.data_root / "dataset.csv").resolve()
    args.split_json = (args.split_json or args.data_root / "dataset_split.json").resolve()
    args.outputs_dir = (args.outputs_dir or root / "outputs").resolve()
    args.pipeline_dir = (args.pipeline_dir or root / "udd_full_pipeline_fixed").resolve()

    for path in (args.dataset_root if hasattr(args, "dataset_root") else args.data_root,
                 args.dataset_csv, args.split_json, args.outputs_dir, args.pipeline_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    sys.path.insert(0, str(args.pipeline_dir))


def load_project_modules(pipeline_dir):
    # Import after adding the project pipeline directory to sys.path.
    from data import FrameDataset, normalize_on_device
    from model import UDDViT
    return FrameDataset, normalize_on_device, UDDViT


def load_manifest(csv_path):
    manifest = pd.read_csv(csv_path)
    if manifest.empty:
        raise ValueError(f"Dataset manifest is empty: {csv_path}")

    path_column = "image_path" if "image_path" in manifest.columns else "path"
    if path_column not in manifest.columns:
        raise ValueError(
            f"{csv_path} must contain 'image_path' or 'path'. "
            f"Found: {list(manifest.columns)}"
        )

    manifest_paths = set(
        manifest[path_column].dropna().astype(str).map(
            lambda x: x.replace("\\\\", "/")
        )
    )
    return manifest, manifest_paths


def load_split_tables(split_json):
    """
    Read dataset_split.json entries containing image_path and label.
    Derive the dataset from the second path component and video ID from
    the relative parent directory.
    """
    if not split_json.is_file():
        raise FileNotFoundError(f"Split JSON not found: {split_json}")

    with split_json.open("r", encoding="utf-8") as handle:
        raw_splits = json.load(handle)

    tables = {}
    # Deliberately load only the external test split. Train and validation
    # entries are not parsed, validated, or passed to the model.
    for split_name in ("test",):
        entries = raw_splits.get(split_name)
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"Split '{split_name}' is missing, empty, or is not a list."
            )

        rows = []
        for index, item in enumerate(entries):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid entry at {split_name}[{index}]: expected an object."
                )

            image_path = item.get("image_path", item.get("path"))
            if not image_path:
                raise ValueError(
                    f"Entry {split_name}[{index}] has no 'image_path' or 'path'."
                )

            image_path = str(image_path).replace("\\\\", "/").lstrip("./")
            parts = Path(image_path).parts
            if len(parts) < 3:
                raise ValueError(f"Cannot infer dataset/video from path: {image_path}")

            # Layout: <split>/<dataset>/<...>/<video>/<frame>.
            dataset_name = parts[1]
            video_key = str(Path(*parts[:-1]))
            label = int(item["label"])
            if label not in (0, 1):
                raise ValueError(
                    f"Invalid binary label {label} at {split_name}[{index}]."
                )

            rows.append({
                "path": image_path,
                "image_path": image_path,
                "label": label,
                "ds": dataset_name,
                "video": video_key,
            })

        tables[split_name] = pd.DataFrame(rows).reset_index(drop=True)

    return tables


def validate_split_against_manifest(tables, manifest_paths, allow_mismatch=False):
    all_split_paths = set()
    for table in tables.values():
        all_split_paths.update(table["path"].astype(str))

    missing = sorted(all_split_paths - manifest_paths)
    if missing:
        message = (
            f"{len(missing)} path(s) referenced by dataset_split.json were not found "
            f"in dataset.csv. Examples: {missing[:10]}"
        )
        if allow_mismatch:
            print(f"[WARNING] {message}")
        else:
            raise ValueError(
                message + "\nIf the CSV and split JSON intentionally use different "
                "path conventions, rerun with --allow_manifest_mismatch."
            )
    else:
        print(
            f"Manifest validation passed: {len(all_split_paths):,} split image paths "
            f"are present in dataset.csv."
        )


def load_bad_paths(path):
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"Bad-path list not found: {path}")
    bad = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                bad.add(line.split("\t")[0].replace("\\", "/"))
    return bad


def choose_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def discover_model_dirs(outputs_dir, checkpoint_name, requested=None,
                        include_root_checkpoint=False):
    checkpoint_file = f"{checkpoint_name}.pt"
    model_dirs = []

    if include_root_checkpoint and (outputs_dir / checkpoint_file).is_file():
        model_dirs.append(outputs_dir)

    if requested:
        for name in requested:
            directory = outputs_dir / name
            if not directory.is_dir():
                raise FileNotFoundError(f"Requested model directory not found: {directory}")
            if not (directory / checkpoint_file).is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found for model '{name}': {directory / checkpoint_file}"
                )
            model_dirs.append(directory)
    else:
        for directory in sorted(outputs_dir.iterdir()):
            if directory.is_dir() and (directory / checkpoint_file).is_file():
                model_dirs.append(directory)

    # Preserve order while removing duplicate paths.
    unique = []
    seen = set()
    for directory in model_dirs:
        key = directory.resolve()
        if key not in seen:
            unique.append(directory)
            seen.add(key)

    if not unique:
        raise FileNotFoundError(
            f"No {checkpoint_file} checkpoints found in {outputs_dir}. "
            "Expected one checkpoint inside each model subdirectory."
        )
    return unique


def read_checkpoint(checkpoint_path, device):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    return checkpoint


def build_model(UDDViT, checkpoint, device, img_size_override=None):
    saved_args = checkpoint.get("args", {}) or {}

    arch = saved_args.get("arch", "vit_base_patch16_clip_224.openai")
    pretrained = not saved_args.get("no_pretrained", False)
    weights_file = saved_args.get("weights_file", None)
    lora_rank = int(saved_args.get("lora_rank", 4))
    lora_alpha = saved_args.get("lora_alpha", None)
    num_classes = 2
    grad_ckpt = bool(saved_args.get("grad_ckpt", True))

    model = UDDViT(
        arch,
        pretrained,
        weights_file,
        lora_rank,
        lora_alpha,
        num_classes,
        grad_ckpt,
    )

    state = checkpoint.get("model")
    if state is None:
        # Allow a plain state_dict checkpoint as a fallback.
        if all(isinstance(k, str) for k in checkpoint.keys()):
            state = checkpoint
        else:
            raise KeyError("Checkpoint does not contain a 'model' state dictionary.")

    if hasattr(model, "load_trainable"):
        model.load_trainable(state)
    else:
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model, saved_args


@torch.inference_mode()
def predict_split(model, table, FrameDataset, normalize_on_device,
                  data_root, img_size, batch_size, num_workers, device,
                  use_amp, bad_log):
    dataset = FrameDataset(
        table,
        str(data_root),
        int(img_size),
        train=False,
        bad_log=str(bad_log),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    probabilities = np.full(len(dataset), np.nan, dtype=np.float32)
    model.eval()

    for images, _labels, indices in tqdm(loader, desc="Inference", leave=False):
        images = normalize_on_device(images, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits, _features = model(images)

        batch_probs = F.softmax(logits.float(), dim=1)[:, 1]
        probabilities[indices.detach().cpu().numpy()] = (
            batch_probs.detach().cpu().numpy()
        )

    evaluated = table.copy().reset_index(drop=True)
    evaluated["prob_fake"] = probabilities
    evaluated = evaluated[np.isfinite(evaluated["prob_fake"])].copy()
    evaluated["prediction"] = (evaluated["prob_fake"] >= 0.5).astype(int)
    return evaluated, len(dataset) - len(evaluated)


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def metric_record(table, split, dataset_name):
    labels = table["label"].to_numpy(dtype=int)
    scores = table["prob_fake"].to_numpy(dtype=float)
    predictions = (scores >= 0.5).astype(int)

    # Video-level score is the mean fake probability across that video's frames.
    video_table = (
        table.groupby("video", dropna=False)
        .agg(prob_fake=("prob_fake", "mean"), label=("label", "first"))
        .reset_index()
    )
    video_labels = video_table["label"].to_numpy(dtype=int)
    video_scores = video_table["prob_fake"].to_numpy(dtype=float)
    video_predictions = (video_scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "split": split,
        "dataset": dataset_name,
        "frame_auc": safe_auc(labels, scores),
        "video_auc": safe_auc(video_labels, video_scores),
        "frame_accuracy": float(accuracy_score(labels, predictions)),
        "video_accuracy": float(accuracy_score(video_labels, video_predictions)),
        "frame_precision": float(precision_score(labels, predictions, zero_division=0)),
        "frame_recall": float(recall_score(labels, predictions, zero_division=0)),
        "frame_f1": float(f1_score(labels, predictions, zero_division=0)),
        "video_precision": float(
            precision_score(video_labels, video_predictions, zero_division=0)
        ),
        "video_recall": float(
            recall_score(video_labels, video_predictions, zero_division=0)
        ),
        "video_f1": float(
            f1_score(video_labels, video_predictions, zero_division=0)
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "frames": int(len(table)),
        "videos": int(len(video_table)),
        "real_frames": int((labels == 0).sum()),
        "fake_frames": int((labels == 1).sum()),
    }


def canonical_dataset_name(value):
    """Normalize common dataset-name variants for stable reporting."""
    raw = str(value).strip()
    key = "".join(ch for ch in raw.lower() if ch.isalnum())
    aliases = {
        "celebdf": "CelebDF",
        "celebdfv2": "CelebDF",
        "celebdf2": "CelebDF",
        "dfd": "DFD",
        "deepfakedetection": "DFD",
        "dfdc": "DFDC",
        "deepfakedetectionchallenge": "DFDC",
        "faceforensics": "FaceForensics++",
        "faceforensicsplusplus": "FaceForensics++",
        "ff": "FaceForensics++",
        "ffpp": "FaceForensics++",
    }
    return aliases.get(key, raw)


def evaluate_one_split(model, table, split_name, *predict_args):
    predictions, skipped = predict_split(model, table, *predict_args)
    records = []

    if not predictions.empty:
        predictions = predictions.copy()
        predictions["dataset_report_name"] = predictions["ds"].map(
            canonical_dataset_name
        )

        # Report only the three external test datasets; no aggregate ALL row
        # and no FaceForensics++ row are emitted.
        preferred = ["CelebDF", "DFD", "DFDC"]
        present = set(predictions["dataset_report_name"].dropna().unique())
        ordered_names = [name for name in preferred if name in present]

        for dataset_name in ordered_names:
            subset = predictions[
                predictions["dataset_report_name"] == dataset_name
            ]
            if not subset.empty:
                records.append(metric_record(subset, split_name, dataset_name))

    return records, predictions, skipped


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    resolve_paths(args)

    FrameDataset, normalize_on_device, UDDViT = (
        load_project_modules(args.pipeline_dir)
    )

    print("=" * 72)
    print("UDD EXTERNAL TEST-ONLY EVALUATION")
    print("=" * 72)
    print(f"Project root : {args.project_root.resolve()}")
    print(f"Dataset root : {args.data_root}")
    print(f"Dataset CSV  : {args.dataset_csv}")
    print(f"Split JSON   : {args.split_json}")
    print(f"Outputs      : {args.outputs_dir}")
    print(f"Checkpoint   : {args.checkpoint}.pt")

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    print(f"Device       : {device}")
    print(f"AMP          : {use_amp}")
    print(f"cuDNN        : {torch.backends.cudnn.enabled}")

    manifest, manifest_paths = load_manifest(args.dataset_csv)
    print(f"Manifest     : {len(manifest):,} rows")
    tables = load_split_tables(args.split_json)

    # Keep only the three external datasets requested for final reporting.
    # Normalize common folder-name variants before filtering.
    allowed_datasets = ["CelebDF", "DFD", "DFDC"]
    test_table = tables["test"].copy()
    test_table["ds"] = test_table["ds"].map(canonical_dataset_name)
    excluded_datasets = sorted(set(test_table["ds"]) - set(allowed_datasets))
    if excluded_datasets:
        print("Excluding non-target test datasets:", ", ".join(excluded_datasets))
    test_table = test_table[test_table["ds"].isin(allowed_datasets)].reset_index(drop=True)
    if test_table.empty:
        raise ValueError(
            "The test split contains no CelebDF, DFD or DFDC entries. "
            "Check the dataset folder names and split JSON paths."
        )
    present_datasets = set(test_table["ds"].unique())
    missing_datasets = [name for name in allowed_datasets if name not in present_datasets]
    if missing_datasets:
        print("[WARNING] No test entries found for:", ", ".join(missing_datasets))
    tables = {"test": test_table}

    validate_split_against_manifest(
        tables, manifest_paths, allow_mismatch=args.allow_manifest_mismatch
    )

    bad_paths = load_bad_paths(args.bad_list)
    if bad_paths:
        for split_name in tables:
            before = len(tables[split_name])
            tables[split_name] = tables[split_name][
                ~tables[split_name]["path"].isin(bad_paths)
            ].reset_index(drop=True)
            print(
                f"Removed {before - len(tables[split_name]):,} bad paths "
                f"from {split_name}."
            )

    print("\nSplit overview:")
    for split_name, table in tables.items():
        print(
            f"  {split_name:5s}: {len(table):,} frames | "
            f"{table['video'].nunique():,} videos | "
            f"{table['ds'].nunique():,} datasets | "
            f"fake fraction {table['label'].mean():.4f}"
        )

    model_dirs = discover_model_dirs(
        args.outputs_dir,
        args.checkpoint,
        requested=args.models,
        include_root_checkpoint=args.include_root_checkpoint,
    )
    print("\nModels to evaluate:")
    for directory in model_dirs:
        print(f"  - {directory.name}")

    all_records = []
    model_summaries = {}
    checkpoint_name = f"{args.checkpoint}.pt"

    for model_dir in model_dirs:
        model_name = model_dir.name
        checkpoint_path = model_dir / checkpoint_name
        print("\n" + "=" * 72)
        print(f"MODEL: {model_name}")
        print(f"CHECKPOINT: {checkpoint_path}")
        print("=" * 72)

        try:
            checkpoint = read_checkpoint(checkpoint_path, device)
            model, saved_args = build_model(
                UDDViT, checkpoint, device, args.img_size
            )
            img_size = (
                args.img_size
                if args.img_size is not None
                else int(saved_args.get("img_size", 224))
            )

            model_records = []
            skipped_by_split = {}

            # This evaluator intentionally runs test inference only.
            for split_name in ("test",):
                print(f"\nEvaluating {split_name.upper()} ONLY...")
                predict_args = (
                    FrameDataset,
                    normalize_on_device,
                    args.data_root,
                    img_size,
                    args.batch_size,
                    args.num_workers,
                    device,
                    use_amp,
                    model_dir / f"bad_files_{split_name}.txt",
                )
                records, _predictions, skipped = evaluate_one_split(
                    model, tables[split_name], split_name, *predict_args
                )
                skipped_by_split[split_name] = int(skipped)

                for record in records:
                    record["model"] = model_name
                    record["checkpoint"] = str(checkpoint_path)
                    model_records.append(record)
                    all_records.append(record)

                    print(
                        f"  {split_name:5s} | {record['dataset']:15s} | "
                        f"video AUC={record['video_auc']} | "
                        f"frame AUC={record['frame_auc']} | "
                        f"video acc={record['video_accuracy']:.4f} | "
                        f"frames={record['frames']:,} | videos={record['videos']:,}"
                    )

                print(
                    f"  Skipped/unloaded frames in {split_name}: {skipped:,}"
                )

            model_summaries[model_name] = {
                "status": "success",
                "checkpoint": str(checkpoint_path),
                "architecture": saved_args.get(
                    "arch", "vit_base_patch16_clip_224.openai"
                ),
                "img_size": img_size,
                "skipped_frames": skipped_by_split,
                "metrics": model_records,
            }

            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"\n[ERROR] Evaluation failed for {model_name}: {error}")
            traceback.print_exc()
            model_summaries[model_name] = {
                "status": "failed",
                "checkpoint": str(checkpoint_path),
                "error": error,
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Save machine-readable JSON and a flat CSV for spreadsheet analysis.
    json_path = args.outputs_dir / f"{args.output_prefix}.json"
    csv_path = args.outputs_dir / f"{args.output_prefix}.csv"

    payload = {
        "metadata": {
            "checkpoint_type": args.checkpoint,
            "dataset_csv": str(args.dataset_csv),
            "split_json": str(args.split_json),
            "data_root": str(args.data_root),
            "device": str(device),
            "amp": bool(use_amp),
            "evaluation_threshold": 0.5,
            "evaluation_split": "test",
            "evaluated_datasets": ["CelebDF", "DFD", "DFDC"],
            "frame_sampling": "all frames in the external test split",
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "models_discovered": [directory.name for directory in model_dirs],
        },
        "models": model_summaries,
        "results": all_records,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)

    results_df = pd.DataFrame(all_records)
    if not results_df.empty:
        preferred_columns = [
            "model", "split", "dataset", "frame_auc", "video_auc",
            "frame_accuracy", "video_accuracy", "frame_precision",
            "frame_recall", "frame_f1", "video_precision", "video_recall",
            "video_f1", "true_negative", "false_positive", "false_negative",
            "true_positive", "frames", "videos", "real_frames", "fake_frames",
            "checkpoint",
        ]
        results_df = results_df[
            [column for column in preferred_columns if column in results_df.columns]
        ]
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 72)
    print("TEST-ONLY EVALUATION FINISHED")
    print("=" * 72)
    print(f"JSON results: {json_path}")
    print(f"CSV results : {csv_path}")
    failed = [
        name for name, result in model_summaries.items()
        if result["status"] != "success"
    ]
    print(f"Successful models: {len(model_summaries) - len(failed)}/{len(model_summaries)}")
    if failed:
        print("Failed models:", ", ".join(failed))


if __name__ == "__main__":
    main()
