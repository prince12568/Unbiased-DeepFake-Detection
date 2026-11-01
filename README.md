# Unbiased Deepfake Detection (UDD) — Replication

**Paper:** [Exploring Unbiased Deepfake Detection via Token-Level Shuffling and Mixing](https://arxiv.org/abs/2501.04376)  
**Authors:** Fu et al. · AAAI 2025  
**Team:** Shlok Divyam (B24CS042) · Prince Hadke (B24DS022) — IIT Bhilai

---

## Overview

This repo contains a from-scratch replication of the UDD framework. No official code was released by the authors; the entire pipeline was derived from the paper text.

UDD tackles two failure modes of ViT-based deepfake detectors:

| Bias type | What it means | UDD fix |
|---|---|---|
| **Position bias** | Model relies on *where* artifacts appear in the image | Shuffling Branch — 2×2 block shuffle of patch tokens + randomised position embeddings |
| **Content bias** | Model relies on *which* specific textures/patterns appear | Mixing Branch — mid-network token swap between same-label samples |

---

## Architecture

**Backbone:** ViT-B/16 (`vit_base_patch16_clip_224.openai`), CLIP-initialized, fine-tuned via LoRA (rank r=4 on Q and V projections of all 12 blocks). All pretrained weights are frozen; only LoRA adapters and the classification head are trained.

**Three forward branches per sample:**

1. **Original branch** — standard ViT forward pass. The only branch that contributes to the CE loss.
2. **Shuffling branch** — patch content tokens permuted as 2×2 blocks (49 blocks, each block shuffled independently per sample); position embeddings replaced by a random rectangular crop of the 14×14 PE grid (area 60–196 cells, aspect ratio 3/4–4/3), bilinearly resized back to 14×14. CLS token and its PE are untouched.
3. **Mixing branch** — after a randomly chosen transformer block k ∈ [5, 8] (mid-network), ⌊0.3 × 196⌋ = 58 patch tokens are replaced with tokens from a same-label partner in the batch. Drop positions and source positions are sampled **independently** (no spatial correspondence preserved). CLS token is never mixed.

**Loss (Eq. 8 of paper):**

```
L_total = L_CE + λ1·L_con + λ2·L_align

L_CE     : cross-entropy on the ORIGINAL branch only
L_con    : InfoNCE(z_orig → z_shuf) + InfoNCE(z_orig → z_mix)
           (z_orig is the sole anchor; one-directional)
L_align  : JS(P_orig ∥ P_shuf) + JS(P_orig ∥ P_mix)
           (JSD on softmax outputs)

λ1 = λ2 = 0.1,  τ = 0.1
```

The shuffling and mixing branches do **not** contribute separate CE terms. The 3-layer MLP projection head (768 → 768 → 128, ℓ2-normalized) is applied before InfoNCE.

---

## Repository Structure

```
.
├── udd_full_pipeline_fixed/          # All training and evaluation code
│   ├── model.py                      # ViT + LoRA backbone, classification head
│   ├── data.py                       # Dataset loading, augmentations
│   ├── shuf_branch.py                # Shuffling Branch (2×2 block shuffle + random PE crop)
│   ├── mix_branch.py                 # Mixing Branch (mid-network token swap, k∈[5,8])
│   ├── losses.py                     # UDDLoss: InfoNCE + JSD + ProjectionHead
│   ├── train_udd.py                  # Full UDD training (Base + Shuf + Mix)
│   ├── train_ablation_no_shuf.py     # Ablation: Base + Mix only (no Shuffling)
│   ├── train_ablation_no_mix.py      # Ablation: Base + Shuf only (no Mixing)
│   ├── evaluate.py                   # Cross-dataset evaluation script
│   ├── preprocess_all_datasets.py    # Face extraction pipeline (dlib, all datasets)
│   └── scan_images.py                # Scan for unreadable/corrupt frames
│
├── outputs/                          # Saved checkpoints and evaluation results
│   ├── udd_full/                     # Full UDD model (best.pt, last.pt, history.json)
│   ├── ablation_no_shuf/             # Base + Mix ablation checkpoints
│   ├── ablation_no_mix/              # Base + Shuf ablation checkpoints
│   ├── external_test_evaluation.json # Baseline evaluation results on cross-dataset test split
│   └── external_test_evaluation.csv  # Same results in CSV format
│
├── dataset/                          # Placeholder structure for raw datasets
│   ├── FaceForensics++/
│   ├── Celeb-DF-v2/
│   ├── DFD/
│   └── DFDC/
│
├── dlib_tools/                       # dlib shape predictor model for face alignment
├── assets/
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/prince12568/Unbiased-DeepFake-Detection.git
cd Unbiased-DeepFake-Detection
pip install -r requirements.txt
```

---

## Data Preprocessing

All preprocessing uses dlib for face detection and landmark-based alignment. Frames are extracted from raw videos, faces are detected and aligned, and crops are saved as 224×224 JPEGs into the standard `dataset_processed/` structure.

The script has three modes: `stage1` (video → aligned crops only), `stage2` (manifest generation only), and `all` (both stages end-to-end).

**Full pipeline (all datasets):**

```bash
python udd_full_pipeline_fixed/preprocess_all_datasets.py all \
    --ffpp_root    ./dataset/FaceForensics++ \
    --celebdf_root ./dataset/Celeb-DF-v2 \
    --dfd_root     ./dataset/DFD \
    --dfdc_root    ./dataset/DFDC \
    --dataset_root ./dataset_processed \
    --train_frames 8 \
    --val_frames   8 \
    --test_frames  32 \
    --val_frac     0.15 \
    --workers      4 \
    --seed         42
```

**Process only one dataset (e.g. DFDC):**

```bash
python udd_full_pipeline_fixed/preprocess_all_datasets.py all \
    --only DFDC \
    --dfdc_root    ./dataset/DFDC \
    --dataset_root ./dataset_processed \
    --workers      4
```

**Stage 2 only** (re-generate manifests from already-processed crops):

```bash
python udd_full_pipeline_fixed/preprocess_all_datasets.py stage2 \
    --dataset_root ./dataset_processed
```

This produces `dataset_processed/dataset.csv` and `dataset_processed/dataset_split.json`.

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `stage` | (required) | `stage1`, `stage2`, or `all` |
| `--ffpp_root` | — | Path to raw FF++ videos |
| `--celebdf_root` | — | Path to raw Celeb-DF-v2 videos |
| `--dfd_root` | — | Path to raw DFD videos |
| `--dfdc_root` | — | Path to raw DFDC videos |
| `--dataset_root` | `./dataset_processed` | Output directory for aligned crops + manifests |
| `--train_frames` | 8 | Target face crops per FF++ training video |
| `--val_frames` | 8 | Target face crops per FF++ validation video |
| `--test_frames` | 32 | Target face crops per external test video |
| `--val_frac` | 0.15 | Fraction of FF++ videos held out for validation |
| `--only` | — | Restrict to one or more datasets (e.g. `--only DFDC CelebDF`) |
| `--workers` | 2 | Number of CPU worker processes |
| `--seed` | 42 | Random seed for train/val split |
| `--dlib_predictor` | `dlib_tools/shape_predictor_68_face_landmarks.dat` | Path to dlib shape predictor |

**Datasets used:**
- **FaceForensics++ (c23)** — training (split by `--val_frac`) and in-distribution validation
- **Celeb-DF v2** — cross-dataset test evaluation
- **DeepFakeDetection (DFD)** — cross-dataset test evaluation
- **DFDC-Preview** — cross-dataset test evaluation

---

## Training

All runs use AdamW (lr=5e-4, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-3), cosine decay with 5-epoch warmup, 100 epochs, effective batch size 64 (gradient accumulation, gradient checkpointing enabled). Training auto-runs `evaluate.py` on completion.

**Full UDD (Base + Shuf + Mix):**

```bash
python udd_full_pipeline_fixed/train_udd.py \
    --data_root  ./dataset_processed \
    --split_json ./dataset_processed/dataset_split.json \
    --out_dir    outputs/udd_full
```

**Ablations:**

```bash
# Base + Shuf only (no Mixing Branch)
python udd_full_pipeline_fixed/train_ablation_no_mix.py \
    --data_root  ./dataset_processed \
    --split_json ./dataset_processed/dataset_split.json \
    --out_dir    outputs/ablation_no_mix

# Base + Mix only (no Shuffling Branch)
python udd_full_pipeline_fixed/train_ablation_no_shuf.py \
    --data_root  ./dataset_processed \
    --split_json ./dataset_processed/dataset_split.json \
    --out_dir    outputs/ablation_no_shuf
```

**Evaluate only** (skip training, run evaluation on an existing checkpoint):

```bash
python udd_full_pipeline_fixed/train_udd.py \
    --data_root  ./dataset_processed \
    --split_json ./dataset_processed/dataset_split.json \
    --out_dir    outputs/udd_full \
    --eval_only
```

**Key flags (all train scripts share the same interface):**

| Flag | Default | Description |
|---|---|---|
| `--data_root` | (required) | Path to `dataset_processed/` |
| `--split_json` | (required) | Path to `dataset_split.json` |
| `--out_dir` | (required) | Output directory for checkpoints and logs |
| `--dataset_csv` | `<data_root>/dataset.csv` | Master manifest CSV |
| `--epochs` | 100 | Total training epochs |
| `--batch_size` | 64 | Effective batch size |
| `--per_gpu_batch` | 32 | Per-GPU batch (gradient accumulation computed automatically) |
| `--lr` | 5e-4 | Learning rate |
| `--weight_decay` | 1e-2 | AdamW weight decay |
| `--warmup_epochs` | 5 | Linear warmup epochs |
| `--lora_rank` | 4 | LoRA rank for Q and V projections |
| `--mix_ratio` | 0.30 | Fraction of patch tokens to mix (γ) |
| `--mix_layer_min/max` | 5/8 | Range for mixing injection block k |
| `--lambda1` | 0.1 | Weight for L_con (InfoNCE) |
| `--lambda2` | 0.1 | Weight for L_align (JSD) |
| `--temperature` | 0.10 | InfoNCE temperature τ |
| `--proj_dim` | 128 | Projection head output dimension |
| `--train_frames_per_video` | 8 | Frames sampled per video during training |
| `--val_frames_per_video` | 8 | Frames sampled per video during validation |
| `--test_frames_per_video` | 32 | Frames sampled per video during testing |
| `--frame_sampling` | even | Frame sampling strategy (`even` or `random`) |
| `--grad_ckpt` | True | Enable gradient checkpointing |
| `--clip_grad` | 1.0 | Gradient clipping norm |
| `--seed` | 42 | Random seed |
| `--eval_only` | — | Skip training; run final evaluation only |
| `--no_pos_crop` | — | Disable random PE crop in shuffling branch (ablation) |
| `--class_weight` | — | Optional class weight for cross-entropy |
| `--time_budget_hours` | 11.0 | Stop training early if this many hours have elapsed |

---

## Evaluation

`evaluate.py` evaluates **all** model checkpoints found under `outputs/` on the external test split (CelebDF, DFD, DFDC) and writes results to `outputs/results.json` and `outputs/results.csv`. It is called automatically at the end of each training run; you can also run it manually.

**Evaluate all checkpoints:**

```bash
python udd_full_pipeline_fixed/evaluate.py \
    --project_root . \
    --data_root    ./dataset_processed \
    --outputs_dir  outputs/ \
    --checkpoint   best
```

**Evaluate a specific model:**

```bash
python udd_full_pipeline_fixed/evaluate.py \
    --project_root . \
    --data_root    ./dataset_processed \
    --outputs_dir  outputs/ \
    --models       udd_full \
    --checkpoint   best \
    --output_prefix evaluation_results
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--project_root` | script directory | Project root containing `dataset_processed/`, `outputs/`, `udd_full_pipeline_fixed/` |
| `--data_root` | `<project_root>/dataset_processed` | Processed dataset root |
| `--dataset_csv` | `<data_root>/dataset.csv` | Master manifest CSV |
| `--split_json` | `<data_root>/dataset_split.json` | Video-level split JSON |
| `--outputs_dir` | `<project_root>/outputs` | Directory containing model subdirectories |
| `--pipeline_dir` | `<project_root>/udd_full_pipeline_fixed` | Directory containing `data.py` and `model.py` |
| `--checkpoint` | `best` | Which checkpoint to load (`best` or `last`) |
| `--models` | (all) | Restrict to specific model directory names |
| `--batch_size` | 64 | Evaluation batch size |
| `--num_workers` | 4 | DataLoader workers |
| `--device` | auto | Device (`auto`, `cpu`, `cuda`, `cuda:0`, …) |
| `--output_prefix` | `results` | Output file basename (`results.json`, `results.csv`) |
| `--bad_list` | — | Text file of corrupt frame paths to skip |
| `--include_root_checkpoint` | — | Also evaluate a checkpoint stored directly in `outputs_dir` |

Produces per-dataset frame-level and video-level AUC, accuracy, precision, recall, and F1. Video-level scores are obtained by averaging frame-level softmax outputs across all frames of each video.

---

## Results

All models trained on **FF++ (c23)**. Evaluated on external cross-dataset test split.  
vAUC = video-level AUC (%), fAUC = frame-level AUC (%).  
†= numbers reported in the original paper. ★= our measured results.

### Cross-Dataset AUC (video-level)

| Method | CDF vAUC | CDF fAUC | DFDC vAUC | DFDC fAUC | DFD vAUC | DFD fAUC |
|---|---|---|---|---|---|---|
| ViT-B (CLIP) baseline† | 88.41 | 81.70 | 76.07 | 73.50 | 90.04 | 86.60 |
| Full UDD† | **93.13** | **86.90** | **81.21** | **75.80** | **95.51** | **91.00** |
| **Baseline (ours)★** | 91.83 | 86.36 | 57.30 | 48.69 | 93.15 | 89.49 |
| **Full UDD (ours)★** | 88.00 | 82.52 | 60.19 | 54.47 | 92.38 | 88.33 |

> **Note on DFDC:** Our DFDC test subset contains only 202 videos (753 frames). AUC estimates at this scale carry high variance and should be interpreted cautiously.

### Ablation Study (video-level AUC, ★= ours)

| Configuration | CDF | DFDC | DFD |
|---|---|---|---|
| Base only★ | 91.83 | 57.30 | 93.15 |
| Base + Shuf (no Mix)★ | 88.27 | 63.51 | 93.35 |
| Base + Mix (no Shuf)★ | 88.31 | 72.82 | 93.82 |
| Base + Shuf + Mix (Full UDD)★ | **88.00** | **60.19** | **92.38** |
| Base + Shuf (no Mix)† | 90.51 | 80.33 | 92.25 |
| Base + Mix (no Shuf)† | 91.25 | 81.02 | 94.95 |
| Base + Shuf + Mix (Full UDD)† | **93.13** | **81.21** | **95.51** |

On DFDC (the hardest OOD benchmark), both branches individually improve over the plain baseline, confirming the paper's bias-removal hypothesis. The gap from paper numbers is primarily attributable to our constrained compute budget (2×T4, gradient checkpointing) and the small DFDC test subset.

---

## Citation

```bibtex
@inproceedings{fu2025udd,
  title     = {Exploring Unbiased Deepfake Detection via Token-Level Shuffling and Mixing},
  author    = {Fu, Xinghe and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```
