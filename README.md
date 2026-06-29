# CORA: COronary Representation learning via Abnormality synthesis for CAD analysis

CORA is a self-supervised framework for coronary CT angiography (CCTA). A 3D
Residual U-Net is pretrained to segment **synthetically inserted coronary
lesions** within anatomically constrained vessel regions, biasing
representations toward clinically relevant vascular pathology. The pretrained
encoder is then fine-tuned for four downstream tasks.

This repository contains the **complete** code for pretraining and all
downstream tasks, released for the Nature Medicine resubmission.

---

## Method overview

- **Synthesis engine** (`pretrain/dataset.py`): generates realistic calcified
  and non-calcified plaques with controlled HU distributions and irregular
  morphology (1–3 overlapping Gaussian blobs) inside coronary artery regions.
- **Multi-window input**: each volume is converted to a 4-channel input
  (fat, soft tissue, angiographic, calcification windows).
- **3D Residual U-Net** (`models/model.py`): 4-stage encoder-decoder
  (manuscript Table 2), pretrained on abnormality segmentation, with downstream
  heads for classification, dense segmentation, and multimodal fusion.

All hyperparameters live in a single source of truth:
[`configs/cora_config.yaml`](configs/cora_config.yaml). 

---

## Repository structure

```
configs/
  cora_config.yaml              # single source of truth for all hyperparameters
preprocessing/
  preprocess_save_npz.py        # resample to 0.5^3 mm, package CTA_HU + CA into NPZ
pretrain/
  dataset.py                    # lesion synthesis engine + multi-window input + noise
  losses.py                     # Tversky + Focal segmentation loss
  pretrain_cora.py              # pretraining w/ validation loop + early stopping + logging
models/
  model.py                      # backbone + downstream heads (shared by all tasks)
downstream/
  plaque_characterization/      # volume-level multi-label classification
  stenosis_detection/           # segmentation + lesion-level (connected-component) eval
  coronary_segmentation/        # ImageCAS; Dice / clDice / MSD
  mace_prediction/              # multimodal: image + frozen Qwen text + clinical MLP
baselines/
  README.md                     # MAE / VolumeFusion / VoCo (via nnssl) + from-scratch
label_extraction/
  plaque_stenosis_extraction_prompt.txt   # two-LLM (GPT-4o + Claude Sonnet 4.5) prompt
  plaque_label_schema.json                # structured-output JSON schema
checkpoints/
  README.md                     # pretrained-weights placeholders + load example
```

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch ≥ 2.0, MONAI, SimpleITK, `transformers`,
`dynamic-network-architectures`, scikit-image, scikit-learn.

---

## Data preparation

Each CCTA volume is preprocessed into a per-subject NPZ containing:
- `CTA_HU`: raw CCTA volume in Hounsfield Units `(D, H, W)`.
- `CA`: binary coronary artery mask `(D, H, W)` (e.g., nnU-Net trained on ImageCAS).

All volumes are resampled to an **isotropic `0.5 × 0.5 × 0.5 mm³`** grid.
See [`preprocessing/README.md`](preprocessing/README.md). A patient index file
(Excel) listing subject identifiers is required.

> **Pretraining cohort:** the pretraining index must contain **only pretraining
> patients** — all internal and external test
> patients are excluded. No script hard-codes a cohort count or a list file that
> would place test patients in the pretraining set.

---

## Pretraining

```bash
cd pretrain
python pretrain_cora.py \
    --config ../configs/cora_config.yaml \
    --excel_file data/CTA_all_list.xlsx \
    --npz_root /path/to/preprocessed/npz \
    --checkpoint_dir checkpoints/cora_pretrain
```

The pretraining script holds out a validation split, evaluates the
self-supervised objective each epoch, applies **early stopping** on the
validation loss, keeps the **best-by-validation checkpoint**
(`cora_pretrained_best.pth`), and logs per-epoch train/val loss to
`training_log.csv`.

### Key hyperparameters (from `configs/cora_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Voxel spacing | `0.5 × 0.5 × 0.5 mm³` |
| Patch size | `96 × 96 × 96` |
| Lesion HU (calcified / soft) | `800–1500` / `30–90` |
| Blob sigma / count | `0.7–2.0` / `1–3` |
| Noise (`I0`, `L`, `σ_e`) | `1e5`, `200.0 mm`, `2.0 HU` |
| Loss | Tversky (`α=0.1, β=0.9`) + Focal (`γ=4.0`) |
| Optimizer | AdamW, LR `1e-4`, warmup 3 ep, cosine decay |

---

## Downstream tasks

Each downstream module has its own `train.py`, `eval.py`, `dataset.py`, and
`README.md`. All load the pretrained encoder via `--pretrained`
(default `checkpoints/cora_pretrained_best.pth`).

| Task | Folder | Formulation | Metrics |
|------|--------|-------------|---------|
| Plaque characterization | `downstream/plaque_characterization/` | volume-level multi-label classification | AUROC / AUPRC / F1 |
| Stenosis detection | `downstream/stenosis_detection/` | segmentation + lesion-level matching | lesion sensitivity / precision / F1 (>10-voxel overlap = TP) |
| Coronary artery segmentation | `downstream/coronary_segmentation/` | ImageCAS dense segmentation (Dice loss) | Dice / clDice / MSD |
| MACE prediction | `downstream/mace_prediction/` | multimodal (image + frozen Qwen + clinical) | AUROC |

Example (plaque characterization):

```bash
cd downstream/plaque_characterization
python train.py --npz_root /path/to/npz --train_index data/train.xlsx \
                --val_index data/val.xlsx --pretrained ../../checkpoints/cora_pretrained_best.pth
python eval.py  --npz_root /path/to/npz --test_index data/test.xlsx \
                --checkpoint outputs/plaque_best.pth
```

To train a **from-scratch baseline**, omit `--pretrained` (random encoder init).

---

## Pretrained weights

Pretrained CORA weights and fine-tuned task checkpoints are released as external
assets; see [`checkpoints/README.md`](checkpoints/README.md) for links and a minimal load-and-run example.

---

## Label extraction

Volume-level plaque/stenosis labels were extracted from CCTA reports by two LLMs
independently (GPT-4o and Claude Sonnet 4.5) with structured JSON output; the
exact prompt and schema are in [`label_extraction/`](label_extraction/).
