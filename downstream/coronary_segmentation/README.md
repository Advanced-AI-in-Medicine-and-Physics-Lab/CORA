# Coronary Artery Segmentation (Downstream Task)

Dense 3D segmentation of the coronary artery tree on the **ImageCAS** dataset.
The model is the full CORA U-Net (`CORASegmentationModel`): a **CORA-pretrained
encoder** plus a **randomly initialized decoder**, fine-tuned end-to-end with a
**Dice loss**.

All hyperparameters come from `../../configs/cora_config.yaml`
(`downstream.coronary_segmentation`); nothing is hard-coded in the scripts.

## Task formulation

- **Input:** an ImageCAS CT volume (single-channel HU). To match CORA
  pretraining and the other downstream tasks, the HU volume is converted on the
  fly to a **4-channel multi-window** representation (fat / soft-tissue /
  angiographic / calcification windows), so the pretrained encoder receives a
  matching input. The input channel count therefore comes from
  `model.num_input_channels` (= 4).
- **Target:** binary voxel-level coronary artery mask.
- **Training:** artery-centric 96^3 patches with positive/negative sampling
  (pos:neg = 7:2, matching CORA-v2) and geometric + intensity augmentation;
  Dice loss (`monai.losses.DiceLoss(sigmoid=True)`).
- **Inference:** MONAI sliding-window inference over the full volume.

The encoder is initialized from the CORA-pretrained weights; the decoder starts
from random initialization and is learned during fine-tuning.

## Evaluation metrics

| Metric | Definition |
| --- | --- |
| **Dice** | volumetric overlap `2|P∩G| / (|P|+|G|)` |
| **clDice** | centerline Dice: harmonic mean of topology precision/recall computed on the 3D skeletons of prediction and ground truth (Shit et al., CVPR 2021) |
| **MSD** | mean surface distance (symmetric), via `monai.metrics.SurfaceDistanceMetric` |

`clRecall` and `clPrecision` (the clDice components) are also reported per case.
The best training checkpoint is selected by mean **Dice**.

## Data layout

Each case is a single `.npz` file containing:

- `image`: single-channel CT volume in Hounsfield Units, shape `(D, H, W)`.
- `label`: binary coronary artery mask, shape `(D, H, W)`
  (override the key with `--label_key`).

Splits are specified with an index argument that is either:

- a **directory** containing `*.npz` files, or
- a **text file** listing one NPZ path per line (absolute, or relative to
  `--data_root`).

```
data/
  npz/                # NPZ files (referenced by the index files below)
  train_index.txt
  val_index.txt
```

## Usage

### Train

```bash
python train.py \
    --data_root data/npz \
    --train_index data/train_index.txt \
    --val_index data/val_index.txt \
    --pretrained checkpoints/cora_pretrained_best.pth \
    --output_dir checkpoints/coronary
```

Writes `coronary_best.pth` (best mean Dice), `coronary_latest.pth`, and
`training_log.csv` to `--output_dir`. If `--pretrained` is missing, the encoder
is trained from scratch (a warning is printed).

### Evaluate

```bash
python eval.py \
    --data_root data/npz \
    --val_index data/val_index.txt \
    --checkpoint checkpoints/coronary/coronary_best.pth \
    --output_dir results \
    --save_nii
```

Writes `summary_results.csv` (mean / std of Dice, clDice, MSD, clRecall,
clPrecision) and `per_case_results.csv` to `--output_dir`. With `--save_nii`,
ground-truth and predicted masks are saved as `.nii.gz` under
`results/predictions/`.

## Files

- `dataset.py` — ImageCAS NPZ loader, 4-channel multi-window input, 96^3 patch
  sampling, MONAI augmentation; train and eval loaders.
- `train.py` — fine-tuning loop, config-driven, Dice loss, sliding-window
  validation, best-by-Dice checkpointing.
- `eval.py` — sliding-window inference computing Dice, clDice, and MSD.
