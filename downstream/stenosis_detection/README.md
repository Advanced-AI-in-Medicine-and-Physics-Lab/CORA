# Stenosis Detection (Downstream Task)

Coronary stenosis detection cast as a **voxel-level segmentation** problem and
evaluated at the **lesion level** by connected-component matching. The model is
the full CORA U-Net (`CORASegmentationModel`): a CORA-pretrained encoder plus a
randomly initialized decoder, fine-tuned end-to-end.

All hyperparameters come from `../../configs/cora_config.yaml`
(`downstream.stenosis_detection`); nothing is hard-coded in the scripts.

## Task formulation

- **Input:** coronary-artery CCTA volume, converted to a 4-channel multi-window
  representation (fat / soft-tissue / angiographic / calcification windows),
  matching pretraining.
- **Target:** voxel-level stenotic-lesion mask.
- **Training:** lesion-centric 96^3 patches with positive/negative sampling and
  geometric + intensity augmentation; recall-prioritized lesion segmentation
  loss `L = Tversky(alpha=0.1, beta=0.9) + Focal(gamma=4.0)`.
- **Inference:** MONAI sliding-window inference over the full volume.

## Lesion-level evaluation protocol

Predictions and ground truth are binarized and decomposed into 3D connected
components (26-connectivity). Matching uses an overlap-count rule:

> A predicted lesion overlapping a ground-truth lesion by **more than 10 voxels**
> (`overlap_threshold_voxels: 10`) counts as a **true positive (TP)**.

- **FP:** a predicted lesion that overlaps no GT lesion by more than 10 voxels.
- **FN:** a GT lesion that is overlapped by no prediction by more than 10 voxels.
- Overlap is measured against a slightly **dilated** GT mask
  (`--dilation_iter`, default 1) to tolerate boundary annotation uncertainty,
  faithful to the original CORA-v2 `LesionDetectionMetric`.

Reported metrics:

| Metric | Definition |
| --- | --- |
| Sensitivity (recall) | detected_GT / total_GT |
| Precision | TP / (TP + FP) |
| F1 | harmonic mean of precision and sensitivity |
| Voxel Dice | mean per-case voxel-level Dice |
| Voxel specificity | voxel TN / (voxel TN + voxel FP) over GT-negative voxels |

The best training checkpoint is selected by **lesion-level F1**.

## Data layout

Each case is a single `.npz` file containing:

- `image`: single-channel CCTA volume in Hounsfield Units, shape `(D, H, W)`.
- `label`: binary voxel-level stenotic-lesion mask, shape `(D, H, W)`
  (override the key with `--label_key`).

Splits are specified with an index argument that is either:

- a **directory** containing `*.npz` files, or
- a **text file** listing one NPZ path per line (absolute, or relative to
  `--npz_root`).

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
    --npz_root data/npz \
    --train_index data/train_index.txt \
    --val_index data/val_index.txt \
    --pretrained checkpoints/cora_pretrained_best.pth \
    --output_dir checkpoints/stenosis
```

Writes `stenosis_best.pth` (best lesion-level F1), `stenosis_latest.pth`, and
`training_log.csv` to `--output_dir`. If `--pretrained` is missing, the encoder
is trained from scratch (a warning is printed).

### Evaluate

```bash
python eval.py \
    --npz_root data/npz \
    --val_index data/val_index.txt \
    --checkpoint checkpoints/stenosis/stenosis_best.pth \
    --output_dir results
```

Writes `summary_results.csv` (aggregate lesion-level + voxel metrics) and
`per_case_results.csv` to `--output_dir`.

## Files

- `dataset.py` — NPZ loader, 4-channel multi-window input, 96^3 patch sampling,
  MONAI augmentation; train and eval loaders.
- `train.py` — fine-tuning loop, config-driven, best-by-F1 checkpointing.
- `eval.py` — sliding-window inference + lesion-level `LesionDetectionMetric`.
