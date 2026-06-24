# Plaque Characterization (Downstream Task)

Volume-level **multi-label classification** of coronary plaque from CCTA, built
on the pretrained CORA encoder.

The model is a pretrained CORA encoder + global pooling + linear head
(`models.model.CORAClassifier`, two-logit head). The **entire CCTA volume** is
the input (no patch sampling). Two labels are predicted independently:

- **calcified plaque present** (0/1)
- **non-calcified plaque present** (0/1)

The 4-channel multi-window input (fat / soft tissue / angiographic /
calcification) is identical to the CORA pretraining input, so the pretrained
encoder receives the same intensity representation it was trained on.

## Data layout

Per-patient preprocessed NPZ files under an `npz_root`:

```
<npz_root>/
  <patient_id>/
    CTA_<patient_id>.npz     # arrays: CTA_HU (D,H,W, HU), CA (D,H,W, vessel mask)
```

Only `CTA_HU` is used by this task; the volume is windowed into 4 channels and
resized/padded to a fixed shape (default `128 x 128 x 128`) for batching.

## Excel index

An Excel file with one row per patient. Expected columns (defaults; override in
`dataset.py` if your sheet differs):

| Column                             | Meaning                                   |
| ---------------------------------- | ----------------------------------------- |
| `Deidentification Patient Name`    | Patient identifier (matches NPZ folder)   |
| `Calcium Plaque`                   | Calcified plaque label (any > 0 -> present) |
| `Soft Plaque`                      | Non-calcified plaque label (any > 0 -> present) |

Provide separate train / val / test index files (or let `train.py` carve a
held-out validation split from the training index via `--val_fraction`).

## Hyperparameters

Read from `../../configs/cora_config.yaml` under
`downstream.plaque_characterization` (epochs, batch_size, learning_rate,
weight_decay). Loss is `BCEWithLogitsLoss` (multi-label); optimizer is AdamW.

## Train

```bash
python train.py \
  --npz_root /path/to/npz \
  --train_index data/plaque_train.xlsx \
  --val_index data/plaque_val.xlsx \
  --pretrained checkpoints/cora_pretrained_best.pth \
  --output_dir checkpoints/plaque_characterization
```

Saves the best-by-validation-AUC checkpoint (`plaque_best.pth`), a latest
checkpoint, and a per-epoch `training_log.csv`.

## Evaluate

```bash
python eval.py \
  --npz_root /path/to/npz \
  --test_index data/plaque_test.xlsx \
  --checkpoint checkpoints/plaque_characterization/plaque_best.pth \
  --output_csv results/plaque_metrics.csv
```

Reports per-label and macro-averaged **AUROC, AUPRC, F1, accuracy** (scikit-learn)
and writes them to a CSV table.

## Reproducibility

The plaque characterization numbers reported in the manuscript are produced by
these scripts (`train.py` + `eval.py`) using the hyperparameters in
`configs/cora_config.yaml`. No constants are hard-coded outside that config.
