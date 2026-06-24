# Multimodal MACE Prediction (Downstream Task)

Survival / **MACE (major adverse cardiac event) prediction** from CCTA, fusing
three modalities for each patient:

- **Image** — the pretrained CORA image encoder over the whole CCTA volume.
- **Text** — a frozen **Qwen** language model encodes a structured clinical
  impression string built from the patient's risk factors.
- **Clinical** — an MLP projects a 21-dim structured clinical feature vector.

The three feature streams are fused and passed to a linear head that outputs a
single **log-risk** logit per patient. The model is
`models.model.CORAMultimodalMACE` (shared backbone; not redefined here).

The 4-channel multi-window input (fat / soft tissue / angiographic /
calcification) is identical to the CORA pretraining input, so the pretrained
encoder receives the same intensity representation it was trained on.

## Distributed note

The MACE numbers reported in the manuscript were produced with multi-GPU
**DistributedDataParallel**. This release provides a clean **single-process**
(single-GPU) implementation for clarity and reproducibility. The Cox loss,
multimodal fusion, and evaluation logic are unchanged.

## Data layout

Per-patient preprocessed NPZ files under an `npz_root`:

```
<npz_root>/
  <patient_id>/
    CTA_<patient_id>.npz     # array: CTA_HU (D,H,W, in HU)
```

Optional per-patient risk-factor JSON files under a `risk_root`:

```
<risk_root>/
  <patient_id>/
    infor.json               # { "Demographic": { ...risk factors... } }
```

If `--risk_root` is omitted, the text stream falls back to a neutral placeholder
string and clinical features default to cohort-level fill values.

### Fold Excel files

5-fold cross-validation. Under `fold_root`, one subfolder per fold, each holding
a train / test split:

```
<fold_root>/
  fold1/  train.xlsx  test.xlsx
  fold2/  train.xlsx  test.xlsx
  ...
  fold5/  train.xlsx  test.xlsx
```

Each Excel file has one row per patient with (defaults; override in `dataset.py`):

| Column                          | Meaning                                      |
| ------------------------------- | -------------------------------------------- |
| `Deidentification Patient Name` | Patient identifier (matches NPZ folder)      |
| `MACE2`                         | Event indicator (1 = MACE observed, 0 = censored) |
| `days between CT and dx`        | Time-to-event / censoring, in days           |

## Structured clinical text

`dataset.riskdict_to_text` renders each risk-factor dict into a clinical
impression paragraph (demographics, vitals, labs, medical history, lifestyle)
that emphasizes cardiovascular risk factors. This string is the input to the
frozen Qwen text encoder.

## Clinical feature schema (21 dims)

Built by `clinical_features.riskdict_to_tensor`:

| Group        | Feature             | Encoding                          | Dims |
| ------------ | ------------------- | --------------------------------- | ---- |
| Categorical  | legal sex           | one-hot [m, f, n, unknown]        | 4    |
| Categorical  | HTN_within_a_yr     | one-hot [0, 1]                    | 2    |
| Categorical  | Diabetes status     | one-hot [0, 1]                    | 2    |
| Categorical  | tobacco use         | one-hot [never, former, yes, unknown] | 4 |
| Categorical  | Chest Pain          | one-hot [0, 1]                    | 2    |
| Continuous   | age                 | raw value                         | 1    |
| Continuous   | BP Systolic value   | raw value                         | 1    |
| Continuous   | BP Diastolic value  | raw value                         | 1    |
| Continuous   | LDL Cholesterol     | raw value                         | 1    |
| Continuous   | HDL Cholesterol     | raw value                         | 1    |
| Continuous   | Total Cholesterol   | raw value                         | 1    |
| Continuous   | BMI value           | raw value                         | 1    |
| **Total**    |                     |                                   | **21** |

Continuous values are passed through unchanged; the clinical MLP inside
`CORAMultimodalMACE` learns its own normalization. Missing labs are imputed with
cohort-level fill values (see `DEFAULT_FILL_VALUES`).

## Frozen Qwen note

The text encoder (`Qwen/Qwen2.5-7B` by default) is **frozen** — its parameters
are excluded from the optimizer and it runs in `eval` mode. Set `--text_encoder`
to a local cache path if the model is not pulled from the HuggingFace Hub.

## Hyperparameters

Read from `../../configs/cora_config.yaml` under `downstream.mace_prediction`
(`epochs`, `batch_size`, `learning_rate`, `weight_decay`, `n_folds`,
`text_encoder`, `freeze_text_encoder`). Loss is the **Cox negative partial
log-likelihood**; optimizer is AdamW with linear warmup + cosine decay. No
constants are hard-coded outside that config.

## Train

```bash
python train.py \
  --npz_root /path/to/npz \
  --fold_root data/folds_mace \
  --risk_root /path/to/risk_json \
  --pretrained checkpoints/cora_pretrained_best.pth \
  --text_encoder Qwen/Qwen2.5-7B \
  --output_dir checkpoints/mace_prediction
```

Runs 5-fold CV. For each fold, saves the best-by-C-index checkpoint
(`fold_<k>/mace_best.pth`), a latest checkpoint, and a per-epoch
`fold_<k>/training_log.csv`.

## Evaluate

```bash
python eval.py \
  --npz_root /path/to/npz \
  --fold_root data/folds_mace \
  --risk_root /path/to/risk_json \
  --checkpoint_dir checkpoints/mace_prediction \
  --output_csv results/mace_metrics.csv
```

Reports per-fold **C-index** and **AUROC** plus the aggregate mean +/- std,
written to a CSV table.

## Reproducibility

The MACE prediction numbers reported in the manuscript are produced by these
scripts (`train.py` + `eval.py`) using the hyperparameters in
`configs/cora_config.yaml`. The original training used DistributedDataParallel;
this single-GPU release preserves the Cox loss and fusion logic exactly.
