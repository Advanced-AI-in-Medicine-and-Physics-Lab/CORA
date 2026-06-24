"""
Evaluate a fine-tuned plaque-characterization model on a held-out test cohort.

Loads a CORAClassifier checkpoint, runs whole-volume inference, and reports
per-label and macro-averaged metrics (AUROC, AUPRC, F1, accuracy) for the two
multi-label targets (calcified / non-calcified plaque present). Results are
written to a CSV table.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from tqdm import tqdm

from dataset import PlaqueCharacterizationDataset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.model import CORAClassifier


LABEL_NAMES = ["calcified", "non_calcified"]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict(model, loader, device):
    """Run inference and return (targets, probabilities) as (N, 2) arrays."""
    model.eval()
    all_targets, all_probs = [], []
    for images, targets in tqdm(loader, desc="Inference", leave=False):
        images = images.to(device, non_blocking=True)
        _, logits2 = model(images)
        all_targets.append(targets.numpy())
        all_probs.append(torch.sigmoid(logits2).cpu().numpy())
    return np.concatenate(all_targets, axis=0), np.concatenate(all_probs, axis=0)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    """Per-label AUROC, AUPRC, F1, accuracy as a list of dict rows."""
    preds = (probs >= threshold).astype(int)
    rows = []
    for j, name in enumerate(LABEL_NAMES):
        y, p, yhat = targets[:, j], probs[:, j], preds[:, j]
        single_class = len(np.unique(y)) < 2
        rows.append({
            "label": name,
            "positives": int(y.sum()),
            "n": int(len(y)),
            "auroc": float("nan") if single_class else roc_auc_score(y, p),
            "auprc": float("nan") if single_class else average_precision_score(y, p),
            "f1": f1_score(y, yhat, zero_division=0),
            "accuracy": accuracy_score(y, yhat),
        })
    return rows


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/cora_config.yaml")
    parser.add_argument("--npz_root", default="data/npz")
    parser.add_argument("--test_index", default="data/plaque_test.xlsx",
                        help="Excel index of test patients + multi-label columns.")
    parser.add_argument("--checkpoint", default="checkpoints/plaque_characterization/plaque_best.pth",
                        help="Fine-tuned CORAClassifier weights.")
    parser.add_argument("--output_csv", default="results/plaque_metrics.csv")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_input_channels = cfg["model"]["num_input_channels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    test_ds = PlaqueCharacterizationDataset(
        excel_file=args.test_index,
        npz_root=args.npz_root,
        target_shape=tuple(args.target_shape),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Test volumes: {len(test_ds)}")

    # --- Model ---
    model = CORAClassifier(num_input_channels=num_input_channels).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Inference + metrics ---
    targets, probs = predict(model, test_loader, device)
    rows = compute_metrics(targets, probs, threshold=args.threshold)

    # Macro average across labels.
    df = pd.DataFrame(rows)
    macro = {
        "label": "macro_avg",
        "positives": int(df["positives"].sum()),
        "n": int(df["n"].iloc[0]) if len(df) else 0,
        "auroc": float(np.nanmean(df["auroc"])),
        "auprc": float(np.nanmean(df["auprc"])),
        "f1": float(df["f1"].mean()),
        "accuracy": float(df["accuracy"].mean()),
    }
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("\nPer-label metrics:")
    print(df.to_string(index=False))
    print(f"\nResults written to {args.output_csv}")


if __name__ == "__main__":
    main()
