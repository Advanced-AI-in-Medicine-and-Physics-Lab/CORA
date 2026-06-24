"""
Evaluate fine-tuned CORAMultimodalMACE models across the 5 CV folds.

For each fold this script loads the best checkpoint, runs multimodal inference
on that fold's held-out (test) split, and reports:
    * C-index  - Harrell's concordance index (survival discrimination).
    * AUROC    - using the MACE event indicator as the binary outcome.

Per-fold metrics and the aggregate mean +/- std are written to a CSV table.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset import MACEDataset, collate_fn
from clinical_features import batch_riskdict_to_tensor, NUM_CLINICAL_FEATURES
from metrics import concordance_index

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.model import CORAMultimodalMACE


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict(model, loader, device):
    """Run inference; return (scores, times, events) as 1-d arrays."""
    model.eval()
    all_scores, all_times, all_events = [], [], []
    for batch in tqdm(loader, desc="Inference", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        clinical = batch_riskdict_to_tensor(batch["risk_dict"]).to(device)
        logits = model(images, batch["text"], clinical)
        all_scores.append(logits.view(-1).cpu().numpy())
        all_times.append(batch["time"].numpy())
        all_events.append(batch["event"].numpy())
    return (
        np.concatenate(all_scores),
        np.concatenate(all_times),
        np.concatenate(all_events),
    )


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/cora_config.yaml")
    parser.add_argument("--npz_root", default="data/npz")
    parser.add_argument("--fold_root", default="data/folds_mace",
                        help="Dir with per-fold subfolders fold<k>/test.xlsx.")
    parser.add_argument("--risk_root", default=None,
                        help="Root dir with per-patient risk-factor JSON files. Optional.")
    parser.add_argument("--checkpoint_dir", default="checkpoints/mace_prediction",
                        help="Dir with per-fold subfolders fold_<k>/mace_best.pth.")
    parser.add_argument("--text_encoder", default=None,
                        help="Path / HF id of the frozen Qwen text encoder. "
                             "Defaults to config downstream.mace_prediction.text_encoder.")
    parser.add_argument("--output_csv", default="results/mace_metrics.csv")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["downstream"]["mace_prediction"]
    n_folds = dc["n_folds"]
    num_input_channels = cfg["model"]["num_input_channels"]
    if args.text_encoder is None:
        args.text_encoder = dc["text_encoder"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | text encoder (frozen): {args.text_encoder}")

    rows = []
    for fold_id in range(1, n_folds + 1):
        ckpt = os.path.join(args.checkpoint_dir, f"fold_{fold_id}", "mace_best.pth")
        test_xlsx = os.path.join(args.fold_root, f"fold{fold_id}", "test.xlsx")
        if not os.path.exists(ckpt):
            print(f"[Warning] Missing checkpoint for fold {fold_id}: {ckpt}. Skipping.")
            continue

        test_ds = MACEDataset(
            excel_file=test_xlsx, npz_root=args.npz_root, risk_root=args.risk_root,
            target_shape=tuple(args.target_shape),
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        )

        model = CORAMultimodalMACE(
            num_input_channels=num_input_channels,
            qwen_model_path=args.text_encoder,
            num_clinical_features=NUM_CLINICAL_FEATURES,
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)

        scores, times, events = predict(model, test_loader, device)
        c_index = concordance_index(times, scores, events)
        try:
            auroc = roc_auc_score(events, scores)
        except ValueError:
            auroc = float("nan")

        print(f"Fold {fold_id} | n={len(events)} | C-index {c_index:.4f} | AUROC {auroc:.4f}")
        rows.append({
            "fold": fold_id,
            "n": int(len(events)),
            "events": int(events.sum()),
            "c_index": c_index,
            "auroc": auroc,
        })

    if not rows:
        print("No folds evaluated; nothing to write.")
        return

    df = pd.DataFrame(rows)
    summary = {
        "fold": "mean_std",
        "n": int(df["n"].sum()),
        "events": int(df["events"].sum()),
        "c_index": f"{df['c_index'].mean():.4f} +/- {df['c_index'].std(ddof=0):.4f}",
        "auroc": f"{np.nanmean(df['auroc']):.4f} +/- {np.nanstd(df['auroc']):.4f}",
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("\nPer-fold metrics:")
    print(df.to_string(index=False))
    print(f"\nResults written to {args.output_csv}")


if __name__ == "__main__":
    main()
