"""
Fine-tune the CORA encoder for volume-level plaque characterization.

Architecture: pretrained CORA encoder + global average/max pooling + linear head
(reused from models.model.CORAClassifier). The two-logit head (`logits2`) is used
for multi-label classification of:
    - calcified plaque present
    - non-calcified plaque present

Training:
    * Loss: BCEWithLogitsLoss over the 2 labels (multi-label).
    * Optimizer: AdamW.
    * Best-by-validation-AUC checkpoint selection.
    * Per-epoch train/val metrics logged to a CSV file.

All hyperparameters are read from configs/cora_config.yaml
(downstream.plaque_characterization) so that the paper, README, and code agree.
"""

import os
import sys
import csv
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset import PlaqueCharacterizationDataset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.model import CORAClassifier


LABEL_NAMES = ["calcified", "non_calcified"]


# =============================================================================
# Utilities
# =============================================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def multilabel_auc(targets: np.ndarray, probs: np.ndarray):
    """Per-label AUROC (NaN where a label is single-class) plus the mean."""
    per_label = []
    for j in range(targets.shape[1]):
        y = targets[:, j]
        if len(np.unique(y)) < 2:
            per_label.append(float("nan"))
        else:
            per_label.append(roc_auc_score(y, probs[:, j]))
    valid = [a for a in per_label if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float("nan")
    return per_label, mean_auc


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return mean BCE loss, per-label AUROC, and mean AUROC on a split."""
    model.eval()
    total, n = 0.0, 0
    all_targets, all_probs = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _, logits2 = model(images)
        loss = criterion(logits2, targets)
        total += loss.item()
        n += 1
        all_targets.append(targets.cpu().numpy())
        all_probs.append(torch.sigmoid(logits2).cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    per_label, mean_auc = multilabel_auc(targets, probs)
    return total / max(n, 1), per_label, mean_auc


# =============================================================================
# Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/cora_config.yaml")
    parser.add_argument("--npz_root", default="data/npz",
                        help="Root dir with per-patient NPZ files.")
    parser.add_argument("--train_index", default="data/plaque_train.xlsx",
                        help="Excel index of training patients + multi-label columns.")
    parser.add_argument("--val_index", default=None,
                        help="Excel index of validation patients. If omitted, a "
                             "held-out split is carved from --train_index.")
    parser.add_argument("--pretrained", default="checkpoints/cora_pretrained_best.pth",
                        help="Path to the CORA-pretrained encoder weights.")
    parser.add_argument("--output_dir", default="checkpoints/plaque_characterization")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[128, 128, 128],
                        help="Fixed (D H W) volume shape for batching.")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Held-out fraction when --val_index is not given.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["downstream"]["plaque_characterization"]
    EPOCHS = dc["epochs"]
    BATCH_SIZE = dc["batch_size"]
    LEARNING_RATE = float(dc["learning_rate"])
    WEIGHT_DECAY = float(dc["weight_decay"])
    NUM_INPUT_CHANNELS = cfg["model"]["num_input_channels"]
    TARGET_SHAPE = tuple(args.target_shape)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.csv")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_ds = PlaqueCharacterizationDataset(
        excel_file=args.train_index,
        npz_root=args.npz_root,
        target_shape=TARGET_SHAPE,
    )
    if args.val_index is not None:
        val_ds = PlaqueCharacterizationDataset(
            excel_file=args.val_index,
            npz_root=args.npz_root,
            target_shape=TARGET_SHAPE,
        )
        train_set, val_set = train_ds, val_ds
    else:
        n_total = len(train_ds)
        n_val = max(1, int(round(n_total * args.val_fraction)))
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_total)
        val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
        train_set = Subset(train_ds, train_idx)
        val_set = Subset(train_ds, val_idx)
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Model / loss / optimizer ---
    pretrained = args.pretrained if args.pretrained and os.path.exists(args.pretrained) else None
    if pretrained is None and args.pretrained:
        print(f"[Warning] Pretrained weights not found at '{args.pretrained}'. "
              f"Training the encoder from scratch.")
    model = CORAClassifier(
        num_input_channels=NUM_INPUT_CHANNELS,
        pretrained_path=pretrained,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss",
                 "val_auc_calcified", "val_auc_non_calcified", "val_auc_mean"]
            )

    # --- Train ---
    best_auc = -float("inf")
    print(f"Starting fine-tuning for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, logits2 = model(images)
            loss = criterion(logits2, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, per_label_auc, mean_auc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"AUC calc {per_label_auc[0]:.4f} / non-calc {per_label_auc[1]:.4f} | "
              f"mean {mean_auc:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{per_label_auc[0]:.6f}", f"{per_label_auc[1]:.6f}", f"{mean_auc:.6f}",
            ])

        # Latest checkpoint (for resuming).
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(), "val_auc": mean_auc},
            os.path.join(args.output_dir, "checkpoint_latest.pth"),
        )

        # Best-by-val-AUC checkpoint.
        if not np.isnan(mean_auc) and mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "plaque_best.pth"))
            print(f"  -> new best val mean AUC {best_auc:.4f} (checkpoint saved)")

    print(f"Fine-tuning complete. Best val mean AUC: {best_auc:.4f}. "
          f"Best weights: {os.path.join(args.output_dir, 'plaque_best.pth')}")


if __name__ == "__main__":
    main()
