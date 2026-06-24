"""
Synthesis-driven self-supervised pretraining on unlabeled CCTA volumes.

The model learns to segment synthetically inserted coronary lesions, biasing
representations toward clinically relevant vascular pathology.

Reproducibility additions (addressing reviewer R3):
    * Held-out validation split of the pretraining data.
    * Per-epoch evaluation of the self-supervised objective on the val split.
    * Early stopping + best-checkpoint selection by validation loss.
    * Per-epoch train/val loss logged to a CSV file.

All hyperparameters are read from configs/cora_config.yaml so that the paper,
README, and code cannot drift apart.
"""

import os
import csv
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from dataset import CORAPretrainingDataset
from losses import LesionSegmentationLoss
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.model import CORAPretrainModel


# =============================================================================
# Utilities
# =============================================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Compute the mean self-supervised (segmentation) loss on a held-out split."""
    model.eval()
    total, n = 0.0, 0
    for images, _labels, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(enabled=True):
            _, logits_seg = model(images)
            loss = criterion(logits_seg, masks)
        total += loss.item()
        n += 1
    return total / max(n, 1)


# =============================================================================
# Pretraining Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/cora_config.yaml")
    parser.add_argument("--excel_file", default="data/CTA_all_list.xlsx",
                        help="Patient index (pretraining patients ONLY; test patients excluded).")
    parser.add_argument("--npz_root", default="/path/to/preprocessed/npz")
    parser.add_argument("--checkpoint_dir", default="checkpoints/cora_pretrain")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pc = cfg["pretrain"]

    EPOCHS = pc["epochs"]
    BATCH_SIZE = pc["batch_size"]
    LEARNING_RATE = float(pc["learning_rate"])
    WEIGHT_DECAY = float(pc["weight_decay"])
    WARMUP_EPOCHS = pc["warmup_epochs"]
    PATCH_SHAPE = tuple(pc["patch_shape"])
    VAL_FRACTION = pc["val_fraction"]
    PATIENCE = pc["early_stopping_patience"]
    SEED = pc["seed"]
    NUM_WORKERS = 8

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, "training_log.csv")

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data: train / val split of the pretraining cohort ---
    full_dataset = CORAPretrainingDataset(
        excel_file=args.excel_file,
        npz_root=args.npz_root,
        patch_shape=PATCH_SHAPE,
        min_mask_voxels=pc["min_mask_voxels"],
        noise_params=pc["noise"],
        lesion_params=pc["lesion"],
    )
    n_total = len(full_dataset)
    n_val = max(1, int(round(n_total * VAL_FRACTION)))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_total)
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    train_set, val_set = Subset(full_dataset, train_idx), Subset(full_dataset, val_idx)
    print(f"Pretraining cohort: {n_total} | train: {len(train_set)} | val: {len(val_set)}")

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, generator=g,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, worker_init_fn=worker_init_fn,
    )

    # --- Model / loss / optimizer ---
    model = CORAPretrainModel(num_input_channels=cfg["model"]["num_input_channels"]).to(device)
    criterion = LesionSegmentationLoss(
        tversky_alpha=pc["loss"]["tversky_alpha"],
        tversky_beta=pc["loss"]["tversky_beta"],
        focal_gamma=pc["loss"]["focal_gamma"],
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler = GradScaler()

    # --- Logging file ---
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    # --- Training with validation + early stopping ---
    best_val = float("inf")
    epochs_without_improve = 0
    print(f"Starting pretraining for up to {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for images, _labels, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=True):
                _, logits_seg = model(images)
                loss = criterion(logits_seg, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1} | train {train_loss:.4f} | val {val_loss:.4f} | LR {lr:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.3e}"])

        # Always keep the latest checkpoint (for resuming).
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scaler_state_dict": scaler.state_dict(),
             "val_loss": val_loss},
            os.path.join(args.checkpoint_dir, "checkpoint_latest.pth"),
        )

        # Best-by-val-loss checkpoint + early stopping.
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, "cora_pretrained_best.pth"))
            print(f"  ↳ new best val loss {best_val:.4f} (checkpoint saved)")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(no val improvement for {PATIENCE} epochs).")
                break

    final_path = os.path.join(args.checkpoint_dir, "cora_pretrained_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Pretraining complete. Best val loss: {best_val:.4f}. "
          f"Best weights: cora_pretrained_best.pth | final: {final_path}")


if __name__ == "__main__":
    main()
