"""
Fine-tune CORA for coronary artery segmentation (ImageCAS).

A pretrained CORA encoder + a randomly initialized decoder (full U-Net,
`CORASegmentationModel`) are fine-tuned end-to-end with a Dice loss to densely
segment the coronary artery tree. Validation uses MONAI sliding-window
inference over the full volume; the best checkpoint is selected by mean Dice.

All hyperparameters are read from configs/cora_config.yaml
(downstream.coronary_segmentation) so the paper, README, and code cannot drift apart.
"""

import os
import csv
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from dataset import get_train_loader, get_eval_loader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.model import CORASegmentationModel


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


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Binary Dice between two boolean/int volumes."""
    pred_b = pred > 0
    gt_b = gt > 0
    inter = float(np.sum(pred_b & gt_b))
    denom = float(pred_b.sum() + gt_b.sum())
    return (2.0 * inter + eps) / (denom + eps)


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(model, loader, device, patch_size, label_key, sw_batch_size):
    """Run sliding-window inference and return the mean foreground Dice."""
    model.eval()
    dices = []
    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        labels = batch[label_key]
        logits = sliding_window_inference(
            inputs=images, roi_size=patch_size, sw_batch_size=sw_batch_size,
            predictor=model, overlap=0.5,
        )
        if logits.shape != labels.shape:
            continue
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0]
        gt = labels.cpu().numpy()[0, 0]
        dices.append(dice_score(pred, gt))
    return float(np.mean(dices)) if dices else -1.0


# =============================================================================
# Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CORA for coronary artery segmentation.")
    parser.add_argument("--config", default="../../configs/cora_config.yaml")
    parser.add_argument("--data_root", default="data/npz",
                        help="Root for NPZ paths listed in the index files.")
    parser.add_argument("--train_index", default="data/train_index.txt",
                        help="Directory of NPZ files, or text file listing one NPZ per line.")
    parser.add_argument("--val_index", default="data/val_index.txt",
                        help="Directory of NPZ files, or text file listing one NPZ per line.")
    parser.add_argument("--pretrained", default="checkpoints/cora_pretrained_best.pth",
                        help="CORA-pretrained encoder weights (placeholder default).")
    parser.add_argument("--output_dir", default="checkpoints/coronary")
    parser.add_argument("--label_key", default="label",
                        help="NPZ key holding the binary coronary artery mask.")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sc = cfg["downstream"]["coronary_segmentation"]
    patch_size = tuple(sc["patch_shape"])
    epochs = sc["epochs"]
    batch_size = sc["batch_size"]
    lr = float(sc["learning_rate"])
    weight_decay = float(sc.get("weight_decay", 1e-5))
    num_channels = cfg["model"]["num_input_channels"]

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.csv")

    # --- Data ---
    train_loader = get_train_loader(
        npz_root=args.data_root, index_file=args.train_index, label_key=args.label_key,
        patch_size=patch_size, batch_size=batch_size,
    )
    val_loader = get_eval_loader(
        npz_root=args.data_root, index_file=args.val_index, label_key=args.label_key,
    )

    # --- Model / loss / optimizer ---
    pretrained = args.pretrained if os.path.exists(args.pretrained) else None
    if pretrained is None:
        print(f"[Warning] pretrained weights '{args.pretrained}' not found; "
              f"training the encoder from scratch.")
    model = CORASegmentationModel(
        num_input_channels=num_channels, num_classes=1, pretrained_path=pretrained,
    ).to(device)

    # Dice loss on the single foreground channel (sigmoid), per the manuscript.
    criterion = DiceLoss(sigmoid=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = min(5, max(1, epochs // 20))
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    scaler = GradScaler()

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "lr"])

    # --- Training ---
    best_dice = -1.0
    best_epoch = -1
    print(f"Starting fine-tuning for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch[args.label_key].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = running_loss / max(steps, 1)
        cur_lr = scheduler.get_last_lr()[0]

        val_dice = -1.0
        if (epoch + 1) % args.val_interval == 0:
            val_dice = validate(
                model, val_loader, device, patch_size, args.label_key,
                sw_batch_size=batch_size,
            )
            print(f"Epoch {epoch + 1} | loss {train_loss:.4f} | "
                  f"Dice {val_dice:.4f} | LR {cur_lr:.2e}")
            torch.cuda.empty_cache()

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, f"{train_loss:.6f}", f"{val_dice:.6f}", f"{cur_lr:.3e}"]
            )

        # Best-by-Dice checkpoint.
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "dice": best_dice},
                os.path.join(args.output_dir, "coronary_best.pth"),
            )
            print(f"  ↳ new best Dice {best_dice:.4f} (checkpoint saved)")

        # Latest checkpoint (for resuming).
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scaler_state_dict": scaler.state_dict()},
            os.path.join(args.output_dir, "coronary_latest.pth"),
        )

    print(f"Training complete. Best Dice: {best_dice:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
