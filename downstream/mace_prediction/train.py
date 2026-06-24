"""
Fine-tune CORAMultimodalMACE for multimodal MACE risk stratification.

Architecture (models.model.CORAMultimodalMACE):
    pretrained CORA image encoder
    + frozen Qwen text encoder (clinical impression text)
    + MLP projection of the 21-dim structured clinical feature vector
    + fusion + linear head -> a single log-risk logit per patient.

Training:
    * Loss: Cox negative partial log-likelihood (survival).
    * Optimizer: AdamW with linear warmup + cosine decay.
    * 5-fold cross-validation; best-by-C-index checkpoint per fold.
    * Per-epoch metrics logged to a CSV file.

Distributed note:
    The manuscript results were produced with multi-GPU DistributedDataParallel.
    This release provides a clean SINGLE-process implementation for clarity and
    reproducibility; the loss, fusion, and evaluation logic are unchanged.

Hyperparameters are read from configs/cora_config.yaml
(downstream.mace_prediction) so the paper, README, and code agree.
"""

import os
import sys
import csv
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from dataset import MACEDataset, collate_fn
from clinical_features import batch_riskdict_to_tensor, NUM_CLINICAL_FEATURES
from metrics import concordance_index

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.model import CORAMultimodalMACE


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


# =============================================================================
# Cox Survival Loss
# =============================================================================

def cox_neg_partial_log_likelihood(
    risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor
) -> torch.Tensor:
    """
    Cox negative partial log-likelihood (Breslow approximation).

    Args:
        risk_scores: Log-risk logits [B] or [B, 1] (higher = higher risk).
        times: Event / censoring times [B].
        events: Event indicators [B] (1 = event observed, 0 = censored).

    Returns:
        Scalar loss, normalized by the number of observed events.
    """
    risk = risk_scores.view(-1)
    times = times.view(-1)
    events = events.view(-1).float()

    # Sort by descending time so the cumulative sum forms each patient's risk set.
    order = torch.argsort(times, descending=True)
    risk_ordered = risk[order]
    events_ordered = events[order]

    exp_risk = torch.exp(risk_ordered)
    log_denom = torch.log(torch.cumsum(exp_risk, dim=0) + 1e-8)
    loss_terms = (risk_ordered - log_denom) * events_ordered
    neg_partial_ll = -torch.sum(loss_terms)

    num_events = torch.clamp(torch.sum(events_ordered), min=1.0)
    return neg_partial_ll / num_events


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Compute the Cox loss and concordance index (C-index) on a split."""
    model.eval()
    all_scores, all_times, all_events = [], [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        times = batch["time"].to(device, non_blocking=True).float()
        events = batch["event"].to(device, non_blocking=True).float()
        clinical = batch_riskdict_to_tensor(batch["risk_dict"]).to(device)

        logits = model(images, batch["text"], clinical)
        loss = cox_neg_partial_log_likelihood(logits, times, events)
        total_loss += loss.item()
        n += 1

        all_scores.append(logits.detach().view(-1).cpu().numpy())
        all_times.append(times.detach().cpu().numpy())
        all_events.append(events.detach().cpu().numpy())

    scores = np.concatenate(all_scores)
    times = np.concatenate(all_times)
    events = np.concatenate(all_events)
    return {
        "loss": total_loss / max(n, 1),
        "c_index": concordance_index(times, scores, events),
    }


# =============================================================================
# Single-Fold Training
# =============================================================================

def train_one_fold(fold_id, train_set, val_set, cfg, args, device) -> float:
    """Train one CV fold; return the best validation C-index."""
    dc = cfg["downstream"]["mace_prediction"]
    EPOCHS = dc["epochs"]
    BATCH_SIZE = dc["batch_size"]
    LEARNING_RATE = float(dc["learning_rate"])
    WEIGHT_DECAY = float(dc["weight_decay"])
    WARMUP_EPOCHS = dc.get("warmup_epochs", 3)
    NUM_INPUT_CHANNELS = cfg["model"]["num_input_channels"]

    fold_dir = os.path.join(args.output_dir, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    log_path = os.path.join(fold_dir, "training_log.csv")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn,
    )

    pretrained = args.pretrained if args.pretrained and os.path.exists(args.pretrained) else None
    if pretrained is None and args.pretrained:
        print(f"[Warning] Pretrained weights not found at '{args.pretrained}'. "
              f"Training the image encoder from scratch.")

    model = CORAMultimodalMACE(
        num_input_channels=NUM_INPUT_CHANNELS,
        qwen_model_path=args.text_encoder,
        num_clinical_features=NUM_CLINICAL_FEATURES,
        pretrained_path=pretrained,
    ).to(device)

    # Optimize only trainable params (the Qwen text encoder is frozen).
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_c_index", "lr"])

    best_c = -float("inf")
    print(f"\n{'=' * 60}\nFold {fold_id} | train {len(train_set)} | val {len(val_set)}\n{'=' * 60}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fold {fold_id} Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            times = batch["time"].to(device, non_blocking=True).float()
            events = batch["event"].to(device, non_blocking=True).float()
            clinical = batch_riskdict_to_tensor(batch["risk_dict"]).to(device)

            optimizer.zero_grad()
            logits = model(images, batch["text"], clinical)
            loss = cox_neg_partial_log_likelihood(logits, times, events)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        val = evaluate(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]
        print(f"Fold {fold_id} Epoch {epoch + 1} | train {train_loss:.4f} | "
              f"val {val['loss']:.4f} | C-index {val['c_index']:.4f} | LR {lr:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.6f}", f"{val['loss']:.6f}",
                f"{val['c_index']:.6f}", f"{lr:.3e}",
            ])

        # Latest checkpoint (for resuming).
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(), "val_c_index": val["c_index"]},
            os.path.join(fold_dir, "checkpoint_latest.pth"),
        )

        # Best-by-C-index checkpoint.
        if val["c_index"] > best_c:
            best_c = val["c_index"]
            torch.save(model.state_dict(), os.path.join(fold_dir, "mace_best.pth"))
            print(f"  -> new best val C-index {best_c:.4f} (checkpoint saved)")

    print(f"Fold {fold_id} complete. Best val C-index: {best_c:.4f}")
    return best_c


# =============================================================================
# Cross-Validation Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/cora_config.yaml")
    parser.add_argument("--npz_root", default="data/npz",
                        help="Root dir with per-patient NPZ files.")
    parser.add_argument("--fold_root", default="data/folds_mace",
                        help="Dir with per-fold subfolders fold<k>/{train,test}.xlsx.")
    parser.add_argument("--risk_root", default=None,
                        help="Root dir with per-patient risk-factor JSON files "
                             "(<risk_root>/<name>/infor.json). Optional.")
    parser.add_argument("--pretrained", default="checkpoints/cora_pretrained_best.pth",
                        help="Path to the CORA-pretrained image-encoder weights.")
    parser.add_argument("--text_encoder", default=None,
                        help="Path / HF id of the (frozen) Qwen text encoder. "
                             "Defaults to config downstream.mace_prediction.text_encoder.")
    parser.add_argument("--output_dir", default="checkpoints/mace_prediction")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[128, 128, 128],
                        help="Fixed (D H W) volume shape for batching.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["downstream"]["mace_prediction"]
    n_folds = dc["n_folds"]
    if args.text_encoder is None:
        args.text_encoder = dc["text_encoder"]

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | text encoder (frozen): {args.text_encoder}")

    fold_scores = []
    for fold_id in range(1, n_folds + 1):
        fold_dir = os.path.join(args.fold_root, f"fold{fold_id}")
        train_set = MACEDataset(
            excel_file=os.path.join(fold_dir, "train.xlsx"),
            npz_root=args.npz_root, risk_root=args.risk_root,
            target_shape=tuple(args.target_shape),
        )
        val_set = MACEDataset(
            excel_file=os.path.join(fold_dir, "test.xlsx"),
            npz_root=args.npz_root, risk_root=args.risk_root,
            target_shape=tuple(args.target_shape),
        )
        best_c = train_one_fold(fold_id, train_set, val_set, cfg, args, device)
        fold_scores.append(best_c)

    print(f"\n{'=' * 60}\nCross-validation summary ({n_folds} folds)\n{'=' * 60}")
    for k, c in enumerate(fold_scores, 1):
        print(f"  Fold {k}: best C-index {c:.4f}")
    print(f"  Mean C-index: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")


if __name__ == "__main__":
    main()
