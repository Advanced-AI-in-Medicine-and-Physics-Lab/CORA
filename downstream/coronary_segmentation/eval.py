"""
Dense evaluation for coronary artery segmentation (ImageCAS).

Sliding-window inference produces a voxel-level coronary probability map per
volume, which is binarized and scored against the ground-truth artery mask with
three complementary metrics (manuscript):

    * Dice  - volumetric overlap.
    * clDice - centerline Dice; topology-aware overlap computed on the 3D
      skeletons of prediction and ground truth (Shit et al., CVPR 2021).
    * MSD   - mean surface distance, via MONAI SurfaceDistanceMetric
      (symmetric), faithful to the CORA-v2 inference script.

Per-case and summary results are written to CSV; NIfTI predictions are saved
optionally with `--save_nii`.
"""

import os
import csv
import argparse

import numpy as np
import torch
import yaml
from scipy.special import expit
from skimage.morphology import skeletonize
from monai.metrics import SurfaceDistanceMetric
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from dataset import get_eval_loader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.model import CORASegmentationModel


# =============================================================================
# Dense Segmentation Metrics
# =============================================================================

def compute_cldice(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Centerline Dice (clDice) and its components.

    Topology sensitivity (clRecall) is the fraction of the GT skeleton lying
    inside the prediction; topology precision (clPrecision) is the fraction of
    the predicted skeleton lying inside the GT. clDice is their harmonic mean.
    """
    pred_b = pred > 0
    gt_b = gt > 0

    skel_gt = skeletonize(gt_b)
    skel_pred = skeletonize(pred_b)

    len_skel_gt = np.sum(skel_gt)
    cl_recall = (np.sum(skel_gt & pred_b) / len_skel_gt) if len_skel_gt > 0 else 1.0

    len_skel_pred = np.sum(skel_pred)
    cl_precision = (np.sum(skel_pred & gt_b) / len_skel_pred) if len_skel_pred > 0 else 1.0

    denom = cl_recall + cl_precision
    cl_dice = (2 * cl_recall * cl_precision / denom) if denom > 0 else 0.0
    return {"clRecall": float(cl_recall), "clPrecision": float(cl_precision),
            "clDice": float(cl_dice)}


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute Dice, clDice (and components), and MSD for one case.

    `pred` and `gt` are binary integer volumes (D, H, W). MSD uses MONAI's
    symmetric SurfaceDistanceMetric; if it cannot be computed (e.g. an empty
    mask) MSD is reported as NaN.
    """
    pred_b = pred > 0
    gt_b = gt > 0

    tp = float(np.sum(pred_b & gt_b))
    denom = float(pred_b.sum() + gt_b.sum())
    eps = 1e-6
    dice = (2.0 * tp + eps) / (denom + eps)

    # MSD via MONAI SurfaceDistanceMetric: inputs as (B, C, D, H, W) binary.
    msd = float("nan")
    try:
        p_tensor = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        g_tensor = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        surface_metric = SurfaceDistanceMetric(
            include_background=True, symmetric=True, reduction="mean"
        )
        msd = float(surface_metric(y_pred=p_tensor, y=g_tensor).item())
    except Exception:
        msd = float("nan")

    metrics = {"Dice": float(dice), "MSD": msd}
    metrics.update(compute_cldice(pred, gt))
    return metrics


# =============================================================================
# NIfTI Saving
# =============================================================================

def save_prediction(gt: np.ndarray, pred: np.ndarray, save_dir: str, name: str):
    """Save the ground-truth and predicted masks as `.nii.gz` for inspection."""
    import SimpleITK as sitk

    os.makedirs(save_dir, exist_ok=True)
    gt_itk = sitk.GetImageFromArray(gt.astype(np.uint8))
    sitk.WriteImage(gt_itk, os.path.join(save_dir, f"{name}_gt.nii.gz"))
    pred_itk = sitk.GetImageFromArray(pred.astype(np.uint8))
    sitk.WriteImage(pred_itk, os.path.join(save_dir, f"{name}_pred.nii.gz"))


# =============================================================================
# Inference
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(args):
    cfg = load_config(args.config)
    sc = cfg["downstream"]["coronary_segmentation"]
    patch_size = tuple(sc["patch_shape"])
    num_channels = cfg["model"]["num_input_channels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CORASegmentationModel(num_input_channels=num_channels, num_classes=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    loader = get_eval_loader(args.data_root, args.val_index, label_key=args.label_key)

    os.makedirs(args.output_dir, exist_ok=True)
    per_case_path = os.path.join(args.output_dir, "per_case_results.csv")
    metric_cols = ["Dice", "clDice", "MSD", "clRecall", "clPrecision"]

    rows = []
    with open(per_case_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"] + metric_cols)

        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch[args.label_key]
            name = batch["name"][0]

            logits = sliding_window_inference(
                inputs=images, roi_size=patch_size, sw_batch_size=args.sw_batch_size,
                predictor=model, overlap=0.5,
            )
            if logits.shape != labels.shape:
                print(f"[Warning] shape mismatch {logits.shape} vs {labels.shape}; skipping.")
                continue

            pred = (expit(logits.cpu().numpy()[0, 0]) > 0.5).astype(np.uint8)
            gt = (labels.cpu().numpy()[0, 0] > 0).astype(np.uint8)

            m = compute_metrics(pred, gt)
            rows.append(m)
            writer.writerow([name] + [f"{m[c]:.4f}" for c in metric_cols])

            if args.save_nii:
                save_prediction(gt, pred, os.path.join(args.output_dir, "predictions"), name)

    # Aggregate (mean / std over cases).
    summary_path = os.path.join(args.output_dir, "summary_results.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["statistic"] + metric_cols)
        means = [float(np.nanmean([r[c] for r in rows])) for c in metric_cols]
        stds = [float(np.nanstd([r[c] for r in rows])) for c in metric_cols]
        writer.writerow(["mean"] + [f"{v:.4f}" for v in means])
        writer.writerow(["std"] + [f"{v:.4f}" for v in stds])

    print("\n=== Coronary Artery Segmentation (ImageCAS) ===")
    print(f"Cases evaluated: {len(rows)}")
    for col, mean, std in zip(metric_cols, means, stds):
        print(f"{col:<12}: {mean:.4f} +/- {std:.4f}")
    print(f"\nResults saved to: {summary_path} and {per_case_path}")


def main():
    parser = argparse.ArgumentParser(description="Dense coronary artery segmentation evaluation.")
    parser.add_argument("--config", default="../../configs/cora_config.yaml")
    parser.add_argument("--data_root", default="data/npz",
                        help="Root for NPZ paths listed in --val_index.")
    parser.add_argument("--val_index", default="data/val_index.txt",
                        help="Directory of NPZ files, or text file listing one NPZ per line.")
    parser.add_argument("--checkpoint", default="checkpoints/coronary/coronary_best.pth")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--label_key", default="label",
                        help="NPZ key holding the binary coronary artery mask.")
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--save_nii", action="store_true",
                        help="Save GT and predicted masks as NIfTI for inspection.")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
