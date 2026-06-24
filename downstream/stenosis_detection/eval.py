"""
Lesion-level evaluation for stenosis detection (segmentation formulation).

Sliding-window inference produces a voxel-level lesion probability map per
volume. Lesions are then matched at the connected-component level:

    * Ground-truth and predicted lesions are extracted as 3D connected
      components (26-connectivity).
    * A predicted lesion overlapping a ground-truth lesion by MORE THAN
      `overlap_threshold_voxels` (manuscript: 10) voxels counts as a
      true positive (TP). A predicted lesion that matches no GT lesion is a
      false positive (FP). A GT lesion matched by no prediction is a false
      negative (FN).
    * Reports lesion-level sensitivity, precision, F1; plus voxel-level Dice
      and specificity.

Faithful to the CORA-v2 LesionDetectionMetric; overlap is measured against a
dilated GT mask (dilation tolerance) so that near-misses at lesion boundaries
are credited, as in the original implementation.
"""

import os
import csv
import argparse

import numpy as np
import torch
import yaml
from scipy import ndimage
from scipy.special import expit
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from dataset import get_eval_loader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.model import CORASegmentationModel


# =============================================================================
# Lesion-Level Connected-Component Metric
# =============================================================================

class LesionDetectionMetric:
    """
    Connected-component (lesion-level) TP/FP/FN matching with voxel-level
    sensitivity / specificity / Dice accumulation.

    Args:
        overlap_threshold: Min overlapping voxels for a predicted lesion to
            count as matching a ground-truth lesion (manuscript: >10).
        dilation_iter: GT dilation tolerance (voxels) applied before overlap
            testing, accounting for boundary annotation uncertainty.
        prob_threshold: Probability threshold to binarize predictions.
    """

    def __init__(self, overlap_threshold: int = 10, dilation_iter: int = 1,
                 prob_threshold: float = 0.5):
        self.overlap_threshold = overlap_threshold
        self.dilation_iter = dilation_iter
        self.prob_threshold = prob_threshold
        self.structure = ndimage.generate_binary_structure(3, 3)
        self.reset()

    def reset(self):
        self.tp = 0          # predicted lesions matching a GT lesion
        self.fp = 0          # predicted lesions matching no GT lesion
        self.total_gt = 0    # total GT lesions
        self.detected_gt = 0 # GT lesions matched by a prediction
        self.voxel_tn = 0
        self.voxel_fp = 0
        self.dice_sum = 0.0
        self.dice_n = 0

    def update(self, logits, labels):
        """Accumulate statistics for a batch of logits and binary labels."""
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        preds = (expit(logits) > self.prob_threshold).astype(np.uint8)
        labels = (labels > self.prob_threshold).astype(np.uint8)

        for i in range(preds.shape[0]):
            self._compute_instance(preds[i], labels[i])

    def _compute_instance(self, pred, label):
        if pred.ndim == 4:
            pred = pred[0]
        if label.ndim == 4:
            label = label[0]

        pred_labeled, num_pred = ndimage.label(pred, structure=self.structure)
        label_labeled, num_gt = ndimage.label(label, structure=self.structure)
        self.total_gt += num_gt

        # Voxel-level statistics over GT-negative voxels (specificity) and Dice.
        neg_mask = label == 0
        self.voxel_tn += int(np.sum((pred == 0) & neg_mask))
        self.voxel_fp += int(np.sum((pred == 1) & neg_mask))
        self.dice_sum += self._dice(pred, label)
        self.dice_n += 1

        # TP / FP: each predicted lesion is matched against the dilated GT.
        if num_pred > 0:
            for idx, slice_obj in enumerate(ndimage.find_objects(pred_labeled)):
                if slice_obj is None:
                    continue
                pred_component = pred_labeled[slice_obj] == (idx + 1)

                expanded = self._expand_slice(slice_obj, self.dilation_iter, label.shape)
                label_crop = label[expanded]
                if not np.any(label_crop):
                    self.fp += 1
                    continue

                label_crop_dilated = ndimage.binary_dilation(
                    label_crop, structure=self.structure, iterations=self.dilation_iter
                )
                rel = self._relative_slice(slice_obj, expanded)
                pred_in_expanded = np.zeros_like(label_crop, dtype=bool)
                pred_in_expanded[rel] = pred_component

                overlap = int(np.logical_and(pred_in_expanded, label_crop_dilated).sum())
                if overlap > self.overlap_threshold:
                    self.tp += 1
                else:
                    self.fp += 1

        # FN: each GT lesion is checked for any overlapping prediction.
        if num_gt > 0:
            for idx, slice_obj in enumerate(ndimage.find_objects(label_labeled)):
                if slice_obj is None:
                    continue
                gt_component = label_labeled[slice_obj] == (idx + 1)
                gt_dilated = ndimage.binary_dilation(
                    gt_component, structure=self.structure, iterations=self.dilation_iter
                )
                overlap = int(np.logical_and(gt_dilated, pred[slice_obj]).sum())
                if overlap > self.overlap_threshold:
                    self.detected_gt += 1

    @staticmethod
    def _dice(pred, label) -> float:
        inter = float(np.sum(pred & label))
        denom = float(pred.sum() + label.sum())
        if denom == 0:
            return 1.0  # both empty: perfect agreement
        return 2.0 * inter / denom

    @staticmethod
    def _expand_slice(slice_obj, margin, shape):
        return tuple(
            slice(max(0, s.start - margin), min(dim, s.stop + margin))
            for s, dim in zip(slice_obj, shape)
        )

    @staticmethod
    def _relative_slice(inner, outer):
        return tuple(
            slice(i.start - o.start, i.start - o.start + (i.stop - i.start))
            for i, o in zip(inner, outer)
        )

    def compute(self) -> dict:
        eps = 1e-6
        precision = self.tp / (self.tp + self.fp + eps)
        sensitivity = self.detected_gt / (self.total_gt + eps)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)
        specificity = self.voxel_tn / (self.voxel_tn + self.voxel_fp + eps)
        dice = self.dice_sum / max(self.dice_n, 1)
        return {
            "tp": self.tp, "fp": self.fp,
            "total_gt": self.total_gt, "detected_gt": self.detected_gt,
            "precision": precision, "sensitivity": sensitivity, "f1": f1,
            "voxel_specificity": specificity, "voxel_dice": dice,
        }


# =============================================================================
# Inference
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(args):
    cfg = load_config(args.config)
    sc = cfg["downstream"]["stenosis_detection"]
    patch_size = tuple(sc["patch_shape"])
    overlap_threshold = sc["overlap_threshold_voxels"]
    num_channels = cfg["model"]["num_input_channels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CORASegmentationModel(num_input_channels=num_channels, num_classes=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    loader = get_eval_loader(args.npz_root, args.val_index, label_key=args.label_key)

    metric = LesionDetectionMetric(
        overlap_threshold=overlap_threshold, dilation_iter=args.dilation_iter
    )

    os.makedirs(args.output_dir, exist_ok=True)
    per_case_path = os.path.join(args.output_dir, "per_case_results.csv")
    with open(per_case_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "tp", "fp", "total_gt", "detected_gt", "voxel_dice"])

        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch[args.label_key]
            logits = sliding_window_inference(
                inputs=images, roi_size=patch_size, sw_batch_size=args.sw_batch_size,
                predictor=model, overlap=0.5,
            )
            if logits.shape != labels.shape:
                print(f"[Warning] shape mismatch {logits.shape} vs {labels.shape}; skipping.")
                continue

            case = LesionDetectionMetric(
                overlap_threshold=overlap_threshold, dilation_iter=args.dilation_iter
            )
            case.update(logits, labels)
            metric.update(logits, labels)
            r = case.compute()
            writer.writerow([
                batch["name"][0], r["tp"], r["fp"],
                r["total_gt"], r["detected_gt"], f"{r['voxel_dice']:.4f}",
            ])

    results = metric.compute()
    summary_path = os.path.join(args.output_dir, "summary_results.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(results.keys()))
        writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in results.values()])

    print("\n=== Stenosis Detection (lesion-level) ===")
    print(f"Overlap threshold (TP): >{overlap_threshold} voxels")
    print(f"TP={results['tp']}  FP={results['fp']}  "
          f"GT={results['total_gt']}  detected={results['detected_gt']}")
    print(f"Sensitivity (recall): {results['sensitivity']:.4f}")
    print(f"Precision:            {results['precision']:.4f}")
    print(f"F1:                   {results['f1']:.4f}")
    print(f"Voxel Dice:           {results['voxel_dice']:.4f}")
    print(f"Voxel specificity:    {results['voxel_specificity']:.4f}")
    print(f"\nResults saved to: {summary_path} and {per_case_path}")


def main():
    parser = argparse.ArgumentParser(description="Lesion-level stenosis-detection evaluation.")
    parser.add_argument("--config", default="../../configs/cora_config.yaml")
    parser.add_argument("--npz_root", default="data/npz",
                        help="Root for NPZ paths listed in --val_index.")
    parser.add_argument("--val_index", default="data/val_index.txt",
                        help="Directory of NPZ files, or text file listing one NPZ per line.")
    parser.add_argument("--checkpoint", default="checkpoints/stenosis_best.pth")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--label_key", default="label",
                        help="NPZ key holding the voxel-level lesion mask.")
    parser.add_argument("--dilation_iter", type=int, default=1,
                        help="GT dilation tolerance (voxels) before overlap testing.")
    parser.add_argument("--sw_batch_size", type=int, default=4)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
