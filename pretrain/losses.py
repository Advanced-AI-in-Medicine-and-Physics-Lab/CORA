"""
Loss functions for synthesis-driven lesion-segmentation pretraining.

L_total = L_Tversky(alpha=0.1, beta=0.9) + L_Focal(gamma=4.0)

Tversky with beta > 0.5 penalizes false negatives more heavily (recall-oriented
for sparse lesions); Focal down-weights easy background voxels.
"""

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """Tversky loss with controllable FP/FN trade-off."""

    def __init__(self, alpha=0.1, beta=0.9, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = ((1 - targets) * probs).sum(dim=1)
        FN = (targets * (1 - probs)).sum(dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky).mean()


class FocalLoss(nn.Module):
    """Focal loss to down-weight easy background voxels."""

    def __init__(self, alpha=0.25, gamma=4.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()


class LesionSegmentationLoss(nn.Module):
    """Combined Tversky + Focal loss for lesion-segmentation pretraining."""

    def __init__(self, tversky_alpha=0.1, tversky_beta=0.9, focal_gamma=4.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal = FocalLoss(gamma=focal_gamma)

    def forward(self, logits, targets):
        return self.tversky(logits, targets) + self.focal(logits, targets)
