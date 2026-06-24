"""
Stenosis-detection data loaders (segmentation formulation).

Each sample is a coronary-artery-centric CCTA volume stored as an NPZ file
containing an HU volume and a voxel-level lesion mask. The pipeline:

    1. Loads the HU volume and lesion mask from NPZ.
    2. Converts HU to a 4-channel multi-window input (matching pretraining).
    3. For training: extracts artery / lesion-centric 96^3 patches with MONAI
       positive/negative sampling and applies geometric + intensity augmentation.
    4. For inference: returns the full volume (batch size 1) for sliding-window
       prediction and lesion-level evaluation.

All default parameters match configs/cora_config.yaml / the manuscript.
"""

import os
import glob
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate
from monai.transforms import (
    Compose,
    MapTransform,
    EnsureChannelFirstd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    CastToTyped,
    RandFlipd,
    RandRotate90d,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAffined,
    ToTensord,
)


# =============================================================================
# Multi-Window Input Strategy (shared with pretraining)
# =============================================================================

def apply_window(img_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply CT windowing: clip HU values and normalize to [0, 1]."""
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(img_hu, low, high)
    return (clipped - low) / (high - low)


def get_multichannel_input(cta_hu: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel HU volume (D, H, W) to a 4-channel input
    (4, D, H, W) using clinically motivated CT windows.

    Channels:
        0 - Fat            (WC=-100, WW=140)
        1 - Soft tissue    (WC=50,   WW=400)
        2 - Angiographic   (WC=350,  WW=700)
        3 - Calcification  (WC=500,  WW=2000)
    """
    windows = [
        (-100, 140),   # Fat
        (50, 400),     # Soft tissue
        (350, 700),    # Angiographic (contrast-enhanced lumen)
        (500, 2000),   # Calcification
    ]
    cta_hu = cta_hu.astype(np.float32)
    channels = [apply_window(cta_hu, wc, ww) for wc, ww in windows]
    return np.stack(channels, axis=0)


# =============================================================================
# NPZ Loading Transforms
# =============================================================================

class LoadStenosisNPZd(MapTransform):
    """Load the HU volume and lesion mask from an NPZ file into the data dict."""

    def __init__(self, image_key: str, label_key: str, npz_key: str = "npz_path"):
        super().__init__([image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key
        self.npz_key = npz_key

    def __call__(self, data):
        d = dict(data)
        npz_data = np.load(d[self.npz_key])
        d[self.image_key] = npz_data["image"]
        d[self.label_key] = npz_data[self.label_key]
        return d


class MultiWindowd(MapTransform):
    """Convert a single-channel HU volume to the 4-channel multi-window input."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = get_multichannel_input(d[key])
        return d


# =============================================================================
# Loaders
# =============================================================================

def _list_npz(npz_root: str, index_file: str) -> list:
    """
    Resolve the list of NPZ files for a split.

    `index_file` may be either a directory containing `*.npz` files or a text
    file listing one NPZ path (absolute, or relative to `npz_root`) per line.
    """
    if os.path.isdir(index_file):
        files = sorted(glob.glob(os.path.join(index_file, "*.npz")))
    else:
        with open(index_file, "r") as f:
            entries = [ln.strip() for ln in f if ln.strip()]
        files = [
            e if os.path.isabs(e) else os.path.join(npz_root, e) for e in entries
        ]
    if not files:
        raise ValueError(f"No NPZ files resolved from '{index_file}'.")
    return files


def get_train_loader(
    npz_root: str,
    index_file: str,
    label_key: str = "label",
    patch_size: Sequence[int] = (96, 96, 96),
    batch_size: int = 4,
    samples_per_image: int = 4,
    pos_ratio: float = 1.0,
    neg_ratio: float = 1.0,
    num_workers: int = 4,
) -> DataLoader:
    """
    Build the training loader with lesion-centric patch sampling and augmentation.

    Positive/negative cropping (`RandCropByPosNegLabeld`) anchors patches on
    lesion voxels so that sparse stenoses are seen often enough during training.
    """
    files = _list_npz(npz_root, index_file)
    data_dicts = [
        {"npz_path": p, "name": os.path.basename(p).replace(".npz", "")} for p in files
    ]

    transforms = Compose([
        LoadStenosisNPZd(image_key="image", label_key=label_key),
        MultiWindowd(keys=["image"]),
        EnsureChannelFirstd(keys=[label_key], channel_dim="no_channel"),
        SpatialPadd(
            keys=["image", label_key], spatial_size=patch_size,
            method="end", mode="constant", constant_values=0,
        ),
        RandCropByPosNegLabeld(
            keys=["image", label_key], label_key=label_key,
            spatial_size=patch_size, pos=pos_ratio, neg=neg_ratio,
            num_samples=samples_per_image, image_key="image", image_threshold=0,
        ),
        CastToTyped(keys=["image", label_key], dtype=[np.float32, np.float32]),
        RandFlipd(keys=["image", label_key], spatial_axis=[0, 1, 2], prob=0.5),
        RandRotate90d(keys=["image", label_key], prob=0.5, max_k=3),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=["image"], prob=0.4, mean=0.0, std=0.1),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15),
        ),
        RandAffined(
            keys=["image", label_key], prob=0.5,
            rotate_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest"),
        ),
        ToTensord(keys=["image", label_key]),
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate,
        drop_last=True,
    )


def get_eval_loader(
    npz_root: str,
    index_file: str,
    label_key: str = "label",
    num_workers: int = 2,
) -> DataLoader:
    """
    Build the evaluation loader: one full volume per batch for sliding-window
    inference and lesion-level connected-component matching.
    """
    files = _list_npz(npz_root, index_file)
    data_dicts = [
        {"npz_path": p, "name": os.path.basename(p).replace(".npz", "")} for p in files
    ]

    transforms = Compose([
        LoadStenosisNPZd(image_key="image", label_key=label_key),
        MultiWindowd(keys=["image"]),
        EnsureChannelFirstd(keys=[label_key], channel_dim="no_channel"),
        ToTensord(keys=["image", label_key]),
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)
    return DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
