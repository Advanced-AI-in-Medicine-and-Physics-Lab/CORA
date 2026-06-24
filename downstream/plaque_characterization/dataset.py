"""
Volume-level dataset for plaque characterization (multi-label classification).

Each sample is an ENTIRE CCTA volume (no patch sampling). The dataset:
    1. Loads a CCTA volume (HU) from a per-patient NPZ file.
    2. Converts HU to a 4-channel multi-window input.
    3. Resizes / pads the whole volume to a fixed shape so volumes can be batched.
    4. Returns the image tensor and a 2-d multi-label target
       (calcified plaque present, non-calcified plaque present).

The 4-channel windowing matches the CORA pretraining input exactly, so the
pretrained encoder sees the same intensity representation it was trained on.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =============================================================================
# Multi-Window Input Strategy (identical to pretraining)
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
# Volume Resizing (whole-volume, fixed output shape for batching)
# =============================================================================

def resize_volume_to_shape(img: np.ndarray, target_shape) -> np.ndarray:
    """
    Pad-then-center-crop a multi-channel volume (C, D, H, W) to a fixed
    spatial shape (target_shape = (D, H, W)) so that whole volumes of varying
    size can be stacked into a batch.

    Padding uses the per-volume minimum (background) value; cropping is centered.
    """
    td, th, tw = target_shape
    c, d, h, w = img.shape

    # --- Pad smaller dimensions symmetrically ---
    pad_d = max(0, td - d)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_d or pad_h or pad_w:
        pad_width = (
            (0, 0),  # channel
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        )
        img = np.pad(img, pad_width, mode="constant", constant_values=float(img.min()))
        c, d, h, w = img.shape

    # --- Center-crop larger dimensions ---
    start_d = (d - td) // 2
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return img[
        :,
        start_d:start_d + td,
        start_h:start_h + th,
        start_w:start_w + tw,
    ]


# =============================================================================
# Plaque Characterization Dataset
# =============================================================================

class PlaqueCharacterizationDataset(Dataset):
    """
    Volume-level multi-label plaque dataset.

    Args:
        excel_file: Path to an Excel index with one row per patient. Must contain
            the patient identifier column and the two multi-label columns.
        npz_root: Root directory holding per-patient NPZ files
            (npz_root/<name>/CTA_<name>.npz with arrays `CTA_HU` and `CA`).
        target_shape: Fixed (D, H, W) volume shape used for batching.
        id_column: Column holding the patient identifier.
        calcified_column: Column holding the calcified-plaque-present label (0/1).
        noncalcified_column: Column holding the non-calcified-plaque-present label (0/1).
    """

    def __init__(
        self,
        excel_file: str,
        npz_root: str,
        target_shape=(128, 128, 128),
        id_column: str = "Deidentification Patient Name",
        calcified_column: str = "Calcium Plaque",
        noncalcified_column: str = "Soft Plaque",
    ):
        self.df = pd.read_excel(excel_file)
        self.npz_root = npz_root
        self.target_shape = tuple(target_shape)
        self.id_column = id_column
        self.calcified_column = calcified_column
        self.noncalcified_column = noncalcified_column

    def __len__(self) -> int:
        return len(self.df)

    def _load_npz(self, name: str) -> np.ndarray:
        """Load the CCTA volume (HU) from the per-patient NPZ file."""
        npz_path = os.path.join(self.npz_root, name, f"CTA_{name}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        data = np.load(npz_path)
        return data["CTA_HU"]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row[self.id_column])

        # Labels are coerced to binary presence (any positive count -> 1).
        label_calcified = int(int(row[self.calcified_column]) > 0)
        label_noncalcified = int(int(row[self.noncalcified_column]) > 0)
        target = torch.tensor([label_calcified, label_noncalcified], dtype=torch.float32)

        # Whole-volume input: HU -> 4 channels -> fixed shape.
        image_hu = self._load_npz(name)
        image = get_multichannel_input(image_hu)
        image = resize_volume_to_shape(image, self.target_shape)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        return image, target
