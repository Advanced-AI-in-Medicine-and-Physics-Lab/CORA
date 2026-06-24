"""
Anatomy-guided lesion synthesis engine for self-supervised pretraining.

For each sample the dataset:
    1. Loads a CCTA volume (HU) and its coronary artery mask.
    2. Extracts an artery-centric 96^3 patch.
    3. Synthesizes a calcified or non-calcified plaque within the vessel.
    4. Applies Poisson (Beer-Lambert) acquisition noise.
    5. Converts HU to a 4-channel multi-window input.
    6. Applies geometric augmentation.

All default parameters match configs/cora_config.yaml / the manuscript.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, binary_erosion


# =============================================================================
# Multi-Window Input Strategy
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
# Lesion Synthesis Engine
# =============================================================================

class LesionSynthesizer:
    """
    Anatomy-guided lesion synthesis engine.

    Generates synthetic calcified and non-calcified plaques with controlled
    Hounsfield Unit distributions and irregular morphology (1-3 overlapping
    Gaussian blobs, sigma 0.7-2.0), inserting them into coronary artery regions.

    Args:
        calc_hu_range: HU range for calcified plaques (default 800-1500).
        soft_hu_range: HU range for non-calcified plaques (default 30-90).
        blob_sigma: Gaussian sigma range for lesion blob generation (default 0.7-2.0).
    """

    def __init__(
        self,
        calc_hu_range=(800, 1500),
        soft_hu_range=(30, 90),
        blob_sigma=(0.7, 2.0),
    ):
        self.calc_hu_range = calc_hu_range
        self.soft_hu_range = soft_hu_range
        self.sigma_range = blob_sigma

    def _get_random_center(self, mask: np.ndarray, erode_iter: int = 0):
        """Sample a random voxel coordinate within the (optionally eroded) mask."""
        if mask is None or mask.sum() == 0:
            return None

        valid_mask = mask
        if erode_iter > 0:
            eroded = binary_erosion(mask, iterations=erode_iter)
            if eroded.sum() > 0:
                valid_mask = eroded

        coords = np.argwhere(valid_mask > 0)
        return coords[random.randint(0, len(coords) - 1)]

    def _generate_composite_blob(
        self, shape: tuple, center: np.ndarray, num_blobs: int = 2
    ) -> np.ndarray:
        """
        Generate an irregular lesion morphology by superimposing multiple
        Gaussian blobs with random offsets and anisotropic smoothing.
        """
        canvas = np.zeros(shape, dtype=np.float32)

        # Primary blob at center
        canvas[center[0], center[1], center[2]] = 1.0

        # Secondary blobs with small random offsets for irregular shape
        for _ in range(num_blobs - 1):
            offset = [random.randint(-2, 2), random.randint(-3, 3), random.randint(-3, 3)]
            pos = [np.clip(center[i] + offset[i], 0, shape[i] - 1) for i in range(3)]
            canvas[pos[0], pos[1], pos[2]] = random.uniform(0.5, 1.0)

        # Anisotropic Gaussian smoothing to create ellipsoidal morphology
        sigma = [random.uniform(*self.sigma_range) for _ in range(3)]
        blob = gaussian_filter(canvas, sigma=sigma)

        if blob.max() > 0:
            blob /= blob.max()
        return blob

    def synthesize_calcified(self, img_hu: np.ndarray, vessel_mask: np.ndarray):
        """Synthesize a calcified plaque (high-attenuation, sharp boundaries)."""
        center = self._get_random_center(vessel_mask, erode_iter=0)
        if center is None:
            return img_hu, 0, np.zeros_like(img_hu)

        blob = self._generate_composite_blob(
            img_hu.shape, center, num_blobs=random.randint(1, 3)
        )
        target_hu = random.uniform(*self.calc_hu_range)

        # Sigmoid sharpening for hard calcification boundaries
        core = 1.0 / (1.0 + np.exp(-10 * (blob - 0.3)))
        core = (core - core.min()) / (core.max() - core.min())

        img_aug = img_hu * (1.0 - core) + target_hu * core
        lesion_mask = (core > 0.2).astype(np.float32)

        return img_aug, 1, lesion_mask

    def synthesize_soft(self, img_hu: np.ndarray, vessel_mask: np.ndarray):
        """Synthesize a non-calcified (soft) plaque (low-attenuation, vessel-constrained)."""
        center = self._get_random_center(vessel_mask, erode_iter=1)
        if center is None:
            return img_hu, 0, np.zeros_like(img_hu)

        blob = self._generate_composite_blob(
            img_hu.shape, center, num_blobs=random.randint(1, 3)
        )
        target_hu = random.uniform(*self.soft_hu_range)

        # Constrain lesion within smoothed vessel mask
        soft_mask = gaussian_filter(vessel_mask.astype(np.float32), sigma=1.0)
        effective = blob * soft_mask
        if effective.max() > 0:
            effective /= effective.max()

        img_aug = img_hu * (1.0 - effective) + target_hu * effective
        lesion_mask = (effective > 0.2).astype(np.float32)

        return img_aug, 2, lesion_mask

    def __call__(self, img_hu: np.ndarray, vessel_mask: np.ndarray):
        """Randomly synthesize either a calcified or non-calcified plaque."""
        img_hu = img_hu.astype(np.float32)
        if random.random() < 0.5:
            return self.synthesize_calcified(img_hu, vessel_mask)
        else:
            return self.synthesize_soft(img_hu, vessel_mask)


# =============================================================================
# Geometric Augmentation
# =============================================================================

class JointTransform3D:
    """Joint random flipping for image (C, D, H, W) and mask (1, D, H, W)."""

    def __call__(self, img_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        img_np = img_tensor.numpy()
        mask_np = mask_tensor.numpy()

        for axis in [1, 2, 3]:  # spatial axes of (C, D, H, W)
            if random.random() > 0.5:
                img_np = np.flip(img_np, axis=axis).copy()
                mask_np = np.flip(mask_np, axis=axis).copy()

        return torch.from_numpy(img_np), torch.from_numpy(mask_np)


# =============================================================================
# Poisson Noise Simulation
# =============================================================================

def add_poisson_noise(
    ct_image: np.ndarray,
    I0: float = 1e5,
    L: float = 200.0,
    sigma_e: float = 2.0,
) -> np.ndarray:
    """
    Simulate realistic CCTA acquisition noise via a Beer-Lambert approximation.

    Args:
        ct_image: Input image in HU.
        I0: Incident photon count (manuscript: 1e5).
        L: Equivalent path length in mm (manuscript: 200.0).
        sigma_e: Electronic noise standard deviation in HU (manuscript: 2.0).

    Returns:
        Noisy image in HU.
    """
    mu_water = 0.02

    # HU -> linear attenuation coefficient
    mu = (ct_image + 1000.0) / 1000.0 * mu_water
    mu = np.clip(mu, 0, None)

    # Beer-Lambert: transmitted intensity
    I_clean = I0 * np.exp(-mu * L)

    # Poisson photon noise
    I_noisy = np.random.poisson(I_clean).astype(np.float32)
    I_noisy = np.clip(I_noisy, 1, None)

    # Back to HU
    mu_noisy = -np.log(I_noisy / I0) / L
    ct_noisy = (mu_noisy / mu_water * 1000.0) - 1000.0

    # Electronic noise
    ct_noisy += np.random.normal(0, sigma_e, ct_noisy.shape)

    # Mild smoothing
    ct_noisy = gaussian_filter(ct_noisy, sigma=0.3)

    return ct_noisy.astype(ct_image.dtype)


# =============================================================================
# Pretraining Dataset
# =============================================================================

class CORAPretrainingDataset(Dataset):
    """
    Self-supervised pretraining dataset for CORA.

    Args:
        excel_file: Path to an Excel file listing patient identifiers.
                    MUST contain only pretraining patients (all internal and
                    external test patients excluded).
        npz_root: Root directory containing preprocessed NPZ files.
        patch_shape: Spatial dimensions of extracted patches (D, H, W).
        min_mask_voxels: Min vessel mask voxels for artery-centric sampling.
        noise_params: dict with I0 / L / sigma_e for the noise model.
        lesion_params: dict with calc_hu / soft_hu / blob_sigma.
    """

    def __init__(
        self,
        excel_file: str,
        npz_root: str,
        patch_shape=(96, 96, 96),
        min_mask_voxels: int = 50,
        noise_params=None,
        lesion_params=None,
        id_column: str = "Deidentification Patient Name",
    ):
        self.df = pd.read_excel(excel_file)
        self.npz_root = npz_root
        self.patch_shape = np.array(patch_shape)
        self.min_mask_voxels = min_mask_voxels
        self.id_column = id_column

        noise_params = noise_params or {}
        self.noise_I0 = noise_params.get("I0", 1e5)
        self.noise_L = noise_params.get("L", 200.0)
        self.noise_sigma_e = noise_params.get("sigma_e", 2.0)

        lesion_params = lesion_params or {}
        self.synthesizer = LesionSynthesizer(
            calc_hu_range=tuple(lesion_params.get("calcified_hu", (800, 1500))),
            soft_hu_range=tuple(lesion_params.get("soft_hu", (30, 90))),
            blob_sigma=tuple(lesion_params.get("blob_sigma", (0.7, 2.0))),
        )
        self.augmentor = JointTransform3D()

    def __len__(self):
        return len(self.df)

    def _load_npz(self, name: str):
        """Load CCTA volume (HU) and coronary artery mask from NPZ file."""
        npz_path = os.path.join(self.npz_root, name, f"CTA_{name}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        data = np.load(npz_path)
        return data["CTA_HU"], data["CA"]

    def _extract_patch(self, volume: np.ndarray, center: np.ndarray, shape: np.ndarray):
        """Extract a patch centered at the given coordinate with zero-padding."""
        start = center - shape // 2
        end = start + shape

        pad_before = np.maximum(-start, 0)
        pad_after = np.maximum(end - np.array(volume.shape), 0)
        padding = tuple((b, a) for b, a in zip(pad_before, pad_after))

        padded = np.pad(volume, padding, mode="constant", constant_values=volume.min())

        start += pad_before
        end += pad_before
        return padded[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    def __getitem__(self, idx):
        try:
            name = str(self.df.iloc[idx][self.id_column])
            image_hu, mask = self._load_npz(name)

            # Artery-centric sampling: anchor patch center to a vessel voxel
            mask_coords = np.argwhere(mask > 0)
            if len(mask_coords) < self.min_mask_voxels:
                center = np.array(image_hu.shape) // 2
            else:
                center = mask_coords[random.randint(0, len(mask_coords) - 1)]

            patch_hu = self._extract_patch(image_hu, center, self.patch_shape)
            patch_mask = self._extract_patch(mask, center, self.patch_shape)
            del image_hu, mask

            # Synthesize lesion
            img_aug_hu, label, lesion_mask = self.synthesizer(patch_hu, patch_mask)

            # Apply Poisson noise (manuscript values via config)
            img_aug_hu = add_poisson_noise(
                img_aug_hu, I0=self.noise_I0, L=self.noise_L, sigma_e=self.noise_sigma_e
            )

            # Convert to multi-channel input
            img_multi = get_multichannel_input(img_aug_hu)

            img_tensor = torch.from_numpy(img_multi).float()
            mask_tensor = torch.from_numpy(lesion_mask).unsqueeze(0).float()

            img_tensor, mask_tensor = self.augmentor(img_tensor, mask_tensor)

            return img_tensor, torch.tensor([label], dtype=torch.float32), mask_tensor

        except Exception as e:
            print(f"[Warning] Error loading index {idx}: {e}. Sampling a replacement.")
            return self.__getitem__(random.randint(0, len(self.df) - 1))
