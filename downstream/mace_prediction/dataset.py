"""
Multimodal dataset for MACE (major adverse cardiac event) prediction.

Each sample combines three modalities for a single patient:
    1. Image    - an ENTIRE CCTA volume (HU) -> 4-channel multi-window input,
                  resized/padded to a fixed shape for batching.
    2. Text     - a structured clinical impression string generated from the
                  patient's risk-factor dictionary (consumed by the frozen Qwen
                  text encoder).
    3. Clinical - the same risk-factor dictionary, kept for encoding into the
                  21-dim structured feature vector (see clinical_features.py).

The survival target is (time-to-event, event) where `event` is the MACE
indicator and `time` is the number of days between the CT and the diagnosis
(or last follow-up for censored patients).

The 4-channel windowing matches the CORA pretraining input exactly, so the
pretrained encoder sees the same intensity representation it was trained on.
"""

import os
import json
from typing import Dict, List

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
    spatial shape (target_shape = (D, H, W)) so volumes of varying size can be
    stacked into a batch. Padding uses the per-volume minimum (background)
    value; cropping is centered.
    """
    td, th, tw = target_shape
    c, d, h, w = img.shape

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
# Clinical Impression Text
# =============================================================================

def riskdict_to_text(demo: Dict) -> str:
    """
    Render a risk-factor dictionary as a structured clinical-impression string.

    The text emphasizes cardiovascular risk factors (demographics, vitals, labs,
    medical history, lifestyle) and is the input to the frozen Qwen text encoder.
    """
    parts = [
        "The following is a summary of the patient's demographic and clinical "
        "information. Please focus on key risk factors for cardiovascular disease."
    ]

    # Demographics
    age = f"{int(demo['age'])}-year-old" if "age" in demo else "An individual"
    sex_map = {"m": "male", "f": "female", "n": "patient of unknown sex"}
    sex = sex_map.get(str(demo.get("legal sex", "")).strip().lower(), "patient")
    parts.append(f"This is a {age} {sex}.")

    # Vitals
    vitals = []
    if "BP Systolic value" in demo and "BP Diastolic value" in demo:
        sbp = int(float(demo["BP Systolic value"]))
        dbp = int(float(demo["BP Diastolic value"]))
        vitals.append(f"blood pressure is {sbp}/{dbp} mmHg")
    if "BMI value" in demo:
        vitals.append(f"BMI is {float(demo['BMI value']):.1f} kg/m2")
    if vitals:
        parts.append("The patient's " + ", ".join(vitals) + ".")

    # Labs
    labs = []
    for key, label in [
        ("LDL Cholesterol", "LDL"),
        ("HDL Cholesterol", "HDL"),
        ("Total Cholesterol", "total cholesterol"),
    ]:
        if key in demo:
            labs.append(f"{label} of {demo[key]} mg/dL")
    if labs:
        parts.append("Laboratory results show " + ", ".join(labs) + ".")

    # Medical history
    history = []
    if demo.get("Diabetes status") == 1:
        history.append("diabetes")
    if demo.get("HTN_within_a_yr") == 1:
        history.append("hypertension diagnosed within the past year")
    if demo.get("Chest Pain") == 1:
        history.append("recent episodes of chest pain")
    if history:
        parts.append("Medical history includes " + ", ".join(history) + ".")

    # Lifestyle
    tobacco_map = {
        "never": "never smoked",
        "former": "is a former smoker",
        "yes": "is a current smoker",
    }
    if "tobacco use" in demo:
        usage = tobacco_map.get(
            str(demo["tobacco use"]).strip().lower(), "unknown smoking history"
        )
        parts.append(f"Regarding lifestyle, the patient {usage}.")

    return " ".join(parts)


# =============================================================================
# Risk-Factor Loading (with cohort-level imputation)
# =============================================================================

FILL_VALUES = {
    "BMI value": 29.311759776536313,
    "tobacco use": "Smoking history unknown",
    "BP Systolic value": 125.48467966573816,
    "BP Diastolic value": 73.13440111420613,
    "Total Cholesterol": 172.42280837858806,
    "LDL Cholesterol": 99.31949882537197,
    "HDL Cholesterol": 50.72304111714507,
}


def load_risk_factors(json_path: str) -> Dict:
    """
    Load and clean a patient's risk-factor dictionary from a JSON file.

    The JSON stores demographics under a `Demographic` key. Missing or blank
    fields are filled with cohort-level defaults so that both the text and the
    structured feature vector are well-defined.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            demo_data = json.load(f).get("Demographic", {})
    except (OSError, json.JSONDecodeError):
        demo_data = {}

    cleaned = {}
    for key, value in demo_data.items():
        blank = value is None or (isinstance(value, str) and value.strip() == "")
        if blank:
            cleaned[key] = FILL_VALUES.get(key, 0.0)
        else:
            cleaned[key] = value
    return cleaned


# =============================================================================
# MACE Dataset
# =============================================================================

class MACEDataset(Dataset):
    """
    Multimodal survival dataset for MACE prediction.

    Args:
        excel_file: Excel index (one row per patient) with the identifier,
            outcome, and survival-time columns.
        npz_root: Root directory of per-patient NPZ files
            (npz_root/<name>/CTA_<name>.npz with array `CTA_HU`).
        risk_root: Root directory of per-patient risk-factor JSON files
            (risk_root/<name>/infor.json). If None, an empty risk dict is used.
        target_shape: Fixed (D, H, W) volume shape used for batching.
        id_column / event_column / time_column: Excel column names.
    """

    def __init__(
        self,
        excel_file: str,
        npz_root: str,
        risk_root: str = None,
        target_shape=(128, 128, 128),
        id_column: str = "Deidentification Patient Name",
        event_column: str = "MACE2",
        time_column: str = "days between CT and dx",
    ):
        self.df = pd.read_excel(excel_file)
        self.npz_root = npz_root
        self.risk_root = risk_root
        self.target_shape = tuple(target_shape)
        self.id_column = id_column
        self.event_column = event_column
        self.time_column = time_column

    def __len__(self) -> int:
        return len(self.df)

    def get_events(self) -> List[int]:
        """Return the event labels (useful for stratified splitting / sampling)."""
        return self.df[self.event_column].astype(int).tolist()

    def _load_npz(self, name: str) -> np.ndarray:
        """Load the CCTA volume (HU) from the per-patient NPZ file."""
        npz_path = os.path.join(self.npz_root, name, f"CTA_{name}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        return np.load(npz_path)["CTA_HU"]

    def _load_risk(self, name: str) -> Dict:
        """Load the per-patient risk-factor dict (empty if no risk_root)."""
        if self.risk_root is None:
            return {}
        return load_risk_factors(os.path.join(self.risk_root, name, "infor.json"))

    def __getitem__(self, idx) -> Dict:
        row = self.df.iloc[idx]
        name = str(row[self.id_column])
        event = int(row[self.event_column])
        time = float(row[self.time_column])

        # Image: HU -> 4 channels -> fixed shape.
        image_hu = self._load_npz(name)
        image = get_multichannel_input(image_hu)
        image = resize_volume_to_shape(image, self.target_shape)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        # Clinical text + structured risk dict.
        risk = self._load_risk(name)
        text = riskdict_to_text(risk) if risk else "No clinical information available."

        return {
            "name": name,
            "image": image,
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
            "text": text,
            "risk_dict": risk,
        }


# =============================================================================
# Collate
# =============================================================================

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate multimodal samples: stack image / time / event tensors and keep the
    text strings and risk dicts as lists (the model tokenizes text and the
    feature encoder consumes the dicts).
    """
    return {
        "name": [b["name"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
        "text": [b["text"] for b in batch],
        "risk_dict": [b["risk_dict"] for b in batch],
    }
