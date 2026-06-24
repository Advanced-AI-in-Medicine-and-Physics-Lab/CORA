"""
Structured clinical feature encoding for multimodal MACE prediction.

The structured clinical branch of CORAMultimodalMACE consumes a fixed-length
21-dimensional feature vector built from a per-patient risk-factor dictionary.
The layout below is faithful to the CORA training pipeline:

    Categorical features (one-hot) ............................. 14 dims
        legal sex          [m, f, n, unknown] ................. 4
        HTN_within_a_yr    [0, 1] ............................. 2
        Diabetes status    [0, 1] ............................. 2
        tobacco use        [never, former, yes, unknown] ...... 4
        Chest Pain         [0, 1] ............................. 2

    Continuous features (normalized) ............................ 7 dims
        age, BP Systolic value, BP Diastolic value,
        LDL Cholesterol, HDL Cholesterol, Total Cholesterol,
        BMI value

    Total ...................................................... 21 dims

The continuous values are passed through unchanged here (the downstream MLP in
CORAMultimodalMACE applies its own learned normalization); cohort-level mean
imputation for missing labs is handled upstream in the dataset (see
`DEFAULT_FILL_VALUES`).
"""

from typing import Dict, List

import torch


# =============================================================================
# Schema
# =============================================================================

NUM_CLINICAL_FEATURES = 21

SEX_CATEGORIES = ["m", "f", "n"]            # + unknown bucket
TOBACCO_CATEGORIES = ["never", "former", "yes"]  # + unknown bucket

CONTINUOUS_KEYS = [
    "age",
    "BP Systolic value",
    "BP Diastolic value",
    "LDL Cholesterol",
    "HDL Cholesterol",
    "Total Cholesterol",
    "BMI value",
]

# Cohort-level fallback values for missing labs (used by the dataset loader when
# a field is absent or blank). Documented here so the schema stays self-contained.
DEFAULT_FILL_VALUES = {
    "BMI value": 29.311759776536313,
    "BP Systolic value": 125.48467966573816,
    "BP Diastolic value": 73.13440111420613,
    "Total Cholesterol": 172.42280837858806,
    "LDL Cholesterol": 99.31949882537197,
    "HDL Cholesterol": 50.72304111714507,
}


# =============================================================================
# Encoding
# =============================================================================

def _one_hot(value: str, categories: List[str]) -> List[int]:
    """One-hot encode `value` against `categories`, with a trailing unknown bin."""
    onehot = [0] * (len(categories) + 1)
    value = (value or "").strip().lower()
    if value in categories:
        onehot[categories.index(value)] = 1
    else:
        onehot[-1] = 1
    return onehot


def _binary(value) -> List[int]:
    """Two-dim one-hot for a binary [0, 1] field (defaults to 0)."""
    onehot = [0, 0]
    try:
        idx = int(value)
    except (TypeError, ValueError):
        idx = 0
    onehot[idx if idx in (0, 1) else 0] = 1
    return onehot


def riskdict_to_tensor(risk: Dict) -> torch.Tensor:
    """
    Convert a risk-factor dictionary to a 21-dim clinical feature tensor.

    Args:
        risk: Per-patient demographic / clinical dictionary.

    Returns:
        A float32 tensor of shape [21] (14 categorical one-hot + 7 continuous).
    """
    features: List[float] = []

    # --- Categorical (one-hot, 14 dims) ---
    features.extend(_one_hot(risk.get("legal sex", ""), SEX_CATEGORIES))      # 4
    features.extend(_binary(risk.get("HTN_within_a_yr", 0)))                  # 2
    features.extend(_binary(risk.get("Diabetes status", 0)))                  # 2
    features.extend(_one_hot(risk.get("tobacco use", ""), TOBACCO_CATEGORIES))  # 4
    features.extend(_binary(risk.get("Chest Pain", 0)))                       # 2

    # --- Continuous (7 dims) ---
    for key in CONTINUOUS_KEYS:
        try:
            features.append(float(risk.get(key, DEFAULT_FILL_VALUES.get(key, 0.0))))
        except (TypeError, ValueError):
            features.append(float(DEFAULT_FILL_VALUES.get(key, 0.0)))

    return torch.tensor(features, dtype=torch.float32)


def batch_riskdict_to_tensor(risk_dicts: List[Dict]) -> torch.Tensor:
    """Stack a list of risk-factor dicts into a [B, 21] feature tensor."""
    return torch.stack([riskdict_to_tensor(r) for r in risk_dicts])
