"""
Survival metrics for MACE prediction (shared by train.py and eval.py).

Provides Harrell's concordance index (C-index) over the observed risk-set
ordering. Higher predicted risk should correspond to shorter time-to-event.
"""

import numpy as np


def concordance_index(
    event_times: np.ndarray, pred_scores: np.ndarray, events: np.ndarray
) -> float:
    """
    Harrell's concordance index.

    For every comparable pair (one patient with an observed event occurring
    before the other), a pair is concordant if the patient who experienced the
    event earlier has the higher predicted risk score (ties count as 0.5).

    Args:
        event_times: Time-to-event / censoring for each patient [N].
        pred_scores: Predicted log-risk scores [N] (higher = higher risk).
        events: Event indicators [N] (1 = event observed, 0 = censored).

    Returns:
        C-index in [0, 1] (0.5 = random).
    """
    event_times = np.asarray(event_times).reshape(-1)
    pred_scores = np.asarray(pred_scores).reshape(-1)
    events = np.asarray(events).reshape(-1)

    n = len(event_times)
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t_i, t_j = event_times[i], event_times[j]
            e_i, e_j = events[i], events[j]
            if e_i == 1 and t_i < t_j:
                permissible += 1
                if pred_scores[i] > pred_scores[j]:
                    concordant += 1
                elif pred_scores[i] == pred_scores[j]:
                    concordant += 0.5
            if e_j == 1 and t_j < t_i:
                permissible += 1
                if pred_scores[j] > pred_scores[i]:
                    concordant += 1
                elif pred_scores[j] == pred_scores[i]:
                    concordant += 0.5
    return concordant / (permissible + 1e-8)
