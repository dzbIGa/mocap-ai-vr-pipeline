"""
Kalman Filter — Motion Smoothing & Noise Reduction
====================================================
Implements Equation (2) from the manuscript:

    P̂_i(t) = K · P_i(t) + (1 − K) · P̂_i(t−1)

Parameters used in the paper:
    K  = 0.3  (Kalman gain, range 0.2–0.4)
    measurement noise = 0.01–0.05
    process noise = low (prioritize stability)

Reference:
    He et al., "The Role of Motion Capture and AI in VR-Based 3D Animation
    Design and Production", submitted to The Visual Computer, 2025.
"""

import numpy as np


def kalman_smooth(positions: np.ndarray, gain: float = 0.3) -> np.ndarray:
    """
    Apply simplified Kalman filter to 3D marker trajectories.

    Parameters
    ----------
    positions : np.ndarray, shape (T, N, 3)
        Raw marker positions. T = frames, N = markers.
    gain : float
        Kalman gain K ∈ (0, 1). Default 0.3 (paper value).

    Returns
    -------
    smoothed : np.ndarray, shape (T, N, 3)
    """
    if not (0.0 < gain < 1.0):
        raise ValueError(f"Kalman gain must be in (0, 1), got {gain}")

    smoothed = np.empty_like(positions, dtype=np.float64)
    smoothed[0] = positions[0]
    for t in range(1, len(positions)):
        smoothed[t] = gain * positions[t] + (1.0 - gain) * smoothed[t - 1]
    return smoothed


def compute_noise_reduction(
    raw: np.ndarray,
    smoothed: np.ndarray,
) -> dict:
    """
    Compute noise reduction statistics after smoothing.

    Returns dict with:
        raw_std        — mean std of raw trajectories per axis
        smoothed_std   — mean std of smoothed trajectories per axis
        reduction_pct  — % noise reduction
    """
    raw_std      = float(np.std(np.diff(raw,      axis=0)))
    smoothed_std = float(np.std(np.diff(smoothed, axis=0)))
    reduction    = 100.0 * (1.0 - smoothed_std / raw_std) if raw_std > 0 else 0.0
    return {
        "raw_jitter_std":      raw_std,
        "smoothed_jitter_std": smoothed_std,
        "noise_reduction_pct": round(reduction, 2),
    }
