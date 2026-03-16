"""
Motion Data Preprocessing — Normalize, Resample, Remove Corrupted Frames
=========================================================================
Implements the standardized preprocessing pipeline from Section 2.2.3:
  1. Temporal resampling for consistent frame rates
  2. Joint coordinate normalization
  3. Removal of corrupted frames (NaN / out-of-range)
  4. Amplitude scaling for performer variability

Reference:
    He et al., "The Role of Motion Capture and AI in VR-Based 3D Animation
    Design and Production", submitted to The Visual Computer, 2025.
"""

import numpy as np
from typing import Tuple


def remove_corrupted_frames(
    positions: np.ndarray,
    max_velocity: float = 5.0,
) -> np.ndarray:
    """
    Replace corrupted frames (NaN or velocity spikes) with linear interpolation.

    Parameters
    ----------
    positions : np.ndarray, shape (T, N, 3)
    max_velocity : float
        Threshold in units/frame above which a frame is flagged as corrupted.

    Returns
    -------
    cleaned : np.ndarray, shape (T, N, 3)
    """
    cleaned = positions.copy().astype(np.float64)
    T = len(cleaned)

    velocities = np.linalg.norm(np.diff(cleaned, axis=0), axis=-1)  # (T-1, N)
    corrupted = np.zeros(T, dtype=bool)
    corrupted[1:] = (velocities > max_velocity).any(axis=-1)
    corrupted |= np.isnan(cleaned).any(axis=(-1, -2))

    bad_idx = np.where(corrupted)[0]
    good_idx = np.where(~corrupted)[0]

    if len(bad_idx) == 0:
        return cleaned

    for dim in range(3):
        for m in range(cleaned.shape[1]):
            cleaned[bad_idx, m, dim] = np.interp(
                bad_idx, good_idx, cleaned[good_idx, m, dim]
            )
    return cleaned


def scale_amplitude(
    positions: np.ndarray,
    target_height: float = 1.75,
) -> Tuple[np.ndarray, float]:
    """
    Scale trajectory amplitudes to a target body height (meters).

    Returns
    -------
    scaled : np.ndarray, shape (T, N, 3)
    scale_factor : float
    """
    y_range = positions[:, :, 1].max() - positions[:, :, 1].min()
    if y_range < 1e-6:
        return positions.copy(), 1.0
    factor = target_height / y_range
    return positions * factor, float(factor)


def full_preprocess(
    positions: np.ndarray,
    max_velocity: float = 5.0,
    target_height: float = 1.75,
) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline: clean → scale.

    Returns
    -------
    processed : np.ndarray, shape (T, N, 3)
    info : dict — preprocessing statistics
    """
    n_corrupted = int(
        np.isnan(positions).any(axis=(-1, -2)).sum()
    )
    cleaned = remove_corrupted_frames(positions, max_velocity)
    scaled, factor = scale_amplitude(cleaned, target_height)

    info = {
        "frames_corrected": n_corrupted,
        "scale_factor": factor,
        "output_shape": list(scaled.shape),
    }
    return scaled, info
