"""
Motion Capture Data Reader
===========================
Loads raw motion capture data from JSON sample files and normalizes it
(coordinate alignment, temporal resampling, amplitude scaling).

Reference:
    He et al., "The Role of Motion Capture and AI in VR-Based 3D Animation
    Design and Production", submitted to The Visual Computer, 2025.
"""

import numpy as np
import json
import os
from typing import Tuple, Dict


def load_mocap_json(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load motion capture data from JSON file.

    Returns
    -------
    dict : subject_id -> np.ndarray of shape (T, N_markers, 3)
    """
    with open(filepath, "r") as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float64) for k, v in raw.items()}


def normalize(
    positions: np.ndarray,
    target_fps: int = 90,
    source_fps: int = 120,
) -> np.ndarray:
    """
    Normalize raw motion capture data:
      1. Temporal resampling to target_fps
      2. Zero-center coordinates (subtract mean of first frame)
      3. Amplitude scaling to unit body height

    Parameters
    ----------
    positions : np.ndarray, shape (T, N, 3)
    target_fps : int
    source_fps : int

    Returns
    -------
    normalized : np.ndarray, shape (T_new, N, 3)
    """
    T, N, D = positions.shape

    # 1. Temporal resampling
    ratio = target_fps / source_fps
    T_new = max(1, int(T * ratio))
    indices = np.linspace(0, T - 1, T_new)
    t_floor = np.floor(indices).astype(int)
    t_ceil  = np.minimum(t_floor + 1, T - 1)
    alpha   = (indices - t_floor)[:, None, None]
    resampled = (1 - alpha) * positions[t_floor] + alpha * positions[t_ceil]

    # 2. Zero-center
    origin = resampled[0].mean(axis=0, keepdims=True)   # (1, 3)
    centered = resampled - origin[np.newaxis]

    # 3. Amplitude scaling (normalize by approx body height = max Y range)
    y_range = centered[:, :, 1].max() - centered[:, :, 1].min()
    if y_range > 1e-6:
        centered /= y_range

    return centered


def load_and_normalize(
    data_dir: str,
    filename: str = "raw_mocap.json",
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Convenience: load + normalize all subjects."""
    path = os.path.join(data_dir, filename)
    data = load_mocap_json(path)
    return {k: normalize(v, **kwargs) for k, v in data.items()}
