"""
VR Rendering Latency Monitor
==============================
Implements Equation (5) from the manuscript:

    Δt = t_render − t_mocap

Measures frame-by-frame latency between motion capture acquisition
and rendered frame output. Averages are reported per session.

Target: Δt < 20 ms for comfortable VR experience.

Reference:
    He et al., "The Role of Motion Capture and AI in VR-Based 3D Animation
    Design and Production", submitted to The Visual Computer, 2025.
"""

import numpy as np
import time
from typing import List


class LatencyMonitor:
    """
    Records and reports pipeline latency per frame.

    Usage
    -----
        monitor = LatencyMonitor(budget_ms=20)
        t0 = monitor.mark_mocap()
        # ... run pipeline ...
        monitor.mark_render(t0)
        report = monitor.summary()
    """

    def __init__(self, budget_ms: float = 20.0):
        self.budget_ms = budget_ms
        self._records: List[float] = []

    def mark_mocap(self) -> float:
        """Record motion capture acquisition timestamp. Returns t_mocap (s)."""
        return time.perf_counter()

    def mark_render(self, t_mocap: float) -> float:
        """Record render completion, compute Δt. Returns latency_ms."""
        t_render = time.perf_counter()
        delta_ms = (t_render - t_mocap) * 1000.0
        self._records.append(delta_ms)
        return delta_ms

    def summary(self) -> dict:
        """Return latency statistics over all recorded frames."""
        if not self._records:
            return {}
        arr = np.array(self._records)
        return {
            "n_frames":             len(arr),
            "mean_latency_ms":      round(float(arr.mean()), 2),
            "max_latency_ms":       round(float(arr.max()),  2),
            "min_latency_ms":       round(float(arr.min()),  2),
            "std_latency_ms":       round(float(arr.std()),  2),
            "budget_ms":            self.budget_ms,
            "budget_compliance_pct": round(float((arr < self.budget_ms).mean() * 100), 1),
        }

    def reset(self):
        self._records.clear()


def simulate_latency_report(n_frames: int = 300, seed: int = 42) -> dict:
    """
    Simulate the latency profile reported in the paper
    (mean ~17.4 ms, max ~19.8 ms, compliance ~98.2%).
    """
    rng = np.random.default_rng(seed)
    latencies = rng.normal(loc=17.4, scale=1.1, size=n_frames).clip(13.0, 22.0)
    arr = latencies
    return {
        "n_frames":              n_frames,
        "mean_latency_ms":       round(float(arr.mean()), 2),
        "max_latency_ms":        round(float(arr.max()),  2),
        "min_latency_ms":        round(float(arr.min()),  2),
        "std_latency_ms":        round(float(arr.std()),  2),
        "budget_ms":             20.0,
        "budget_compliance_pct": round(float((arr < 20.0).mean() * 100), 1),
    }
