"""
Comparative Analysis: Optical vs Inertial vs Markerless Motion Capture
=======================================================================
Reproduces Table 3 / Table 3a from the manuscript.

Reference:
    He et al., "The Role of Motion Capture and AI in VR-Based 3D Animation
    Design and Production", submitted to The Visual Computer, 2025.
"""

import numpy as np
import json


SYSTEMS = {
    "Optical (Vicon)": {
        "accuracy_pct":    95.0,
        "cost_usd":        200_000,
        "latency_ms":      8.2,
        "setup_time_h":    2.5,
        "ai_compatibility": "High",
        "portability":     "Low",
        "key_limitation":  "High cost, complex setup",
    },
    "Inertial (Xsens)": {
        "accuracy_pct":    85.0,
        "cost_usd":        15_000,
        "latency_ms":      12.5,
        "setup_time_h":    0.5,
        "ai_compatibility": "Medium",
        "portability":     "High",
        "key_limitation":  "Drift accumulation over time",
    },
    "Markerless (DeepMotion)": {
        "accuracy_pct":    70.0,
        "cost_usd":        10_000,
        "latency_ms":      18.7,
        "setup_time_h":    0.1,
        "ai_compatibility": "High",
        "portability":     "Very High",
        "key_limitation":  "Lower accuracy, occlusion sensitivity",
    },
}


def print_comparison_table():
    header = f"{'System':<28} {'Accuracy':>10} {'Cost (USD)':>12} {'Latency (ms)':>14} {'Setup (h)':>10}"
    print("\n" + "=" * len(header))
    print("TABLE 3 — Motion Capture System Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, vals in SYSTEMS.items():
        print(f"{name:<28} {vals['accuracy_pct']:>9.1f}% "
              f"${vals['cost_usd']:>10,}  "
              f"{vals['latency_ms']:>12.1f}  "
              f"{vals['setup_time_h']:>9.1f}")
    print("=" * len(header))


def cost_accuracy_tradeoff() -> dict:
    """Return normalised cost-accuracy scores for each system."""
    results = {}
    for name, v in SYSTEMS.items():
        score = v["accuracy_pct"] / (v["cost_usd"] / 1000)   # accuracy per $1K
        results[name] = round(score, 4)
    return results


if __name__ == "__main__":
    print_comparison_table()
    print("\nCost-Accuracy Score (accuracy% per $1K):")
    for name, score in cost_accuracy_tradeoff().items():
        print(f"  {name:<30}  {score:.4f}")
