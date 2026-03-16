"""
Evaluation Metrics
==================
Computes accuracy, latency, and production efficiency metrics
as reported in the paper (Tables and Section 3).

Reference:
    He et al., "The Role of Motion Capture and AI in VR-Based 3D Animation
    Design and Production", submitted to The Visual Computer, 2025.
"""

import numpy as np
import json
import os


def motion_accuracy(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """
    Compute motion accuracy metrics.

    Parameters
    ----------
    predicted, ground_truth : np.ndarray (T, N, 3)

    Returns
    -------
    dict with RMSE, MAE, and accuracy_pct
    """
    diff = predicted - ground_truth
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))
    # Accuracy as % of frames within 5 mm threshold
    per_frame = np.sqrt(np.mean(diff ** 2, axis=(-1, -2)))
    accuracy_pct = float((per_frame < 0.005).mean() * 100)
    return {
        "rmse_m":        round(rmse, 5),
        "mae_m":         round(mae,  5),
        "accuracy_pct":  round(accuracy_pct, 1),
    }


def production_efficiency(
    before: dict,
    after: dict,
) -> dict:
    """
    Compute improvement ratios between pre- and post-AI-integration workflows.

    Parameters
    ----------
    before, after : dict with keys matching evaluation_metrics.json
    """
    results = {}
    for k in before:
        if isinstance(before[k], (int, float)):
            b, a = before[k], after[k]
            if b > 0:
                reduction = round((b - a) / b * 100, 1)
                results[f"{k}_improvement_pct"] = reduction
    return results


def load_and_report(metrics_path: str) -> dict:
    """Load evaluation_metrics.json and print a summary report."""
    with open(metrics_path) as f:
        m = json.load(f)

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\n[1] Motion Capture System Accuracy")
    for system, vals in m["motion_capture_systems"].items():
        print(f"  {system:35s}  accuracy={vals['accuracy_pct']:5.1f}%  "
              f"cost=${vals['cost_usd']:>7,}  latency={vals['latency_ms']} ms")

    print("\n[2] Pipeline Performance (AI Integration)")
    b = m["pipeline_performance"]["before_ai_integration"]
    a = m["pipeline_performance"]["after_ai_integration"]
    imp = m["pipeline_performance"]["improvement_pct"]
    print(f"  Production time : {b['production_time_days']}d → {a['production_time_days']}d  "
          f"(−{imp['production_time']}%)")
    print(f"  Motion cleanup  : {b['motion_cleanup_manual_h']}h → {a['motion_cleanup_manual_h']}h  "
          f"(−{imp['motion_cleanup']}%)")
    print(f"  Facial animation: {b['facial_animation_h']}h → {a['facial_animation_h']}h  "
          f"(−{imp['facial_animation']}%)")

    print("\n[3] VR Rendering Latency")
    r = m["vr_rendering"]
    print(f"  Mean latency : {r['mean_latency_ms']} ms  (budget: {r['latency_budget_ms']} ms)")
    print(f"  Max latency  : {r['max_latency_ms']} ms")
    print(f"  Budget compliance: {r['budget_compliance_pct']}%")
    print("=" * 60)
    return m
