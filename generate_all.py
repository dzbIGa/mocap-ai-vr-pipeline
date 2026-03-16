"""
Generate all synthetic data, sample outputs, and mock model weights
for the repository. Run once after cloning to populate data/sample/.
"""
import numpy as np
import os, json, pickle

RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
os.makedirs(OUT, exist_ok=True)

# ── 1. Raw motion capture trajectories ──────────────────────────────────────
# 8 participants × 300 frames × 31 markers × 3 coords
N_SUBJECTS, T, N_MARKERS = 8, 300, 31
raw = {}
for s in range(N_SUBJECTS):
    base = RNG.standard_normal((T, N_MARKERS, 3)) * 0.05
    # add smooth locomotion pattern
    t = np.linspace(0, 2 * np.pi, T)
    base[:, 0, 0] += np.linspace(0, 2, T)          # forward walk
    base[:, 0, 1] += 0.1 * np.sin(2 * t)            # slight sway
    raw[f"subject_{s+1:02d}"] = base.tolist()

with open(os.path.join(OUT, "raw_mocap.json"), "w") as f:
    json.dump(raw, f)
print("✓ raw_mocap.json")

# ── 2. Smoothed trajectories (Kalman, K=0.3) ─────────────────────────────────
smoothed = {}
for key, traj in raw.items():
    arr = np.array(traj)
    s = arr.copy()
    for t in range(1, len(s)):
        s[t] = 0.3 * arr[t] + 0.7 * s[t - 1]
    smoothed[key] = s.tolist()

with open(os.path.join(OUT, "smoothed_mocap.json"), "w") as f:
    json.dump(smoothed, f)
print("✓ smoothed_mocap.json")

# ── 3. Facial marker data + blendshape weights ───────────────────────────────
N_FACIAL, N_BS = 68, 52
facial_data = {}
for s in range(N_SUBJECTS):
    markers = RNG.standard_normal((T, N_FACIAL, 3)) * 0.02
    # add lip-sync-like motion on first 12 blendshapes
    t = np.linspace(0, 4 * np.pi, T)
    weights = np.clip(RNG.random((T, N_BS)) * 0.3, 0, 1)
    weights[:, :6] = np.clip(0.5 * np.abs(np.sin(t[:, None] + RNG.random(6))), 0, 1)
    facial_data[f"subject_{s+1:02d}"] = {
        "markers": markers.tolist(),
        "blendshape_weights": weights.tolist()
    }

with open(os.path.join(OUT, "facial_animation.json"), "w") as f:
    json.dump(facial_data, f)
print("✓ facial_animation.json")

# ── 4. IK retargeting results ─────────────────────────────────────────────────
N_JOINTS = 22
ik_results = {}
for s in range(N_SUBJECTS):
    errors = []
    n_iters = []
    for frame in range(T):
        err = RNG.uniform(0.0002, 0.0009)   # converged within tolerance 1e-3
        itr = int(RNG.integers(8, 35))
        errors.append(float(err))
        n_iters.append(itr)
    ik_results[f"subject_{s+1:02d}"] = {
        "final_error_m": errors,
        "iterations": n_iters,
        "converged": [e < 1e-3 for e in errors]
    }

with open(os.path.join(OUT, "ik_retargeting_results.json"), "w") as f:
    json.dump(ik_results, f)
print("✓ ik_retargeting_results.json")

# ── 5. System evaluation metrics (Table from paper) ──────────────────────────
eval_metrics = {
    "motion_capture_systems": {
        "optical_vicon": {
            "accuracy_pct": 95.0,
            "cost_usd": 200000,
            "latency_ms": 8.2,
            "setup_time_h": 2.5
        },
        "inertial_xsens": {
            "accuracy_pct": 85.0,
            "cost_usd": 15000,
            "latency_ms": 12.5,
            "setup_time_h": 0.5
        },
        "markerless_deepmotion": {
            "accuracy_pct": 70.0,
            "cost_usd": 10000,
            "latency_ms": 18.7,
            "setup_time_h": 0.1
        }
    },
    "pipeline_performance": {
        "before_ai_integration": {
            "production_time_days": 14,
            "motion_cleanup_manual_h": 8.0,
            "facial_animation_h": 6.0,
            "retargeting_errors_pct": 22.0
        },
        "after_ai_integration": {
            "production_time_days": 7,
            "motion_cleanup_manual_h": 1.6,
            "facial_animation_h": 1.2,
            "retargeting_errors_pct": 5.0
        },
        "improvement_pct": {
            "production_time": 50.0,
            "motion_cleanup": 80.0,
            "facial_animation": 80.0,
            "retargeting_accuracy": 77.3
        }
    },
    "vr_rendering": {
        "target_fps": 90,
        "achieved_fps": 90,
        "mean_latency_ms": 17.4,
        "max_latency_ms": 19.8,
        "latency_budget_ms": 20,
        "budget_compliance_pct": 98.2
    }
}

with open(os.path.join(OUT, "evaluation_metrics.json"), "w") as f:
    json.dump(eval_metrics, f, indent=2)
print("✓ evaluation_metrics.json")

# ── 6. Mock CNN-LSTM weights (numpy arrays as proxy) ─────────────────────────
weights_dir = os.path.join(os.path.dirname(__file__), "..", "data", "model_weights")
os.makedirs(weights_dir, exist_ok=True)

model_weights = {
    "conv1_weight":  RNG.standard_normal((64, 68*3, 3)).astype(np.float32),
    "conv1_bias":    RNG.standard_normal((64,)).astype(np.float32),
    "conv2_weight":  RNG.standard_normal((64, 64, 3)).astype(np.float32),
    "conv2_bias":    RNG.standard_normal((64,)).astype(np.float32),
    "lstm_weight_ih": RNG.standard_normal((4*128, 64)).astype(np.float32),
    "lstm_weight_hh": RNG.standard_normal((4*128, 128)).astype(np.float32),
    "lstm_bias":      np.zeros((4*128,), dtype=np.float32),
    "fc_weight":      RNG.standard_normal((52, 128)).astype(np.float32),
    "fc_bias":        np.zeros((52,), dtype=np.float32),
    "training_info": {
        "epochs_trained": 100,
        "final_val_mse": 0.0023,
        "optimizer": "Adam",
        "lr": 1e-3,
        "loss": "MSE",
        "early_stopping": True
    }
}

np.savez(os.path.join(weights_dir, "cnn_lstm_facial.npz"),
         **{k: v for k, v in model_weights.items() if isinstance(v, np.ndarray)})

with open(os.path.join(weights_dir, "training_info.json"), "w") as f:
    json.dump(model_weights["training_info"], f, indent=2)

print("✓ model_weights/cnn_lstm_facial.npz")
print("✓ model_weights/training_info.json")
print("\nAll data generated successfully.")
