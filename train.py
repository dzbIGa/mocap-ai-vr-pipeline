"""
Training Script — CNN-LSTM Facial Animation Model
===================================================
Trains the blendshape weight predictor using:
  - Loss:      Mean Squared Error (MSE)
  - Optimiser: Adam (lr = 1e-3)
  - Stopping:  Early stopping (patience = 10 epochs)

Usage
-----
    python -m pipeline.facial_animation.train \
        --data  data/sample/facial_animation.json \
        --out   data/model_weights/cnn_lstm_facial.npz \
        --epochs 100

Reference:
    He et al., "The Role of Motion Capture and AI in VR-Based 3D Animation
    Design and Production", submitted to The Visual Computer, 2025.
"""

import numpy as np
import json
import os
import argparse
from pipeline.facial_animation.cnn_lstm_model import CNNLSTMFacialAnimator, _sigmoid, _relu, _conv1d


# ── Adam optimiser state ──────────────────────────────────────────────────────
class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params: dict, grads: dict) -> dict:
        self.t += 1
        updated = {}
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            updated[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated


def mse(pred, target):
    return float(np.mean((pred - target) ** 2))


def load_data(path):
    with open(path) as f:
        raw = json.load(f)
    X, Y = [], []
    for subj in raw.values():
        X.append(np.array(subj["markers"], dtype=np.float32))         # (T, 68, 3)
        Y.append(np.array(subj["blendshape_weights"], dtype=np.float32))  # (T, 52)
    return X, Y


def train(data_path, out_path, epochs=100, lr=1e-3, patience=10, val_split=0.2):
    X_all, Y_all = load_data(data_path)

    # train/val split by subject
    n_val = max(1, int(len(X_all) * val_split))
    X_train, Y_train = X_all[n_val:], Y_all[n_val:]
    X_val,   Y_val   = X_all[:n_val], Y_all[:n_val]

    model = CNNLSTMFacialAnimator()
    optim = Adam(lr=lr)

    best_val, wait, best_weights = np.inf, 0, None
    history = {"train_mse": [], "val_mse": []}

    for epoch in range(1, epochs + 1):
        train_losses = []
        for x, y in zip(X_train, Y_train):
            pred = model.predict(x, reset_state=True)
            loss = mse(pred, y)

            # Numerical gradient on fc_weight only (lightweight demo)
            eps = 1e-4
            grad_fc = np.zeros_like(model.weights["fc_weight"])
            for i in range(grad_fc.shape[0]):
                for j in range(grad_fc.shape[1]):
                    model.weights["fc_weight"][i, j] += eps
                    l_plus = mse(model.predict(x, reset_state=True), y)
                    model.weights["fc_weight"][i, j] -= eps
                    grad_fc[i, j] = (l_plus - loss) / eps

            grads = {k: np.zeros_like(v) for k, v in model.weights.items()}
            grads["fc_weight"] = grad_fc
            model.weights = optim.step(model.weights, grads)
            train_losses.append(loss)

        val_losses = [mse(model.predict(x, reset_state=True), y)
                      for x, y in zip(X_val, Y_val)]
        t_mse = float(np.mean(train_losses))
        v_mse = float(np.mean(val_losses))
        history["train_mse"].append(t_mse)
        history["val_mse"].append(v_mse)

        print(f"Epoch {epoch:3d}/{epochs}  train_mse={t_mse:.4f}  val_mse={v_mse:.4f}")

        if v_mse < best_val:
            best_val, wait = v_mse, 0
            best_weights = {k: v.copy() for k, v in model.weights.items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Save best weights
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **best_weights)
    print(f"\nModel saved → {out_path}  (best val_mse={best_val:.4f})")
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="data/sample/facial_animation.json")
    ap.add_argument("--out",    default="data/model_weights/cnn_lstm_facial.npz")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr",     type=float, default=1e-3)
    args = ap.parse_args()
    train(args.data, args.out, args.epochs, args.lr)
