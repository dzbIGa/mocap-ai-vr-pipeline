"""
CNN-LSTM Facial Animation Model (NumPy inference)
===================================================
Implements Equation (4) from the manuscript:

    w(t) = f_{CNN-LSTM}(F(t))

where:
    F(t) — facial marker positions at time t  (N_markers × 3)
    w(t) — blendshape weights at time t       (N_blendshapes,) ∈ [0,1]

This module provides:
  1. NumPy-based inference (no PyTorch required for deployment)
  2. Weight loading from the pretrained .npz file
  3. Training script (train.py) using only numpy/scipy

Reference:
    He et al., "The Role of Motion Capture and AI in VR-Based 3D Animation
    Design and Production", submitted to The Visual Computer, 2025.
"""

import numpy as np
import os


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _conv1d(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    1-D convolution, same padding, stride=1.
    x : (C_in, T), W : (C_out, C_in, K), b : (C_out,)
    returns : (C_out, T)
    """
    C_out, C_in, K = W.shape
    pad = K // 2
    x_pad = np.pad(x, ((0, 0), (pad, pad)))
    T = x.shape[1]
    out = np.zeros((C_out, T))
    for k in range(K):
        out += W[:, :, k] @ x_pad[:, k: k + T]
    return out + b[:, None]


class CNNLSTMFacialAnimator:
    """
    Lightweight CNN-LSTM blendshape predictor.

    Architecture
    ------------
    Input  : (T, N_markers * 3)
    Conv1  : 1D conv, 64 filters, kernel=3, ReLU
    Conv2  : 1D conv, 64 filters, kernel=3, ReLU
    LSTM   : 128 hidden units
    FC     : 52 outputs, Sigmoid → blendshape weights ∈ [0,1]

    Training
    --------
    Loss: MSE   |   Optimiser: Adam   |   Early stopping
    """

    def __init__(self, weights_path: str = None):
        self.weights = None
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            self._init_random_weights()

        # LSTM state
        self._h = np.zeros(128)
        self._c = np.zeros(128)

    # ── weight init ────────────────────────────────────────────────────────
    def _init_random_weights(self):
        rng = np.random.default_rng(42)
        scale = 0.1
        self.weights = {
            "conv1_weight": rng.standard_normal((64, 68 * 3, 3)).astype(np.float32) * scale,
            "conv1_bias":   np.zeros(64, dtype=np.float32),
            "conv2_weight": rng.standard_normal((64, 64, 3)).astype(np.float32) * scale,
            "conv2_bias":   np.zeros(64, dtype=np.float32),
            "lstm_weight_ih": rng.standard_normal((512, 64)).astype(np.float32) * scale,
            "lstm_weight_hh": rng.standard_normal((512, 128)).astype(np.float32) * scale,
            "lstm_bias":      np.zeros(512, dtype=np.float32),
            "fc_weight": rng.standard_normal((52, 128)).astype(np.float32) * scale,
            "fc_bias":    np.zeros(52, dtype=np.float32),
        }

    def load_weights(self, path: str):
        data = np.load(path)
        self.weights = {k: data[k] for k in data.files}

    # ── LSTM cell (single step) ────────────────────────────────────────────
    def _lstm_step(self, x: np.ndarray) -> np.ndarray:
        """x : (64,) → output : (128,)"""
        W_ih = self.weights["lstm_weight_ih"]   # (512, 64)
        W_hh = self.weights["lstm_weight_hh"]   # (512, 128)
        b    = self.weights["lstm_bias"]          # (512,)

        gates = W_ih @ x + W_hh @ self._h + b   # (512,)
        i, f, g, o = np.split(gates, 4)
        i, f, o = _sigmoid(i), _sigmoid(f), _sigmoid(o)
        g = np.tanh(g)

        self._c = f * self._c + i * g
        self._h = o * np.tanh(self._c)
        return self._h.copy()

    # ── forward pass ──────────────────────────────────────────────────────
    def predict(self, markers: np.ndarray, reset_state: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        markers : np.ndarray, shape (T, N_markers, 3) or (T, N_markers*3)
            Facial marker positions.
        reset_state : bool
            Reset LSTM hidden state before sequence (True for offline inference).

        Returns
        -------
        weights : np.ndarray, shape (T, 52)
            Predicted blendshape weights ∈ [0, 1].
        """
        T = markers.shape[0]
        x = markers.reshape(T, -1).astype(np.float32)   # (T, 204)

        if reset_state:
            self._h = np.zeros(128, dtype=np.float32)
            self._c = np.zeros(128, dtype=np.float32)

        # CNN: transpose to (C, T) for conv
        x_t = x.T                                         # (204, T)
        c1 = _relu(_conv1d(x_t,
                           self.weights["conv1_weight"],
                           self.weights["conv1_bias"]))    # (64, T)
        c2 = _relu(_conv1d(c1,
                           self.weights["conv2_weight"],
                           self.weights["conv2_bias"]))    # (64, T)

        # LSTM: iterate over time
        out = np.zeros((T, 52), dtype=np.float32)
        for t in range(T):
            h = self._lstm_step(c2[:, t])
            logit = self.weights["fc_weight"] @ h + self.weights["fc_bias"]
            out[t] = _sigmoid(logit)

        return out
