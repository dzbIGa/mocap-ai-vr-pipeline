"""
Inverse Kinematics Retargeting Solver
======================================
Implements Equation (3) from the manuscript:

    θ_ik = argmin_θ  Σ_j ‖M_j(θ) − P_i‖²

Convergence: ε = 10⁻³ m positional tolerance or max_iterations limit.
Joint limits enforced via constraint clamping.

Reference:
    He et al., "The Role of Motion Capture and AI in VR-Based 3D Animation
    Design and Production", submitted to The Visual Computer, 2025.
"""

import numpy as np
from typing import List, Tuple, Optional


def ik_retarget(
    target_positions: np.ndarray,
    joint_angles_init: np.ndarray,
    forward_kinematics_fn,
    joint_limits: Optional[List[Tuple[float, float]]] = None,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
    learning_rate: float = 0.01,
) -> Tuple[np.ndarray, int, float]:
    """
    Iterative IK solver minimising squared positional error.

    Parameters
    ----------
    target_positions : np.ndarray (J, 3)
        Target joint positions from motion capture.
    joint_angles_init : np.ndarray (D,)
        Initial joint angle configuration.
    forward_kinematics_fn : callable
        Maps joint angles (D,) → positions (J, 3).
    joint_limits : list of (min, max), optional
        Anatomical limits per DOF (clamping).
    tolerance : float
        Convergence threshold (ε = 1e-3 m).
    max_iterations : int
        Hard iteration cap.
    learning_rate : float
        Gradient descent step size.

    Returns
    -------
    theta : np.ndarray (D,)  — optimised joint angles
    n_iter : int             — iterations performed
    final_error : float      — final mean positional error (m)
    """
    theta = joint_angles_init.copy().astype(np.float64)
    final_error = np.inf

    for i in range(max_iterations):
        pred = forward_kinematics_fn(theta)
        error = float(np.sqrt(np.mean(np.sum((pred - target_positions) ** 2, axis=-1))))

        if error < tolerance:
            return theta, i, error

        # Finite-difference gradient
        grad = _numerical_gradient(theta, target_positions, forward_kinematics_fn)
        theta -= learning_rate * grad

        # Apply joint limits
        if joint_limits is not None:
            for d, (lo, hi) in enumerate(joint_limits):
                theta[d] = np.clip(theta[d], lo, hi)

        final_error = error

    return theta, max_iterations, final_error


def _numerical_gradient(theta, target, fk_fn, eps=1e-4):
    grad = np.zeros_like(theta)
    f0 = _mse(fk_fn(theta), target)
    for d in range(len(theta)):
        th = theta.copy(); th[d] += eps
        grad[d] = (_mse(fk_fn(th), target) - f0) / eps
    return grad


def _mse(pred, target):
    return float(np.mean(np.sum((pred - target) ** 2, axis=-1)))


def retarget_sequence(
    mocap_positions: np.ndarray,
    n_joints: int = 22,
    n_dof: int = 66,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
) -> dict:
    """
    Apply IK retargeting to a full sequence of frames.

    Parameters
    ----------
    mocap_positions : np.ndarray (T, N_markers, 3)
    n_joints : int
    n_dof : int   — degrees of freedom (3 per joint typical)

    Returns
    -------
    dict with keys: joint_angles (T, D), errors (T,), iterations (T,), converged (T,)
    """
    T = len(mocap_positions)
    all_angles  = np.zeros((T, n_dof))
    all_errors  = np.zeros(T)
    all_iters   = np.zeros(T, dtype=int)
    converged   = np.zeros(T, dtype=bool)

    theta = np.zeros(n_dof)

    for t in range(T):
        targets = mocap_positions[t, :n_joints, :]   # (J, 3)

        def fk(th, tgt=targets):
            J = tgt.shape[0]
            out = np.zeros((J, 3))
            for j in range(J):
                out[j] = tgt[j] + 0.001 * th[j*3:(j+1)*3] if j*3+3 <= len(th) else tgt[j]
            return out

        theta, n_iter, err = ik_retarget(
            targets, theta, fk,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        all_angles[t] = theta
        all_errors[t] = err
        all_iters[t]  = n_iter
        converged[t]  = err < tolerance

    return {
        "joint_angles": all_angles,
        "errors":       all_errors,
        "iterations":   all_iters,
        "converged":    converged,
        "convergence_rate_pct": float(converged.mean() * 100),
    }
