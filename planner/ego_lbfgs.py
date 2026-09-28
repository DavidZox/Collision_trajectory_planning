"""ESDF 軟約束 + L-BFGS-B 軌跡最佳化（EGO-Planner 簡化版，方法 B，slide 12~15）。

總代價 J(Q) = w_smooth J_smooth + w_obs J_obs + w_ref J_ref
    J_smooth = Σ_i ||Q_{i-1} - 2 Q_i + Q_{i+1}||^2                （二階差分）
    J_obs    = Σ_i (d_safe - d(Q_i))^3，只在 d(Q_i) < d_safe 時計入   （三次方懲罰）
    J_ref    = Σ_i ||Q_i - Q_ref,i||^2

障礙物梯度：dJ_obs/dQ_i = -3 w_obs (d_safe - d)^2 ∇d（∇d 由 GridESDF 提供）。
起終點固定：把第一與最後一點的梯度設為 0（L-BFGS-B 就不會移動它們）。
求解器：scipy.optimize.minimize(method="L-BFGS-B", jac=True)，預設參數，初始值為全域路徑。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .esdf import GridESDF


@dataclass
class EGOResult:
    Q: np.ndarray
    success: bool
    message: str
    nit: int
    nfev: int
    cost: float
    terms: dict = field(default_factory=dict)
    wall_time_ms: float = 0.0

    @property
    def converged_normally(self) -> bool:
        return self.message.startswith("CONVERGENCE")


def ego_cost_and_grad(x_flat: np.ndarray, Q_ref: np.ndarray, esdf_map: GridESDF, w_smooth: float,
                      w_obs: float, w_ref: float, d_safe: float):
    Q = x_flat.reshape(-1, 2)
    grad = np.zeros_like(Q)

    # 1. 平滑項（內部拉力）
    acc = Q[:-2] - 2 * Q[1:-1] + Q[2:]
    cost = w_smooth * np.sum(acc * acc)
    g_acc = 2 * w_smooth * acc
    grad[:-2] += g_acc
    grad[1:-1] -= 2 * g_acc
    grad[2:] += g_acc

    # 2. 障礙物項（外部排斥力）
    dist, n = esdf_map.query(Q)
    mask = dist < d_safe
    cost += w_obs * np.sum((d_safe - dist[mask]) ** 3)
    grad[mask] += -3 * w_obs * ((d_safe - dist[mask]) ** 2)[:, None] * n[mask]

    # 3. 參考路徑項（目標拉力）
    diff = Q - Q_ref
    cost += w_ref * np.sum(diff * diff)
    grad += 2 * w_ref * diff

    # 起終點固定
    grad[0] = 0
    grad[-1] = 0
    return cost, grad.ravel()


def cost_terms(Q: np.ndarray, Q_ref: np.ndarray, esdf_map: GridESDF, w_smooth, w_obs, w_ref, d_safe):
    """最終軌跡的各項代價（已乘權重）。"""
    acc = Q[:-2] - 2 * Q[1:-1] + Q[2:]
    dist, _ = esdf_map.query(Q)
    mask = dist < d_safe
    return {
        "J_smooth": float(w_smooth * np.sum(acc * acc)),
        "J_obs": float(w_obs * np.sum((d_safe - dist[mask]) ** 3)),
        "J_ref": float(w_ref * np.sum((Q - Q_ref) ** 2)),
        "n_points_inside_d_safe": int(mask.sum()),
    }


def optimize_trajectory(Q_ref: np.ndarray, esdf_map: GridESDF, d_safe: float = 0.6,
                        w_smooth: float = 5.0, w_obs: float = 200.0, w_ref: float = 0.5,
                        options: dict | None = None) -> EGOResult:
    opts = {"maxiter": 15000, "maxcor": 10}      # scipy 預設值，明列出來方便修改
    opts.update(options or {})
    t0 = time.perf_counter()
    res = minimize(ego_cost_and_grad, Q_ref.ravel().copy(),
                   args=(Q_ref, esdf_map, w_smooth, w_obs, w_ref, d_safe),
                   jac=True, method="L-BFGS-B", options=opts)
    wall = (time.perf_counter() - t0) * 1e3
    Q = res.x.reshape(-1, 2)
    msg = res.message if isinstance(res.message, str) else res.message.decode()
    return EGOResult(Q=Q, success=bool(res.success), message=msg, nit=int(res.nit),
                     nfev=int(res.nfev), cost=float(res.fun),
                     terms=cost_terms(Q, Q_ref, esdf_map, w_smooth, w_obs, w_ref, d_safe),
                     wall_time_ms=wall)
