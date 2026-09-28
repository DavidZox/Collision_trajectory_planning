"""走廊硬約束 + OSQP 軌跡最佳化（方法 A 階段二，slide 3、25）。

變數 X = [x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}]，OSQP 標準式
    min 1/2 X^T P X + q^T X   s.t.  l <= A X <= u

目標函數：
    平滑項  Σ ||Q_{i+2} - 2 Q_{i+1} + Q_i||^2   權重 w_smooth
    參考項  Σ ||Q_i - Q_ref,i||^2               權重 w_ref
約束：
    走廊    A_i Q_i <= b_i（l = -inf）
    起終點  Q_0 = start、Q_{N-1} = goal（l = u）

重現注意：簡報的結果只有在「平滑項以 w_smooth * M 累加進 P（沒有乘 2）」時才對得上，
即實際等效成本為 1/2·w_smooth·Σ||Δ²Q||² + w_ref·Σ||Q-Q_ref||²。
smooth_p_scale=1.0 為原程式行為（預設）；改成 2.0 即為 slide 25 的公式。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import osqp
import scipy.sparse as sp

from .corridor import corridor_halfspaces

SOLVED = ("solved", "solved inaccurate")


@dataclass
class QPResult:
    Q: np.ndarray | None           # (N, 2) 最佳軌跡；不可行時為 None
    status: str
    iterations: int
    run_time_ms: float             # OSQP setup + solve
    solve_time_ms: float
    wall_time_ms: float            # 含 Python 端矩陣組裝
    objective: float
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.Q is not None


def build_trajectory_qp(ref: np.ndarray, corridor_boxes: np.ndarray, w_smooth: float,
                        w_ref: float, smooth_p_scale: float = 1.0):
    """組出 OSQP 的 (P, q, A, l, u)。"""
    N = len(ref)
    n = 2 * N
    P = np.zeros((n, n))
    q = np.zeros(n)

    # 平滑項：三點二階差分向量 v = [1, -2, 1]，外積矩陣 M = v v^T
    v = np.array([1.0, -2.0, 1.0])
    M = np.outer(v, v)
    for i in range(N - 2):
        for d in range(2):
            idx = [2 * (i + k) + d for k in range(3)]
            P[np.ix_(idx, idx)] += smooth_p_scale * w_smooth * M

    # 參考路徑追蹤項：w_ref (x_i - x_ref,i)^2 -> P[i,i] += 2 w_ref, q[i] += -2 w_ref x_ref,i
    for i in range(N):
        for d in range(2):
            P[2 * i + d, 2 * i + d] += 2.0 * w_ref
            q[2 * i + d] += -2.0 * w_ref * ref[i, d]

    rows, lo, hi = [], [], []
    # 走廊半平面約束（l = -inf）
    for i, (A_i, b_i) in enumerate(corridor_halfspaces(corridor_boxes)):
        for a_row, b in zip(A_i, b_i):
            r = np.zeros(n)
            r[2 * i:2 * i + 2] = a_row
            rows.append(r)
            lo.append(-np.inf)
            hi.append(b)
    # 起終點鎖定（l = u）
    for i in (0, N - 1):
        for d in range(2):
            r = np.zeros(n)
            r[2 * i + d] = 1.0
            rows.append(r)
            lo.append(ref[i, d])
            hi.append(ref[i, d])

    A = np.array(rows)
    return sp.csc_matrix(np.triu(P)), q, sp.csc_matrix(A), np.array(lo), np.array(hi)


def solve_trajectory_qp(ref: np.ndarray, corridor_boxes: np.ndarray, w_smooth: float = 15.0,
                        w_ref: float = 1.0, smooth_p_scale: float = 1.0,
                        osqp_settings: dict | None = None) -> QPResult:
    t0 = time.perf_counter()
    P, q, A, l, u = build_trajectory_qp(ref, corridor_boxes, w_smooth, w_ref, smooth_p_scale)
    settings = {"verbose": False}
    settings.update(osqp_settings or {})
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, **settings)
    res = prob.solve()
    wall = (time.perf_counter() - t0) * 1e3

    status = res.info.status
    # 不可行時 OSQP 1.x 的 res.x 不是 NaN（是 2.1e9），一律以 status 判斷
    Q = res.x.reshape(-1, 2).copy() if status in SOLVED else None
    return QPResult(Q=Q, status=status, iterations=int(res.info.iter),
                    run_time_ms=res.info.run_time * 1e3, solve_time_ms=res.info.solve_time * 1e3,
                    wall_time_ms=wall, objective=float(res.info.obj_val) if Q is not None else float("nan"),
                    info={"n_var": P.shape[0], "n_con": A.shape[0], "nnz_P": int(P.nnz), "nnz_A": int(A.nnz)})
