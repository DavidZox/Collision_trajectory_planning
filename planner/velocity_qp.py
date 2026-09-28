"""1D OSQP 速度規劃（Phase 2，slide 24、25、26~31）。

變數代換 q_i = v_i^2，決策向量 X = [q_0..q_{N-1}, a_0..a_{N-2}]。
    目標   Σ w_v (q_i - v_target^2)^2 + Σ w_a a_i^2
           -> P[i,i] += 2 w_v, q[i] += -2 w_v v_target^2, P[N+i,N+i] += 2 w_a
    約束 1 離散運動學（等式）  q_{i+1} - q_i - 2 Δs_i a_i = 0
    約束 2 速限（不等式）      0 <= q_i <= v_max,i^2
    邊界   q_0 = q_{N-1} = 0（靜止起步、靜止停車）
每點速限取三者最小值：
    v_max,i = min(v_target, sqrt(a_lat_max / κ_i), v_env,i)
    v_env,i：ESDF 距離 < d_safety_margin 或走廊寬度 < narrow_width 時為限速值。

重現注意：原程式雖傳入 a_max，但沒有加入 |a_i| <= a_max 約束（slide 28、31 起步
加速度約 0.69 m/s^2 > 0.5）。enforce_a_max=False 為原程式行為；True 為建議改良。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp

from .corridor_qp import SOLVED


def arc_lengths(Q: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(Q, axis=0), axis=1)


def curvature(Q: np.ndarray) -> np.ndarray:
    """離散曲率（三點外接圓，Menger curvature）；起終點為 0。"""
    k = np.zeros(len(Q))
    for i in range(1, len(Q) - 1):
        a = Q[i] - Q[i - 1]
        b = Q[i + 1] - Q[i]
        c = Q[i + 1] - Q[i - 1]
        cross = abs(a[0] * b[1] - a[1] * b[0])
        den = np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c)
        k[i] = 2.0 * cross / den if den > 1e-12 else 0.0
    return k


@dataclass
class VelocityResult:
    v: np.ndarray            # 每點速度 (N,)
    q: np.ndarray            # v^2
    a: np.ndarray            # 每段縱向加速度 (N-1,)
    t: np.ndarray            # 每點到達時間 (N,)
    ds: np.ndarray
    kappa: np.ndarray
    v_max: np.ndarray        # 最終速限
    v_kappa: np.ndarray      # 曲率速限
    v_env: np.ndarray        # 環境速限（ESDF 距離或走廊寬度）
    env_limited: np.ndarray  # bool：該點是否觸發環境限速
    status: str
    iterations: int
    run_time_ms: float
    wall_time_ms: float

    @property
    def travel_time(self) -> float:
        return float(self.t[-1])


def speed_limits(Q: np.ndarray, env_limited: np.ndarray, v_target: float, a_lat_max: float,
                 v_env_limit: float):
    kappa = curvature(Q)
    v_kappa = np.array([np.sqrt(a_lat_max / k) if k > 1e-3 else v_target for k in kappa])
    v_env = np.where(env_limited, v_env_limit, v_target)
    v_max = np.minimum(np.minimum(v_target, v_kappa), v_env)
    return v_max, v_kappa, v_env, kappa


def solve_velocity_qp(ds: np.ndarray, v_max: np.ndarray, v_target: float = 1.0, w_v: float = 1.0,
                      w_a: float = 5.0, a_max: float | None = None, osqp_settings: dict | None = None):
    N = len(v_max)
    n = N + (N - 1)
    P = np.zeros((n, n))
    qv = np.zeros(n)
    for i in range(N):
        P[i, i] += 2.0 * w_v
        qv[i] += -2.0 * w_v * v_target ** 2
    for i in range(N - 1):
        P[N + i, N + i] += 2.0 * w_a

    rows, lo, hi = [], [], []
    # 約束 1：離散運動學方程（等式）
    for i in range(N - 1):
        r = np.zeros(n)
        r[i], r[i + 1], r[N + i] = -1.0, 1.0, -2.0 * ds[i]
        rows.append(r); lo.append(0.0); hi.append(0.0)
    # 約束 2：每點速度上限 0 <= q_i <= v_max,i^2
    for i in range(N):
        r = np.zeros(n)
        r[i] = 1.0
        rows.append(r); lo.append(0.0); hi.append(v_max[i] ** 2)
    # 邊界：靜止起步與靜止停車
    lo[N - 1] = hi[N - 1] = 0.0
    lo[2 * N - 2] = hi[2 * N - 2] = 0.0
    # 建議改良：縱向加速度硬限制
    if a_max is not None:
        for i in range(N - 1):
            r = np.zeros(n)
            r[N + i] = 1.0
            rows.append(r); lo.append(-a_max); hi.append(a_max)

    settings = {"verbose": False}
    settings.update(osqp_settings or {})
    prob = osqp.OSQP()
    prob.setup(sp.csc_matrix(np.triu(P)), qv, sp.csc_matrix(np.array(rows)), np.array(lo), np.array(hi), **settings)
    return prob.solve()


def _time_stamps(ds: np.ndarray, v: np.ndarray) -> np.ndarray:
    """等加速度段的時間：Δt_i = 2 Δs_i / (v_i + v_{i+1})。"""
    dt = np.array([2.0 * s / (v[i] + v[i + 1]) if v[i] + v[i + 1] > 1e-9 else np.inf
                   for i, s in enumerate(ds)])
    return np.concatenate([[0.0], np.cumsum(dt)])


def optimize_velocity_profile(Q_opt: np.ndarray, env_limited: np.ndarray, v_target: float = 1.0,
                              a_max: float = 0.5, a_lat_max: float = 0.5, v_env_limit: float = 0.5,
                              w_v: float = 1.0, w_a: float = 5.0, enforce_a_max: bool = False,
                              osqp_settings: dict | None = None) -> VelocityResult:
    t0 = time.perf_counter()
    ds = arc_lengths(Q_opt)
    v_max, v_kappa, v_env, kappa = speed_limits(Q_opt, env_limited, v_target, a_lat_max, v_env_limit)
    res = solve_velocity_qp(ds, v_max, v_target, w_v, w_a, a_max if enforce_a_max else None, osqp_settings)
    wall = (time.perf_counter() - t0) * 1e3
    N = len(Q_opt)
    if res.info.status in SOLVED:
        q = res.x[:N].copy()
        a = res.x[N:].copy()
        v = np.sqrt(np.maximum(q, 0.0))
    else:
        q = np.full(N, np.nan); a = np.full(N - 1, np.nan); v = np.full(N, np.nan)
    return VelocityResult(v=v, q=q, a=a, t=_time_stamps(ds, v), ds=ds, kappa=kappa, v_max=v_max,
                          v_kappa=v_kappa, v_env=v_env, env_limited=np.asarray(env_limited, bool),
                          status=res.info.status, iterations=int(res.info.iter),
                          run_time_ms=res.info.run_time * 1e3, wall_time_ms=wall)


def optimize_velocity_profile_esdf(Q_opt, esdf_map, v_target=1.0, a_max=0.5, a_lat_max=0.5,
                                   d_safety_margin=0.6, v_obstacle_limit=0.5, **kw) -> VelocityResult:
    """slide 26~28：利用 ESDF 距離場判斷與障礙物距離，dist < d_safety_margin 時限速。"""
    dist, _ = esdf_map.query(Q_opt)
    return optimize_velocity_profile(Q_opt, dist < d_safety_margin, v_target, a_max, a_lat_max,
                                     v_obstacle_limit, **kw)


def optimize_velocity_profile_corridor(Q_opt, corridor_boxes, v_target=1.0, a_max=0.5, a_lat_max=0.5,
                                       v_corridor_limit=0.5, narrow_width=2.0, **kw) -> VelocityResult:
    """slide 29~31：走廊寬度 corridor_boxes[i][2] - corridor_boxes[i][1] < 2.0 m 時限速。"""
    width = corridor_boxes[:, 2] - corridor_boxes[:, 1]
    return optimize_velocity_profile(Q_opt, width < narrow_width, v_target, a_max, a_lat_max,
                                     v_corridor_limit, **kw)
