"""軌跡與速度指標。

淨空（clearance）以「真實牆面矩形」計算有號距離（外正內負），
同時檢查控制點本身與相鄰點之間的線段（每 1 cm 取樣），
因為簡報圖中的軌跡是以折線連接各點。
"""
from __future__ import annotations

import numpy as np

from .scenarios import Rect, clearance
from .velocity_qp import arc_lengths, curvature


def sample_polyline(Q: np.ndarray, step: float = 0.01) -> np.ndarray:
    pts = [Q[:1]]
    for a, b in zip(Q[:-1], Q[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        t = np.linspace(0, 1, n + 1)[1:, None]
        pts.append(a + t * (b - a))
    return np.vstack(pts)


def path_metrics(Q: np.ndarray, Q_ref: np.ndarray, rects: list[Rect]) -> dict:
    ds = arc_lengths(Q)
    acc = Q[:-2] - 2 * Q[1:-1] + Q[2:]
    kappa = curvature(Q)
    heading = np.arctan2(np.diff(Q[:, 1]), np.diff(Q[:, 0]))
    dtheta = np.abs(np.angle(np.exp(1j * np.diff(heading))))
    dev = np.linalg.norm(Q - Q_ref, axis=1)
    c_pts = clearance(Q, rects)
    c_line = clearance(sample_polyline(Q), rects)
    return {
        "path_length_m": float(ds.sum()),
        "smoothness_sum_d2": float(np.sum(acc * acc)),        # Σ||Δ²Q||²
        "max_d2_m": float(np.linalg.norm(acc, axis=1).max()),
        "max_curvature_1pm": float(kappa.max()),
        "max_heading_change_deg": float(np.degrees(dtheta.max())) if len(dtheta) else 0.0,
        "min_segment_m": float(ds.min()),
        "max_segment_m": float(ds.max()),
        "mean_dev_from_ref_m": float(dev.mean()),
        "max_dev_from_ref_m": float(dev.max()),
        "min_clearance_points_m": float(c_pts.min()),
        "min_clearance_polyline_m": float(c_line.min()),
        "polyline_collision": bool(c_line.min() < 0),
    }


def velocity_metrics(vres, a_max: float) -> dict:
    v, a, kappa = vres.v, vres.a, vres.kappa
    a_lat = v ** 2 * kappa
    at_limit = np.isclose(v, vres.v_max, atol=5e-3) & (vres.v_max < np.max(vres.v_max) - 1e-9)
    return {
        "travel_time_s": vres.travel_time,
        "mean_speed_mps": float(vres.ds.sum() / vres.travel_time) if np.isfinite(vres.travel_time) else 0.0,
        "peak_speed_mps": float(np.nanmax(v)),
        "max_abs_accel_mps2": float(np.nanmax(np.abs(a))),
        "a_max_violation": bool(np.nanmax(np.abs(a)) > a_max + 1e-6),
        "max_lat_accel_mps2": float(np.nanmax(a_lat)),
        "n_env_limited": int(vres.env_limited.sum()),
        "n_points_at_reduced_limit": int(at_limit.sum()),
        "velocity_status": vres.status,
        "velocity_iterations": vres.iterations,
        "velocity_solve_ms": vres.run_time_ms,
    }
