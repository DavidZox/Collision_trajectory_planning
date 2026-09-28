#!/usr/bin/env python
"""求解時間基準測試（Python 實作，單執行緒 CPU）。

每個案例先暖機 5 次，再重複 --repeat 次（預設 200），記錄每次的 wall time：
    A  走廊生成 + OSQP 軌跡 QP             N = 15 / 30
    B  ESDF + L-BFGS-B（最近格，原程式）    N = 15 / 30
    B' ESDF + L-BFGS-B（雙線性，建議）      N = 15 / 30
    ESDF 建圖（220 x 80 EDT，一次性成本）
    速度 QP（N = 15，Phase 2）
輸出 results/benchmark.csv 與 results/figures/benchmark.png。
簡報 slide 2 的 1~4 ms / 1~5 ms 為 C++ 實作的文獻值，這裡是 Python 版本的實測值。

BLAS 固定單執行緒：L-BFGS-B 內部只有幾十維的向量運算，OpenBLAS 多執行緒在
CPU 忙碌時反而慢 7 倍（實測 88 ms -> 12 ms），且不影響任何結果。
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner import plotting  # noqa: E402
from planner.corridor import generate_corridor  # noqa: E402
from planner.corridor_qp import solve_trajectory_qp  # noqa: E402
from planner.ego_lbfgs import optimize_trajectory  # noqa: E402
from planner.esdf import GridESDF  # noqa: E402
from planner.scenarios import WallChannel, WallPointCloud, global_path  # noqa: E402
from planner.velocity_qp import optimize_velocity_profile_corridor  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results"


def timeit(fn, repeat, warmup=5):
    for _ in range(warmup):
        out = fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return np.array(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=200)
    args = ap.parse_args()
    load = os.getloadavg()[0]            # 共用機器：記錄量測當下的負載
    print(f"load average (1 min) = {load:.1f}, CPUs = {os.cpu_count()}")

    obs = WallPointCloud().points()
    walls = WallChannel((4.0, 6.0), (0.3, 0.9))
    maps = {"nearest": GridESDF(walls, lookup="nearest"), "bilinear": GridESDF(walls, lookup="bilinear")}
    cases = []
    for N in (15, 30):
        ref = global_path(N)

        def corridor_qp(ref=ref):
            boxes = generate_corridor(ref, obs, 0.2, (-2.0, 2.0), 0.5)
            return solve_trajectory_qp(ref, boxes, 15.0, 1.0)
        cases.append(("DecompROS + OSQP", f"corridor + OSQP QP, N={N}", corridor_qp,
                      lambda r: (r.iterations, r.run_time_ms)))
        for lk in ("nearest", "bilinear"):
            def ego(ref=ref, m=maps[lk]):
                return optimize_trajectory(ref, m, 0.6, 5.0, 200.0, 0.5)
            tag = "nearest-cell (original)" if lk == "nearest" else "bilinear (suggested)"
            cases.append(("ESDF + L-BFGS", f"L-BFGS-B {tag}, N={N}", ego, lambda r: (r.nit, float("nan"))))

    cases.append(("ESDF + L-BFGS", "ESDF map build 220x80 (one-off)",
                  lambda: GridESDF(walls), lambda r: (0, float("nan"))))
    ref = global_path(15)
    boxes = generate_corridor(ref, obs, 0.2, (-2.0, 2.0), 0.5)
    Q = solve_trajectory_qp(ref, boxes, 15.0, 1.0).Q
    cases.append(("DecompROS + OSQP", "velocity QP (Phase 2), N=15",
                  lambda: optimize_velocity_profile_corridor(Q, boxes), lambda r: (r.iterations, r.run_time_ms)))

    rows = []
    for method, name, fn, info in cases:
        ts, out = timeit(fn, args.repeat)
        it, internal = info(out)
        rows.append({"method": method, "case": name, "repeat": args.repeat,
                     "median_ms": np.median(ts), "p05_ms": np.percentile(ts, 5), "p95_ms": np.percentile(ts, 95),
                     "iterations": it, "osqp_internal_ms": internal})
        print(f"{name:<45s} median {np.median(ts):7.3f} ms  p95 {np.percentile(ts, 95):7.3f} ms  it={it}")

    RES.mkdir(exist_ok=True)
    with open(RES / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
        f.write(f"# {platform.processor() or platform.machine()} | Python {platform.python_version()} | "
                f"load average {load:.1f} / {os.cpu_count()} CPUs\n")

    with plt.rc_context(plotting.metric_style()):
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        y = np.arange(len(rows))[::-1]
        for yi, r in zip(y, rows):
            c = plotting.METHOD_COLOR[r["method"]]
            ax.barh(yi, r["median_ms"], height=0.55, color=c)
            ax.plot([r["p05_ms"], r["p95_ms"]], [yi, yi], color=plotting.INK2, lw=1.0)
        ax.set_yticks(y, [r["case"] for r in rows], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("wall time per call (ms, log scale) - bar: median, line: 5th-95th percentile")
        ax.set_title(f"Solve time (Python, {args.repeat} runs each; CPU load average {load:.1f} "
                     f"on {os.cpu_count()} cores)", loc="left")
        ax.grid(axis="y", visible=False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in plotting.METHOD_COLOR.values()]
        ax.legend(handles, plotting.METHOD_COLOR.keys(), loc="lower right")
        fig.tight_layout()
        plotting.save(fig, RES / "figures" / "benchmark.png")
    print(f"寫入 {RES / 'benchmark.csv'} 與 results/figures/benchmark.png")


if __name__ == "__main__":
    main()
