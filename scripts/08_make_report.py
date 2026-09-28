#!/usr/bin/env python
"""彙整所有實驗：results/metrics.csv、results/summary.md 與比較圖。

圖（results/figures/）：
    metrics_overview.png        各實驗的最小淨空、路徑長、最大曲率、迭代次數
    velocity_profiles.png       速度與加速度剖面：w_a=5、w_a=0.1、w_a=0.1 + a_max 硬約束
    esdf_lookup_comparison.png  B1~B6：最近格查詢（原程式）vs 雙線性查詢（建議）
    head_to_head.png            同場景對照：兩種方法在窄通道與寬通道的軌跡
需要先執行 01~07。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner import plotting  # noqa: E402
from planner.experiments import RUNS, load_config  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
C = plotting.SERIES


def load_runs():
    cfg = load_config()
    order = [e["id"] for e in cfg["experiments"]]
    runs = {}
    for p in RUNS.glob("*/metrics.json"):
        r = json.load(open(p, encoding="utf-8"))
        runs[r["id"]] = r
    ids = [i for i in order if i in runs] + sorted(i for i in runs if i not in order)
    return {i: runs[i] for i in ids}


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]


def traj_csv(exp_id):
    rows = read_csv(RUNS / exp_id / "trajectory.csv")
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}


# ------------------------------------------------------------------ metrics.csv
COLS = ["id", "group", "method", "slide", "suggested", "desc", "status", "iterations", "wall_time_ms",
        "objective", "path_length_m", "smoothness_sum_d2", "max_curvature_1pm", "max_heading_change_deg",
        "max_dev_from_ref_m", "min_clearance_points_m", "min_clearance_polyline_m", "polyline_collision",
        "travel_time_s", "peak_speed_mps", "max_abs_accel_mps2", "a_max_violation", "max_lat_accel_mps2",
        "velocity_labels"]


def flat(r):
    m = r.get("metrics", {})
    row = {"id": r["id"], "group": r["group"], "method": r["method"], "slide": r["slide"] or "",
           "suggested": r["suggested"], "desc": r["desc"], "status": r["solver"]["status"].strip(),
           "iterations": r["solver"]["iterations"], "wall_time_ms": round(r["solver"]["wall_time_ms"], 3),
           "objective": round(r["solver"]["objective"], 5) if r["solver"]["objective"] == r["solver"]["objective"] else ""}
    for k in COLS:
        if k in m:
            v = m[k]
            row[k] = round(v, 5) if isinstance(v, float) else v
    if "velocity_labels" in r:
        row["velocity_labels"] = " ".join(r["velocity_labels"])
    return row


# ------------------------------------------------------------------ figures
def fig_metrics_overview(runs):
    ids = [i for i, r in runs.items() if r["slide"]]
    rs = [runs[i] for i in ids]
    panels = [("min_clearance_polyline_m", "min clearance along trajectory (m)"),
              ("path_length_m", "path length (m)"),
              ("max_curvature_1pm", "max curvature (1/m)"),
              ("iterations", "solver iterations")]
    with plt.rc_context(plotting.metric_style()):
        fig, axes = plt.subplots(1, 4, figsize=(12, 5.6), sharey=True)
        y = np.arange(len(rs))[::-1]
        for ax, (key, label) in zip(axes, panels):
            vals, cols = [], []
            for r in rs:
                v = r["solver"]["iterations"] if key == "iterations" else r["metrics"].get(key, np.nan)
                vals.append(v)
                cols.append(plotting.METHOD_COLOR[r["method"]])
            ax.barh(y, np.nan_to_num(vals), height=0.6, color=cols)
            ax.set_title(label, loc="left", fontsize=9)
            ax.grid(axis="y", visible=False)
            if key == "min_clearance_polyline_m":
                ax.axvline(0, color=plotting.INK, lw=0.8)
                for yi, v, r in zip(y, vals, rs):
                    if not r["metrics"]:
                        ax.text(0.01, yi, "infeasible", va="center", fontsize=7, color=plotting.INK2)
                    elif v < 0:
                        ax.text(0.01, yi, f"hits wall ({v:+.3f})", va="center", fontsize=7, color=plotting.INK)
            if key == "path_length_m":
                ax.set_xlim(9.8, max(np.nan_to_num(vals)) + 0.1)
        axes[0].set_yticks(y, [f"s{r['slide']} {r['id']}" for r in rs], fontsize=7)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in plotting.METHOD_COLOR.values()]
        fig.legend(handles, plotting.METHOD_COLOR.keys(), loc="lower center", ncol=2, frameon=False)
        fig.suptitle("Trajectory metrics of the reproduced slide experiments", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        plotting.save(fig, FIG / "metrics_overview.png")


def fig_velocity_profiles(runs):
    cols = [("ESDF + L-BFGS path (slides 27/28)", [("C2_w_a_5", "w_a = 5 (slide 27)"),
                                                     ("C3_w_a_0.1", "w_a = 0.1 (slide 28)"),
                                                     ("S2_esdf_w_a_0.1_amax", "w_a = 0.1 + |a| <= a_max (suggested)")]),
            ("DecompROS + OSQP path (slides 30/31)", [("D2_w_a_5", "w_a = 5 (slide 30)"),
                                                        ("D3_w_a_0.1", "w_a = 0.1 (slide 31)"),
                                                        ("S1_corridor_w_a_0.1_amax", "w_a = 0.1 + |a| <= a_max (suggested)")])]
    with plt.rc_context(plotting.metric_style()):
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        for j, (title, series) in enumerate(cols):
            ax_v, ax_a = axes[0, j], axes[1, j]
            for k, (eid, lab) in enumerate(series):
                if eid not in runs:
                    continue
                t = traj_csv(eid)
                T = runs[eid]["metrics"]["travel_time_s"]
                ax_v.plot(t["s"], t["v"], "-o", color=C[k], label=f"{lab}, T = {T:.2f} s", markersize=5)
                a = t["a_to_next"]
                ax_a.step(t["s"], np.append(a[:-1], a[-2]), where="post", color=C[k])
                if k == 0:
                    ax_v.plot(t["s"], t["v_max"], "--", color=plotting.INK2, lw=1.0, label="speed limit v_max")
            ax_v.set_title(title, loc="left")
            ax_v.set_ylim(0, 1.25)
            ax_v.set_ylabel("speed (m/s)")
            ax_v.legend(loc="lower center", fontsize=7)
            for s in (0.5, -0.5):
                ax_a.axhline(s, ls="--", color=plotting.INK2, lw=1.0)
            ax_a.text(ax_a.get_xlim()[1], 0.52, "a_max = 0.5", ha="right", va="bottom", fontsize=7, color=plotting.INK2)
            ax_a.axhline(0, color=plotting.AXIS, lw=0.8)
            ax_a.set_ylabel("accel (m/s$^2$)")
            ax_a.set_xlabel("arc length s (m)")
        fig.suptitle("Phase 2 velocity profiles (OSQP): effect of w_a and of enforcing |a| <= a_max",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        plotting.save(fig, FIG / "velocity_profiles.png")


def _walls(ax, channel=(0.3, 0.9)):
    for y0, y1 in ((-1.5, channel[0]), (channel[1], 2.5)):
        ax.add_patch(Rectangle((4.0, y0), 2.0, y1 - y0, color=plotting.AXIS, alpha=0.6, lw=0))


def fig_esdf_lookup(runs):
    pairs = [("B1_base", "S3_bilinear_B1"), ("B2_N30", "S4_bilinear_B2"), ("B3_d_safe_1.5", "S5_bilinear_B3"),
             ("B4_w_smooth_50", "S6_bilinear_B4"), ("B5_w_obs_2000", "S7_bilinear_B5"), ("B6_w_ref_100", "S8_bilinear_B6")]
    labels = {"orig": "nearest-cell + unit gradient (original code)", "bil": "bilinear interpolation (suggested)"}
    with plt.rc_context(plotting.metric_style()):
        fig, axes = plt.subplots(3, 2, figsize=(12, 7.6), sharex=True, sharey=True)
        for ax, (b, s) in zip(axes.ravel(), pairs):
            if b not in runs or s not in runs:
                continue
            _walls(ax)
            stats = []
            for eid, col, key in ((b, C[1], "orig"), (s, C[2], "bil")):
                Q = np.array(runs[eid]["trajectory"])
                m = runs[eid]["metrics"]
                tag = "converged" if runs[eid]["solver"].get("converged_normally") else "ABNORMAL stop"
                ax.plot(Q[:, 0], Q[:, 1], "-o", color=col, markersize=4, label=labels[key])
                stats.append(f"{tag}, clearance {m['min_clearance_polyline_m']:+.2f} m")
            ax.set_title(f"{b} (slide {runs[b]['slide']})\noriginal: {stats[0]}   |   bilinear: {stats[1]}",
                         loc="left", fontsize=8)
            ax.set_xlim(1.5, 8.5)
            ax.set_ylim(-0.3, 1.1)
        for ax in axes[-1]:
            ax.set_xlabel("x (m)")
        for ax in axes[:, 0]:
            ax.set_ylabel("y (m)")
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=2, frameon=False)
        fig.suptitle("ESDF lookup: original nearest-cell vs. bilinear interpolation (gray: walls; clearance = min "
                     "signed distance along the polyline)", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        plotting.save(fig, FIG / "esdf_lookup_comparison.png")


def fig_head_to_head(runs):
    cases = [("narrow channel y in (0.3, 0.9)", (0.3, 0.9), "S9_decomp_narrow_channel", "B1_base"),
             ("wide gap y in (0.3, 1.8)", (0.3, 1.8), "A1_base", "S10_esdf_wide_gap")]
    with plt.rc_context(plotting.metric_style()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for ax, (title, ch, a, b) in zip(axes, cases):
            if a not in runs or b not in runs:
                continue
            _walls(ax, ch)
            stats = []
            for eid, method in ((a, "DecompROS + OSQP"), (b, "ESDF + L-BFGS")):
                Q = np.array(runs[eid]["trajectory"])
                m = runs[eid]["metrics"]
                ax.plot(Q[:, 0], Q[:, 1], "-o", color=plotting.METHOD_COLOR[method], markersize=4, label=method)
                stats.append(f"{eid}: clearance {m['min_clearance_polyline_m']:.2f} m, length {m['path_length_m']:.2f} m")
            ax.axhline(0, color=plotting.INK2, lw=0.8, ls="--")
            ax.set_title(f"{title}\n" + "\n".join(stats), loc="left", fontsize=8)
            ax.set_xlim(-0.2, 10.2)
            ax.set_ylim(-0.4, 2.0)
            ax.set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=2, frameon=False)
        fig.suptitle("Same scenario, both methods (gray: walls, dashed: global path)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        plotting.save(fig, FIG / "head_to_head.png")


# ------------------------------------------------------------------ summary.md
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def status_text(s: str) -> str:
    """L-BFGS-B / OSQP 狀態的簡短說明（scipy 1.15 起線搜尋失敗的訊息只剩 'ABNORMAL: '）。"""
    s = s.strip()
    if s.startswith("ABNORMAL"):
        return "ABNORMAL（線搜尋失敗）"
    if "RELATIVE REDUCTION" in s:
        return "CONVERGENCE（f 相對下降 < 門檻）"
    if "PROJECTED GRADIENT" in s:
        return "CONVERGENCE（梯度 < 門檻）"
    return s


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def write_summary(runs):
    L = ["# 結果摘要", "", "由 `scripts/08_make_report.py` 產生；完整數值見 `results/metrics.csv`。", ""]
    rc = read_csv(RES / "reproduction_check.csv") if (RES / "reproduction_check.csv").exists() else []
    if rc:
        L += ["## 1. 重現比對（與簡報圖數位化結果）", "",
              md_table(["slide", "實驗", "狀態", "最大誤差 (mm)", "最大誤差 (px)", "速度標籤"],
                       [[r["slide"], r["experiment"], status_text(r["status"]), f"{float(r['max_err_m']) * 1000:.0f}",
                         r["max_err_px"], r["labels_match"] or "-"] for r in rc]), ""]

    def m(r, k, nd=3):
        return fmt(r["metrics"].get(k, "-"), nd) if r["metrics"] else "-"

    for title, keys in (("## 2. A 組 DecompROS + OSQP", ("A",)), ("## 3. B 組 ESDF + L-BFGS", ("B",))):
        rows = []
        for r in runs.values():
            if r["group"] in keys and r["slide"]:
                rows.append([r["id"], r["slide"], status_text(r["solver"]["status"]), r["solver"]["iterations"],
                             m(r, "min_clearance_points_m"), m(r, "min_clearance_polyline_m"),
                             m(r, "path_length_m"), m(r, "max_dev_from_ref_m"), m(r, "max_curvature_1pm")])
        L += [title, "", md_table(["實驗", "slide", "求解狀態", "迭代", "點淨空 (m)", "折線淨空 (m)", "路徑長 (m)",
                                   "最大偏離 (m)", "最大曲率 (1/m)"], rows), ""]

    rows = []
    for r in runs.values():
        if r["group"] in ("C", "D") and r.get("velocity_labels"):
            mm = r["metrics"]
            rows.append([r["id"], r["slide"] or "建議", fmt(mm["travel_time_s"], 2), fmt(mm["peak_speed_mps"], 2),
                         fmt(mm["max_abs_accel_mps2"], 3), "是" if mm["a_max_violation"] else "否",
                         fmt(mm["max_lat_accel_mps2"], 3), " ".join(r["velocity_labels"])])
    L += ["## 4. C、D 組與 S1、S2 速度規劃", "",
          md_table(["實驗", "slide", "行駛時間 (s)", "最高速 (m/s)", "最大 \\|a\\| (m/s²)", "超過 a_max", "最大側向加速度",
                    "各點速度 (m/s)"], rows), ""]

    rows = []
    for b, s in (("B1_base", "S3_bilinear_B1"), ("B2_N30", "S4_bilinear_B2"), ("B3_d_safe_1.5", "S5_bilinear_B3"),
                 ("B4_w_smooth_50", "S6_bilinear_B4"), ("B5_w_obs_2000", "S7_bilinear_B5"), ("B6_w_ref_100", "S8_bilinear_B6")):
        if b in runs and s in runs:
            rb, rs = runs[b], runs[s]
            rows.append([f"{b} → {s}", f"{status_text(rb['solver']['status'])} → {status_text(rs['solver']['status'])}",
                         f"{rb['solver']['iterations']} → {rs['solver']['iterations']}",
                         f"{m(rb, 'min_clearance_polyline_m')} → {m(rs, 'min_clearance_polyline_m')}",
                         f"{m(rb, 'path_length_m')} → {m(rs, 'path_length_m')}"])
    L += ["## 5. 建議：ESDF 雙線性查詢（S3~S8）", "",
          md_table(["原程式 → 雙線性", "L-BFGS-B 結束狀態", "迭代", "折線淨空 (m)", "路徑長 (m)"], rows), ""]

    rows = []
    for eid in ("S9_decomp_narrow_channel", "B1_base", "A1_base", "S10_esdf_wide_gap"):
        if eid in runs:
            r = runs[eid]
            rows.append([eid, r["method"], m(r, "min_clearance_polyline_m"), m(r, "path_length_m"),
                         m(r, "max_dev_from_ref_m"), fmt(r["solver"]["wall_time_ms"], 2)])
    L += ["## 6. 同場景對照（S9、S10）", "",
          md_table(["實驗", "方法", "折線淨空 (m)", "路徑長 (m)", "最大偏離 (m)", "單次耗時 (ms)"], rows), ""]

    if (RES / "benchmark.csv").exists():
        bm = read_csv(RES / "benchmark.csv")
        L += ["## 7. 求解時間（`scripts/06_benchmark.py`）", "",
              md_table(["案例", "中位數 (ms)", "p5 (ms)", "p95 (ms)", "迭代"],
                       [[r["case"], fmt(float(r["median_ms"]), 2), fmt(float(r["p05_ms"]), 2),
                         fmt(float(r["p95_ms"]), 2), r["iterations"]] for r in bm]), ""]
    (RES / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    runs = load_runs()
    RES.mkdir(exist_ok=True)
    with open(RES / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in runs.values():
            w.writerow({k: flat(r).get(k, "") for k in COLS})
    fig_metrics_overview(runs)
    fig_velocity_profiles(runs)
    fig_esdf_lookup(runs)
    fig_head_to_head(runs)
    write_summary(runs)
    print(f"{len(runs)} 個實驗 -> results/metrics.csv、results/summary.md、results/figures/")


if __name__ == "__main__":
    main()
