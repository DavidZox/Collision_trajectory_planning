"""繪圖。

1. 重現圖（plot_corridor_scene、plot_esdf_scene）：沿用簡報圖的 matplotlib 樣式
   （標題、圖例、顏色、網格、速度標籤），方便與原圖並排比對。
2. 新增的指標圖（速度剖面、比較圖）：固定的分類色順序、細線、實線淡色網格、
   不使用雙 y 軸（速度與加速度分成上下兩張子圖）。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# 指標圖的調色盤（依固定順序使用，不循環）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_COLOR = {"DecompROS + OSQP": SERIES[0], "ESDF + L-BFGS": SERIES[1]}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, WASH = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec"
DPI = 150


def metric_style():
    """新指標圖的 rcParams（重現圖不使用）。"""
    return {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.5, "lines.markersize": 6, "font.size": 9,
    }


def _velocity_labels(ax, Q, v, dy=0.12):
    for (x, y), vi in zip(Q, v):
        ax.text(x, y + dy, f"{vi:.2f}m/s", ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="darkblue", bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5,
                                            edgecolor="none"))


def plot_corridor_scene(obs_points, ref, corridor_boxes, Q, title, velocity=None, status=None,
                        figsize=(10, 5), path_alpha=0.6):
    """slide 5~11（無速度）與 slide 29~31（有速度標籤）的圖。"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(obs_points[:, 0], obs_points[:, 1], c="gray", s=15, alpha=0.6, label="Costmap Points")
    ax.plot(corridor_boxes[:, 0], corridor_boxes[:, 2], "g--", label="Auto-generated Corridor Boundary")
    ax.plot(corridor_boxes[:, 0], corridor_boxes[:, 1], "g--")
    ax.plot(ref[:, 0], ref[:, 1], "r--o", alpha=path_alpha, label="Global Path")
    if Q is not None:
        ax.plot(Q[:, 0], Q[:, 1], "b-o", linewidth=2.5, label="OSQP Trajectory")
        if velocity is not None:
            _velocity_labels(ax, Q, velocity)
    else:
        # 與 slide 8 相同：不可行時沒有軌跡，只保留圖例
        ax.plot([], [], "b-o", linewidth=2.5, label="OSQP Trajectory")
        ax.text(0.02, 0.04, f"OSQP status: {status}\n(corridor y_min > y_max -> no feasible trajectory)",
                transform=ax.transAxes, fontsize=9, color="#b00000",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#b00000"))
    ax.set_ylim(-1.5, 2.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_esdf_scene(esdf_map, ref, Q, title, velocity=None, figsize=(10, 5)):
    """slide 18~23（無速度）與 slide 26~28（有速度標籤）的圖。"""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(esdf_map.esdf, extent=esdf_map.extent, origin="lower", cmap="jet", alpha=0.5)
    fig.colorbar(im, ax=ax, label="ESDF Distance (m)")
    for k, r in enumerate(esdf_map.rects()):
        ax.add_patch(Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, color="black", alpha=0.4,
                               label="Obstacle" if k == 0 else None))
    ax.plot(ref[:, 0], ref[:, 1], "r--o", alpha=0.7, label="Global Path")
    ax.plot(Q[:, 0], Q[:, 1], "m-o", linewidth=2.5, label="EGO-Planner Trajectory")
    if velocity is not None:
        _velocity_labels(ax, Q, velocity)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_velocity_profile(vres, title, a_max=None):
    """速度剖面（新增指標圖）：上圖 v(s) 與速限，下圖每段縱向加速度 a(s)。"""
    s = np.concatenate([[0.0], np.cumsum(vres.ds)])
    with plt.rc_context(metric_style()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2]})
        # 環境限速區（窄走廊 / 靠近障礙物）以淡灰底色標示
        idx = np.nonzero(vres.env_limited)[0]
        if len(idx):
            for k, grp in enumerate(np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)):
                ax1.axvspan(s[grp[0]] - 0.05, s[grp[-1]] + 0.05, color=WASH, zorder=0,
                            label="environment speed-limit zone" if k == 0 else None)
        ax1.plot(s, vres.v_max, "--", color=INK2, linewidth=1.2, label="speed limit v_max")
        ax1.plot(s, vres.v, "-o", color=SERIES[0], label="optimized speed v")
        ax1.set_ylabel("speed (m/s)")
        ax1.set_ylim(0, 1.2)
        ax1.set_title(title, loc="left")
        ax1.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

        a_step = np.append(vres.a, vres.a[-1])
        ax2.step(s, a_step, where="post", color=SERIES[0], label="longitudinal accel a")
        if a_max is not None:
            ax2.axhline(a_max, ls="--", color=INK2, lw=1.0, label=f"+/- a_max = {a_max:g}")
            ax2.axhline(-a_max, ls="--", color=INK2, lw=1.0)
        ax2.axhline(0, color=AXIS, lw=0.8)
        ax2.set_ylabel("accel (m/s$^2$)")
        ax2.set_xlabel("arc length s (m)")
        ax2.legend(loc="upper right")
        fig.text(0.99, 0.005, f"travel time {vres.travel_time:.2f} s", ha="right", va="bottom",
                 fontsize=8, color=INK2)
        fig.tight_layout()
    return fig


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
