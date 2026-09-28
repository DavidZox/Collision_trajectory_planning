#!/usr/bin/env python
"""反推原程式的實作細節（重現過程的紀錄，可重跑驗證）。

原始程式已遺失，簡報只列出部分參數；以下三件事是用「和簡報圖比對誤差最小」決定的：
    1. 走廊 QP 的平滑項在 P 中的係數（w_smooth·M 或 slide 25 公式的 2·w_smooth·M）
    2. ESDF 的格點數、查詢方式與梯度處理（B 組 6 張圖）
    3. C 組（slide 26~28）沒有列出的 EGO 權重
結果寫到 results/variant_search.md。需要先跑 00_extract_slides.py。
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from planner.corridor import generate_corridor  # noqa: E402
from planner.corridor_qp import solve_trajectory_qp  # noqa: E402
from planner.ego_lbfgs import optimize_trajectory  # noqa: E402
from planner.esdf import GridESDF  # noqa: E402
from planner.experiments import find, load_config, resolve  # noqa: E402
from planner.reference import load_reference, match  # noqa: E402
from planner.scenarios import WallChannel, WallPointCloud, global_path  # noqa: E402
from planner.velocity_qp import optimize_velocity_profile_esdf  # noqa: E402

OUT = ROOT / "results" / "variant_search.md"


class VariantESDF(GridESDF):
    """GridESDF 的變體：格點數（220x80 / 221x81）、查詢方式、梯度是否單位化。"""

    def __init__(self, walls, grid="int", lookup="floor", unit_grad=True):
        self.grid, self.variant_lookup, self.unit_grad = grid, lookup, unit_grad
        self.walls = walls
        self.x_min, self.x_max, self.y_min, self.y_max, self.res = -0.5, 10.5, -1.5, 2.5, 0.05
        self.lookup = "nearest"
        extra = 1 if grid == "int+1" else 0
        self.nx = int((self.x_max - self.x_min) / self.res) + extra
        self.ny = int((self.y_max - self.y_min) / self.res) + extra
        self.costmap = np.zeros((self.ny, self.nx), dtype=bool)
        self._add_wall_obstacles()
        self.esdf = distance_transform_edt(~self.costmap) * self.res - distance_transform_edt(self.costmap) * self.res
        self.grad_y, self.grad_x = np.gradient(self.esdf, self.res)
        self.dxl = (self.x_max - self.x_min) / (self.nx - 1)
        self.dyl = (self.y_max - self.y_min) / (self.ny - 1)

    def query(self, pts):
        pts = np.atleast_2d(pts)
        E = self.esdf
        if self.variant_lookup in ("floor", "argmin"):
            if self.variant_lookup == "floor":      # int((x - x_min) / res)
                ix, iy = np.trunc((pts[:, 0] - self.x_min) / self.res), np.trunc((pts[:, 1] - self.y_min) / self.res)
            else:                                   # 最近的 linspace 格點
                ix, iy = np.rint((pts[:, 0] - self.x_min) / self.dxl), np.rint((pts[:, 1] - self.y_min) / self.dyl)
            ix = np.clip(ix, 0, self.nx - 1).astype(int)
            iy = np.clip(iy, 0, self.ny - 1).astype(int)
            d, g = E[iy, ix], np.stack([self.grad_x[iy, ix], self.grad_y[iy, ix]], 1)
        else:                                       # 雙線性（格中心）
            fx = (pts[:, 0] - self.x_min) / self.res - 0.5
            fy = (pts[:, 1] - self.y_min) / self.res - 0.5
            x0 = np.clip(np.floor(fx), 0, self.nx - 2).astype(int)
            y0 = np.clip(np.floor(fy), 0, self.ny - 2).astype(int)
            tx, ty = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
            e00, e10, e01, e11 = E[y0, x0], E[y0, x0 + 1], E[y0 + 1, x0], E[y0 + 1, x0 + 1]
            d = (1 - tx) * (1 - ty) * e00 + tx * (1 - ty) * e10 + (1 - tx) * ty * e01 + tx * ty * e11
            g = np.stack([((1 - ty) * (e10 - e00) + ty * (e11 - e01)) / self.res,
                          ((1 - tx) * (e01 - e00) + tx * (e11 - e10)) / self.res], 1)
        if self.unit_grad:
            n = np.linalg.norm(g, axis=1, keepdims=True)
            g = np.where(n > 1e-9, g / np.maximum(n, 1e-12), 0.0)
        return d, g


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    cfg, ref = load_config(), load_reference()
    slides = ref["slides"]
    md = ["# 原程式實作細節反推紀錄", "", "由 `scripts/tools/search_variants.py` 產生。誤差為與簡報圖數位化標記的最大距離。", ""]

    # 1. 走廊 QP 平滑項係數
    rows = []
    for scale in (1.0, 2.0):
        errs = []
        for s in ("5", "6", "7", "9", "10", "11"):
            e = find(cfg, slides[s]["experiment"])
            p = resolve(e, cfg)
            w = p["walls"]
            obs = WallPointCloud(tuple(w["bottom"]), tuple(w["top"]), tuple(w["x_range"]), w["n_x"], w["n_bottom"], w["n_top"]).points()
            r = global_path(p["N"])
            c = p["corridor"]
            boxes = generate_corridor(r, obs, c["margin"], tuple(c["default_bounds"]), c["search_half_width"])
            Q = solve_trajectory_qp(r, boxes, p["qp"]["w_smooth"], p["qp"]["w_ref"], scale).Q
            errs.append(match(slides[s]["markers"], Q)[0])
        rows.append([f"{scale:g} · w_smooth · M" + (" (slide 25 公式)" if scale == 2 else " (採用)"),
                     *[f"{x * 1000:.0f}" for x in errs], f"{max(errs) * 1000:.0f}"])
        print("corridor smooth scale", scale, [round(x, 4) for x in errs])
    md += ["## 1. 走廊 QP：平滑項在 P 中的係數", "",
           table(["P 的平滑項", "s5", "s6", "s7", "s9", "s10", "s11", "最大 (mm)"], rows), ""]

    # 2. ESDF 變體
    walls = WallChannel((4.0, 6.0), (0.3, 0.9))
    b_slides = ["18", "19", "20", "21", "22", "23"]
    rows = []
    for grid, lookup, unit in itertools.product(("int", "int+1"), ("floor", "argmin", "bilinear"), (True, False)):
        m = VariantESDF(walls, grid, lookup, unit)
        errs = []
        for s in b_slides:
            p = resolve(find(cfg, slides[s]["experiment"]), cfg)
            e = p["ego"]
            Q = optimize_trajectory(global_path(p["N"]), m, e["d_safe"], e["w_smooth"], e["w_obs"], e["w_ref"]).Q
            errs.append(match(slides[s]["markers"], Q)[0])
        rows.append((max(errs), f"{m.nx}x{m.ny}", lookup, "單位化" if unit else "原始", errs))
        print(f"esdf {m.nx}x{m.ny} {lookup:8s} unit={unit!s:5s} worst {max(errs):.3f}")
    rows.sort(key=lambda r: r[0])
    md += ["## 2. ESDF：格點數、查詢方式、梯度處理（B 組 slide 18~23）", "",
           "floor = `int((x - x_min) / res)`；argmin = 最近的 linspace 格點；bilinear = 格中心雙線性內插。", "",
           table(["格點", "查詢", "梯度", *[f"s{s}" for s in b_slides], "最大 (mm)"],
                 [[r[1], r[2], r[3], *[f"{x * 1000:.0f}" for x in r[4]], f"{r[0] * 1000:.0f}"] for r in rows]), ""]

    # 3. C 組 EGO 權重（slide 27 軌跡 + slide 27/28 速度標籤）
    m = GridESDF(walls)
    rows = []
    for ds, ws, wo, wr in itertools.product((0.5, 0.6), (5.0, 10.0, 20.0, 30.0, 50.0, 100.0),
                                            (100.0, 200.0, 500.0, 1000.0, 2000.0), (0.5, 1.0, 2.0, 5.0, 10.0)):
        Q = optimize_trajectory(global_path(15), m, ds, ws, wo, wr).Q
        err = match(slides["27"]["markers"], Q)[0]
        ok = 0
        for s, wa in (("27", 5.0), ("28", 0.1)):
            v = optimize_velocity_profile_esdf(Q, m, w_a=wa)
            ok += sum(f"{a:.2f}" == b for a, b in zip(v.v, slides[s]["velocity_labels"]))
        rows.append((err, ds, ws, wo, wr, ok))
    rows.sort(key=lambda r: (-r[5], r[0]))
    print("C group best:", rows[:3])
    md += ["## 3. C 組（slide 26~28）EGO 權重", "",
           f"共 {len(rows)} 組；依「速度標籤相符數（slide 27 + 28，共 30 個）」再依軌跡誤差排序，前 8 名：", "",
           table(["d_safe", "w_smooth", "w_obs", "w_ref", "軌跡誤差 (mm)", "標籤相符"],
                 [[r[1], r[2], r[3], r[4], f"{r[0] * 1000:.0f}", f"{r[5]}/30"] for r in rows[:8]]), ""]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"寫入 {OUT}")


if __name__ == "__main__":
    main()
