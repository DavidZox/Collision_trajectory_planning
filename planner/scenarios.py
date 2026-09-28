"""場景定義：牆面障礙物與全域路徑。

簡報中所有測試都是同一種場景：機器人沿 y=0 從 (0,0) 走到 (10,0)，
x 介於 4~6 m 之間有上下兩道牆，中間留一條通道。

- 方法 A（DecompROS）用「點雲」表示牆（slide 29/30 程式碼）。
- 方法 B（ESDF）用「網格 costmap」表示牆（slide 26/27 程式碼）。

兩者都可以轉成軸對齊矩形，給 metrics 計算真實的幾何淨空距離。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def global_path(n: int, start=(0.0, 0.0), goal=(10.0, 0.0)) -> np.ndarray:
    """直線全域路徑（A* 輸出的簡化版），回傳 (n, 2)。"""
    xs = np.linspace(start[0], goal[0], n)
    ys = np.linspace(start[1], goal[1], n)
    return np.stack([xs, ys], axis=1)


@dataclass(frozen=True)
class Rect:
    """軸對齊矩形障礙物 [x0, x1] x [y0, y1]。"""
    x0: float
    x1: float
    y0: float
    y1: float

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """點到矩形的有號距離（外部為正、內部為負）。"""
        pts = np.atleast_2d(pts)
        cx, cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        hx, hy = (self.x1 - self.x0) / 2, (self.y1 - self.y0) / 2
        qx = np.abs(pts[:, 0] - cx) - hx
        qy = np.abs(pts[:, 1] - cy) - hy
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        inside = np.minimum(np.maximum(qx, qy), 0)
        return outside + inside


@dataclass(frozen=True)
class WallPointCloud:
    """方法 A 的障礙物點雲（slide 29/30 的 obs_points）。

    牆 x 範圍 x_range，下牆 y 在 bottom=(y0, y1)，上牆 y 在 top=(y0, y1)；
    每道牆在 x 方向取 n_x 點，y 方向分別取 n_bottom、n_top 點（np.linspace）。
    """
    bottom: tuple = (-1.5, 0.3)
    top: tuple = (1.8, 2.5)
    x_range: tuple = (4.0, 6.0)
    n_x: int = 15
    n_bottom: int = 15
    n_top: int = 10

    def points(self) -> np.ndarray:
        obs_points = []
        # 下半部牆壁
        for x in np.linspace(self.x_range[0], self.x_range[1], self.n_x):
            for y in np.linspace(self.bottom[0], self.bottom[1], self.n_bottom):
                obs_points.append([x, y])
        # 上半部牆壁
        for x in np.linspace(self.x_range[0], self.x_range[1], self.n_x):
            for y in np.linspace(self.top[0], self.top[1], self.n_top):
                obs_points.append([x, y])
        return np.array(obs_points)

    def rects(self) -> list[Rect]:
        x0, x1 = self.x_range
        return [Rect(x0, x1, *self.bottom), Rect(x0, x1, *self.top)]


@dataclass(frozen=True)
class WallChannel:
    """方法 B 的牆面 costmap（slide 26/27 的 _add_wall_obstacles）。

    x_range 內除了通道 channel=(y_lo, y_hi) 以外都是障礙物，上下延伸到地圖邊界。
    """
    x_range: tuple = (4.0, 6.0)
    channel: tuple = (0.3, 0.9)

    def is_obstacle(self, x, y):
        """與原程式相同的判斷式：4.0 <= x <= 6.0 且 not (y_lo < y < y_hi)。"""
        x = np.asarray(x)
        y = np.asarray(y)
        in_x = (self.x_range[0] <= x) & (x <= self.x_range[1])
        in_channel = (self.channel[0] < y) & (y < self.channel[1])
        return in_x & ~in_channel

    def rects(self, y_min: float, y_max: float) -> list[Rect]:
        x0, x1 = self.x_range
        lo, hi = self.channel
        out = []
        if lo > y_min:
            out.append(Rect(x0, x1, y_min, lo))
        if hi < y_max:
            out.append(Rect(x0, x1, hi, y_max))
        return out


def clearance(pts: np.ndarray, rects: list[Rect]) -> np.ndarray:
    """點到所有矩形障礙物的最小有號距離。"""
    pts = np.atleast_2d(pts)
    if not rects:
        return np.full(len(pts), np.inf)
    return np.min([r.signed_distance(pts) for r in rects], axis=0)
