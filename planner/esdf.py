"""網格 ESDF（Euclidean Signed Distance Field）距離場（方法 B，slide 12、26~28）。

建構流程與原程式相同：
    1. costmap：nx = int((x_max - x_min) / res)、ny = int((y_max - y_min) / res)
       （沒有 +1，所以 11 m x 4 m、res=0.05 時為 220 x 80 格），
       格點座標用 np.linspace(x_min, x_max, nx) 判斷是否為障礙物（_add_wall_obstacles）。
    2. ESDF = EDT(自由空間) * res - EDT(障礙物) * res（外正內負）。
    3. 梯度 = np.gradient(ESDF, res)。

查詢（get_esdf_and_grad）有兩種：
    lookup="nearest"   原程式：ix = int((x - x_min) / res)，取該格的距離與「單位化」梯度。
                       這是重現簡報結果的設定（格點值為分段常數，L-BFGS-B 常以
                       ABNORMAL_TERMINATION_IN_LNSRCH 結束）。
    lookup="bilinear"  建議改良：以格中心 x_min + (k + 0.5) res 做雙線性內插，
                       梯度為內插函數的解析梯度（不單位化），成本與梯度一致。

由 colorbar 與軌跡反推出的格點細節（220x80、floor 查詢、單位化梯度）見 README §6。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .scenarios import WallChannel


class GridESDF:
    def __init__(self, walls: WallChannel | None = None, x_min: float = -0.5, x_max: float = 10.5,
                 y_min: float = -1.5, y_max: float = 2.5, resolution: float = 0.05,
                 lookup: str = "nearest"):
        if lookup not in ("nearest", "bilinear"):
            raise ValueError(f"unknown lookup: {lookup}")
        self.walls = walls or WallChannel()
        self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max
        self.res = resolution
        self.lookup = lookup
        self.nx = int((x_max - x_min) / resolution)
        self.ny = int((y_max - y_min) / resolution)
        self.costmap = np.zeros((self.ny, self.nx), dtype=bool)
        self._add_wall_obstacles()
        self._compute_esdf()

    def _add_wall_obstacles(self):
        """填入 Costmap 障礙物區域"""
        x_indices = np.linspace(self.x_min, self.x_max, self.nx)
        y_indices = np.linspace(self.y_min, self.y_max, self.ny)
        X, Y = np.meshgrid(x_indices, y_indices)
        self.costmap[self.walls.is_obstacle(X, Y)] = True

    def _compute_esdf(self):
        dist_outside = distance_transform_edt(~self.costmap) * self.res
        dist_inside = distance_transform_edt(self.costmap) * self.res
        self.esdf = dist_outside - dist_inside
        self.grad_y, self.grad_x = np.gradient(self.esdf, self.res)

    @property
    def extent(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max]

    def rects(self):
        return self.walls.rects(self.y_min, self.y_max)

    def query(self, pts: np.ndarray):
        """向量化查詢：pts (M, 2) -> (dist (M,), grad (M, 2))。"""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        if self.lookup == "nearest":
            ix = np.trunc((pts[:, 0] - self.x_min) / self.res)
            iy = np.trunc((pts[:, 1] - self.y_min) / self.res)
            ix = np.clip(ix, 0, self.nx - 1).astype(int)
            iy = np.clip(iy, 0, self.ny - 1).astype(int)
            dist = self.esdf[iy, ix]
            grad = np.stack([self.grad_x[iy, ix], self.grad_y[iy, ix]], axis=1)
            norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad = np.where(norm > 1e-9, grad / np.maximum(norm, 1e-12), 0.0)
            return dist, grad

        # bilinear：格中心 x_min + (k + 0.5) res
        E = self.esdf
        fx = (pts[:, 0] - self.x_min) / self.res - 0.5
        fy = (pts[:, 1] - self.y_min) / self.res - 0.5
        x0 = np.clip(np.floor(fx), 0, self.nx - 2).astype(int)
        y0 = np.clip(np.floor(fy), 0, self.ny - 2).astype(int)
        tx = np.clip(fx - x0, 0.0, 1.0)
        ty = np.clip(fy - y0, 0.0, 1.0)
        e00, e10 = E[y0, x0], E[y0, x0 + 1]
        e01, e11 = E[y0 + 1, x0], E[y0 + 1, x0 + 1]
        dist = (1 - tx) * (1 - ty) * e00 + tx * (1 - ty) * e10 + (1 - tx) * ty * e01 + tx * ty * e11
        gx = ((1 - ty) * (e10 - e00) + ty * (e11 - e01)) / self.res
        gy = ((1 - tx) * (e01 - e00) + tx * (e11 - e10)) / self.res
        return dist, np.stack([gx, gy], axis=1)

    def get_esdf_and_grad(self, x: float, y: float):
        """與原程式相同的單點介面：dist, grad = esdf_map.get_esdf_and_grad(x, y)"""
        d, g = self.query(np.array([[x, y]]))
        return float(d[0]), g[0]
