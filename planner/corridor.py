"""DecompROS-like 安全走廊生成（方法 A 階段一）。

對每個全域路徑點，只看 x 方向 ±search_half_width 內的障礙物點，
把它們分成「上方」與「下方」兩群，走廊上下界為（slide 5~11 的程式碼）：

    y_min, y_max = default_bounds              # 附近沒有障礙物時的預設值
    y_max = np.min(top_obs[:, 1]) - margin
    y_min = np.max(bottom_obs[:, 1]) + margin

上下分群採用「局部點雲在 y 方向的最大空隙」：空隙就是可通行的通道，
空隙以下是 bottom_obs、以上是 top_obs。這在路徑點本身落在障礙物 y 範圍內時
（例如 y=0 而下牆延伸到 0.3）仍然能找到正確的通道。

輸出 corridor_boxes[i] = [x_i, y_min_i, y_max_i]，與 slide 29 程式碼的索引方式相同
（corridor_boxes[i][2] - corridor_boxes[i][1] 為走廊寬度）。
"""
from __future__ import annotations

import numpy as np


def split_top_bottom(local: np.ndarray, y_ref: float):
    """以 y 方向最大空隙把局部障礙物點分成 (bottom_obs, top_obs)。"""
    ys = np.unique(local[:, 1])
    if len(ys) >= 2:
        gaps = np.diff(ys)
        significant = np.nonzero(gaps > 2.0 * np.median(gaps))[0]
        if len(significant):
            # 優先選包含路徑點的空隙，否則選最寬的空隙
            containing = [k for k in significant if ys[k] <= y_ref <= ys[k + 1]]
            k = containing[0] if containing else significant[np.argmax(gaps[significant])]
            return local[local[:, 1] <= ys[k]], local[local[:, 1] >= ys[k + 1]]
    # 只有一群障礙物：依群中心在路徑點上方或下方決定
    if local[:, 1].mean() >= y_ref:
        return local[:0], local
    return local, local[:0]


def generate_corridor(path: np.ndarray, obs_points: np.ndarray, margin: float = 0.2,
                      default_bounds=(-2.0, 2.0), search_half_width: float = 0.5) -> np.ndarray:
    """回傳 corridor_boxes，形狀 (N, 3)：[x_i, y_min_i, y_max_i]。

    y_min_i > y_max_i 代表該點的走廊為空集合（例如 margin 太大），
    這裡照原樣輸出，交給 OSQP 判定為 primal infeasible（slide 8）。
    """
    boxes = []
    for p in path:
        y_min, y_max = default_bounds
        local = obs_points[np.abs(obs_points[:, 0] - p[0]) < search_half_width]
        if len(local):
            bottom_obs, top_obs = split_top_bottom(local, p[1])
            if len(top_obs):
                y_max = np.min(top_obs[:, 1]) - margin
            if len(bottom_obs):
                y_min = np.max(bottom_obs[:, 1]) + margin
        boxes.append([p[0], y_min, y_max])
    return np.array(boxes, dtype=float)


def corridor_halfspaces(corridor_boxes: np.ndarray):
    """把每個點的走廊寫成半平面 A_i Q_i <= b_i（slide 25 的矩陣構建方式）。

    [0  1] [x_i]    [ y_max_i]
    [0 -1] [y_i] <= [-y_min_i]

    用 l = -inf 的單邊不等式，即使 y_min > y_max 也能通過 OSQP 的資料檢查，
    由求解器回報不可行，而不是在 setup 階段報錯。
    """
    A_i = np.array([[0.0, 1.0], [0.0, -1.0]])
    return [(A_i, np.array([b[2], -b[1]])) for b in corridor_boxes]
