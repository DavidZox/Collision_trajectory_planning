"""避障軌跡規劃：DecompROS + OSQP 與 ESDF + L-BFGS（EGO-Planner）兩種局部規劃器的 Python 重現。

模組：
    scenarios    場景（牆面障礙物、全域路徑）
    corridor     DecompROS-like 安全走廊生成
    corridor_qp  走廊硬約束 + OSQP 軌跡最佳化（方法 A）
    esdf         網格 ESDF 距離場
    ego_lbfgs    ESDF 軟約束 + L-BFGS-B 軌跡最佳化（方法 B）
    velocity_qp  1D OSQP 速度規劃（Phase 2）
    metrics      軌跡與速度指標
    plotting     與簡報相同風格的繪圖
    experiments  讀取 configs/experiments.yaml 並執行實驗
"""

__version__ = "1.0.0"
