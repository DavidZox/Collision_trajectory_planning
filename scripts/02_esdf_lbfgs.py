#!/usr/bin/env python
"""B 組：ESDF + L-BFGS（EGO-Planner）軌跡最佳化（簡報 slide 18~23）

用法：
    python scripts/02_esdf_lbfgs.py                                   # 跑該組全部簡報實驗
    python scripts/02_esdf_lbfgs.py --exp <實驗id>                     # 只跑指定實驗
    python scripts/02_esdf_lbfgs.py --exp <實驗id> --set 區塊.參數=值 --name my_run
實驗 id 與參數見 configs/experiments.yaml，輸出在 runs/<實驗id>/。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner.experiments import main  # noqa: E402

if __name__ == "__main__":
    main(["B"], suggested=False, log_name="02_esdf_lbfgs")
