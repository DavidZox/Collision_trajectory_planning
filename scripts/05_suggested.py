#!/usr/bin/env python
"""S 組：本專案建議的額外實驗（非簡報內容）。

    S1、S2  速度規劃加入 |a| <= a_max 硬約束（原程式雖傳入 a_max 但未使用）
    S3~S8   B1~B6 改用雙線性 ESDF 查詢（原程式為最近格查詢 + 單位化梯度）
    S9、S10 兩種方法交換場景，做同場景對照
    S11     B6（w_ref 100）改雙線性並提高 w_obs，檢查線段是否仍會撞角

用法與 01~04 相同，例如：python scripts/05_suggested.py --exp S3_bilinear_B1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner.experiments import main  # noqa: E402

if __name__ == "__main__":
    main(["A", "B", "C", "D"], suggested=True, log_name="05_suggested")
