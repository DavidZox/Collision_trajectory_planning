#!/usr/bin/env bash
# 一鍵重現：擷取簡報參考資料 -> 四組簡報實驗 -> 建議實驗 -> 效能測試 -> 重現比對 -> 報告
set -euo pipefail
cd "$(dirname "$0")"

# 規劃問題只有幾十維，BLAS 多執行緒反而變慢（CPU 忙碌時 L-BFGS-B 約慢 7 倍），結果不受影響
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
mkdir -p logs
exec > >(tee logs/run_all.log) 2>&1   # 全流程輸出另存一份

python scripts/00_extract_slides.py        # reference/：簡報原圖與數位化資料
python scripts/01_corridor_qp.py           # A1~A7  slide 5~11
python scripts/02_esdf_lbfgs.py            # B1~B6  slide 18~23
python scripts/03_esdf_velocity.py         # C1~C3  slide 26~28
python scripts/04_corridor_velocity.py     # D1~D3  slide 29~31
python scripts/05_suggested.py             # S1~S11 建議實驗
python scripts/06_benchmark.py "$@"        # 求解時間（可加 --repeat N）
python scripts/07_reproduction_check.py    # 與簡報比對
python scripts/tools/search_variants.py    # 實作細節反推紀錄
python scripts/08_make_report.py           # results/metrics.csv、summary.md、比較圖
echo "完成：runs/（各實驗）、results/（彙整與圖）、logs/"
