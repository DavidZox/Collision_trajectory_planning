#!/usr/bin/env python
"""重現比對：把 runs/<實驗>/ 的結果和簡報數位化資料（reference/slide_reference.json）比對。

軌跡：每個數位化色塊對應 n_est 個重現點（最近的點），比較色塊中心與這些點的平均位置，
      並反向檢查每個重現點到最近色塊的距離；誤差以公尺與簡報圖的像素表示。
速度：比對 15 個速度標籤（兩位小數）是否完全相同。
輸出：
    results/reproduction_check.csv
    results/figures/reproduction_check.png
    results/comparison/slideNN.png      原圖（左）與重現圖（右）並排
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner import plotting  # noqa: E402
from planner.experiments import METHOD, RUNS  # noqa: E402
from planner.reference import load_reference, match  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results"


def side_by_side(slide_img: Path, repro_img: Path, out: Path, title: str):
    a = Image.open(slide_img).convert("RGB")
    b = Image.open(repro_img).convert("RGB")
    h = 520
    a = a.resize((int(a.width * h / a.height), h), Image.LANCZOS)
    b = b.resize((int(b.width * h / b.height), h), Image.LANCZOS)
    fig, axes = plt.subplots(1, 2, figsize=((a.width + b.width) / 100, h / 100 + 0.6), dpi=100,
                             gridspec_kw={"width_ratios": [a.width, b.width]})
    for ax, img, t in zip(axes, (a, b), ("original (slide)", "reproduced")):
        ax.imshow(img)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main():
    ref = load_reference()
    rows = []
    for slide, e in sorted(ref["slides"].items(), key=lambda kv: int(kv[0])):
        run = json.load(open(RUNS / e["experiment"] / "metrics.json", encoding="utf-8"))
        Q = run["trajectory"]
        mx, mean = match(e["markers"], Q)
        px = e["m_per_px"]
        row = {"slide": int(slide), "experiment": e["experiment"], "method": METHOD[e["group"]],
               "points": 0 if Q is None else len(Q), "status": run["solver"]["status"],
               "max_err_m": round(mx, 4), "mean_err_m": round(mean, 4), "max_err_px": round(mx / px, 2),
               "labels_match": ""}
        if Q is None:
            row["note"] = "slide 無軌跡（不可行），重現結果亦為 primal infeasible" if not e["markers"] else "不符"
        if "velocity_labels" in e:
            ok = sum(a == b for a, b in zip(e["velocity_labels"], run["velocity_labels"]))
            row["labels_match"] = f"{ok}/{len(e['velocity_labels'])}"
        rows.append(row)
        side_by_side(ROOT / e["image"], RUNS / e["experiment"] / "trajectory.png",
                     RES / "comparison" / f"slide{int(slide):02d}_{e['experiment']}.png",
                     f"slide {slide}: {e['experiment']}")
        print(f"slide {slide:>2s} {e['experiment']:<22s} max {mx:.3f} m ({mx / px:.1f} px)  "
              f"mean {mean:.3f} m  labels {row['labels_match'] or '-'}")

    RES.mkdir(exist_ok=True)
    keys = ["slide", "experiment", "method", "points", "status", "max_err_m", "mean_err_m", "max_err_px",
            "labels_match", "note"]
    with open(RES / "reproduction_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    # 總覽圖：每張投影片的最大誤差（像素），依方法著色
    with plt.rc_context(plotting.metric_style()):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        names = [f"s{r['slide']} {r['experiment']}" for r in rows]
        vals = [r["max_err_px"] for r in rows]
        cols = [plotting.METHOD_COLOR[r["method"]] for r in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, height=0.6, color=cols)
        for yi, r in zip(y, rows):
            txt = f"velocity labels {r['labels_match']}" if r["labels_match"] else ""
            if r["points"] == 0:
                txt = "infeasible, no trajectory (same as slide)"
            if txt:
                ax.text(r["max_err_px"] + 0.05, yi, txt, va="center", fontsize=7, color=plotting.INK2)
        ax.axvline(1.0, ls="--", color=plotting.INK2, lw=1.0)
        ax.text(1.02, len(rows) - 0.4, "1 px", fontsize=7, color=plotting.INK2)
        ax.set_yticks(y, names, fontsize=7)
        ax.set_xlabel("max position error vs. digitized slide (pixels of the slide image)")
        ax.set_title("Reproduction check: reproduced trajectories vs. slide figures", loc="left")
        ax.grid(axis="y", visible=False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in plotting.METHOD_COLOR.values()]
        ax.legend(handles, plotting.METHOD_COLOR.keys(), loc="lower right")
        ax.set_xlim(0, max(vals) * 1.6 + 0.5)
        fig.tight_layout()
        plotting.save(fig, RES / "figures" / "reproduction_check.png")
    print(f"寫入 {RES / 'reproduction_check.csv'}、results/figures/reproduction_check.png、results/comparison/")


if __name__ == "__main__":
    main()
