#!/usr/bin/env python
"""從簡報擷取結果圖並數位化，產生重現比對用的參考資料。

輸出：
    reference/slides/slideNN.png     簡報中的原始結果圖（slide 5~11、18~23、26~31）
    reference/slide_reference.json   每張圖數位化後的軌跡點（資料座標）、
                                     速度標籤（人工判讀）、投影片上的參數文字

數位化方式：
    1. 由座標軸外框與刻度線（x: 0,2,...,10；y: 2.5,...,-1.5）求出像素 -> 公尺的線性轉換。
    2. 以顏色遮罩（OSQP 軌跡為藍色 'b'，EGO 軌跡為洋紅色 'm'）找出標記點，
       做半徑 3 px 的形態學 opening 去掉線段，只留下圓形標記。
    3. 右上角圖例中的標記會被排除；兩個重疊的標記會合成一個較大的色塊，
       以面積估計包含幾個點（n_est），比對時用這幾個點的平均位置。
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner.experiments import load_config  # noqa: E402

PPTX = ROOT / "避障軌跡規劃.pptx"
OUT = ROOT / "reference"
TRAJ_COLOR = {"A": (0, 0, 255), "D": (0, 0, 255), "B": (191, 0, 191), "C": (191, 0, 191)}
COLOR_TOL = {"A": 60, "D": 60, "B": 80, "C": 80}
SINGLE_MARKER_AREA = 67.0   # 預設 markersize 在 100 dpi 下約 8 px 直徑

# 速度標籤：由投影片放大後人工判讀（slide 26~31）
VELOCITY_LABELS = {
    26: "0.00 0.68 0.85 0.92 0.96 0.98 0.99 0.99 0.99 0.98 0.96 0.92 0.85 0.68 0.00",
    27: "0.00 0.64 0.78 0.81 0.74 0.50 0.50 0.50 0.50 0.50 0.74 0.81 0.78 0.64 0.00",
    28: "0.00 0.98 1.00 1.00 0.98 0.50 0.50 0.50 0.50 0.50 0.98 1.00 1.00 0.98 0.00",
    29: "0.00 0.68 0.85 0.92 0.96 0.98 0.99 0.99 0.99 0.98 0.96 0.92 0.85 0.68 0.00",
    30: "0.00 0.65 0.79 0.81 0.74 0.50 0.50 0.50 0.50 0.50 0.74 0.81 0.79 0.65 0.00",
    31: "0.00 0.98 1.00 1.00 0.98 0.50 0.50 0.50 0.50 0.50 0.98 1.00 1.00 0.98 0.00",
}


def slide_pictures(prs):
    def walk(shapes):
        for sh in shapes:
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from walk(sh.shapes)
            elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                yield sh
    for i, s in enumerate(prs.slides, 1):
        text = "\n".join(sh.text_frame.text.strip() for sh in s.shapes
                         if sh.has_text_frame and sh.text_frame.text.strip())
        yield i, list(walk(s.shapes)), text


def calibrate(im):
    """由外框與刻度線求像素 -> 資料座標的轉換。"""
    dark = im.sum(axis=2) < 150
    H, W = dark.shape
    rows = np.nonzero(dark.sum(axis=1) > 0.3 * W)[0]
    cols = np.nonzero(dark.sum(axis=0) > 0.25 * H)[0]
    top, bottom, left = rows[0], rows[-1], cols[0]
    right = [c for c in cols if c > left + 300][0]
    tick = im.sum(axis=2) < 250

    def group(v):
        g = []
        for a in v:
            if g and a - g[-1][-1] <= 1:
                g[-1].append(a)
            else:
                g.append([a])
        return [float(np.mean(x)) for x in g]

    xt = group([x for x in range(left - 2, right + 3) if tick[bottom + 2, x] and tick[bottom + 3, x]])
    yt = group([y for y in range(top - 2, bottom + 3) if tick[y, left - 2] and tick[y, left - 3]])
    assert len(xt) == 6 and len(yt) == 9, f"刻度數量不符：{len(xt)} x ticks, {len(yt)} y ticks"
    ax, bx = np.polyfit(xt, [0, 2, 4, 6, 8, 10], 1)
    ay, by = np.polyfit(yt, np.linspace(2.5, -1.5, 9), 1)
    return dict(top=int(top), bottom=int(bottom), left=int(left), right=int(right),
                ax=float(ax), bx=float(bx), ay=float(ay), by=float(by))


def detect_markers(im, color, tol, cal):
    d = np.abs(im - np.array(color)[None, None, :]).sum(axis=2)
    mask = d < tol
    inside = np.zeros_like(mask)
    inside[cal["top"] + 1:cal["bottom"], cal["left"] + 1:cal["right"]] = True
    mask &= inside
    yy, xx = np.mgrid[-3:4, -3:4]
    opened = ndi.binary_opening(mask, structure=(xx ** 2 + yy ** 2) <= 9)
    lab, n = ndi.label(opened)
    blobs = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        x = cal["ax"] * xs.mean() + cal["bx"]
        y = cal["ay"] * ys.mean() + cal["by"]
        if x > 6.5 and y > 1.3:        # 右上角圖例
            continue
        blobs.append({"x": round(float(x), 4), "y": round(float(y), 4), "area_px": int(len(xs)),
                      "n_est": max(1, int(round(len(xs) / SINGLE_MARKER_AREA)))})
    return sorted(blobs, key=lambda b: b["x"])


def main():
    cfg = load_config()
    slide_exp = {e["slide"]: e for e in cfg["experiments"] if e.get("slide")}
    (OUT / "slides").mkdir(parents=True, exist_ok=True)
    prs = Presentation(PPTX)
    ref = {"source": PPTX.name, "note": "軌跡點由圖像自動數位化；速度標籤為人工判讀", "slides": {}}
    for i, pics, text in slide_pictures(prs):
        if i not in slide_exp:
            continue
        exp = slide_exp[i]
        assert len(pics) == 1
        path = OUT / "slides" / f"slide{i:02d}.png"
        path.write_bytes(pics[0].image.blob)
        im = np.array(Image.open(path).convert("RGB")).astype(int)
        cal = calibrate(im)
        g = exp["group"]
        blobs = detect_markers(im, TRAJ_COLOR[g], COLOR_TOL[g], cal)
        entry = {"experiment": exp["id"], "group": g, "image": str(path.relative_to(ROOT)),
                 "image_size": [im.shape[1], im.shape[0]], "m_per_px": round(abs(cal["ax"]), 5),
                 "calibration": cal, "slide_text": text, "markers": blobs,
                 "n_points_detected": int(sum(b["n_est"] for b in blobs))}
        if i in VELOCITY_LABELS:
            entry["velocity_labels"] = VELOCITY_LABELS[i].split()
        ref["slides"][str(i)] = entry
        print(f"slide {i:2d} -> {exp['id']:<22s} {len(blobs):2d} blobs / {entry['n_points_detected']:2d} pts, "
              f"{entry['m_per_px'] * 1000:.1f} mm/px")
    with open(OUT / "slide_reference.json", "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=1)
    print(f"寫入 {OUT / 'slide_reference.json'}")


if __name__ == "__main__":
    main()
