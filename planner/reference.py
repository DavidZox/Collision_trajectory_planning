"""簡報參考資料（reference/slide_reference.json）的讀取與比對。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "slide_reference.json"


def load_reference(path: Path = REFERENCE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def match(markers: list[dict], Q) -> tuple[float, float]:
    """重現軌跡 Q 與數位化標記的誤差 (最大, 平均) [m]。

    每個色塊對應 n_est 個最近的重現點（重疊標記會合成一個色塊），比較色塊中心與這些點的
    平均位置；沒有被任何色塊認領的重現點，以它到最近色塊的距離計入誤差。
    """
    if Q is None:
        return (0.0, 0.0) if not markers else (np.inf, np.inf)
    Q = np.asarray(Q)
    if not markers:
        return np.inf, np.inf
    errs, claimed = [], set()
    for b in markers:
        c = np.array([b["x"], b["y"]])
        idx = np.argsort(np.linalg.norm(Q - c, axis=1))[:b["n_est"]]
        claimed.update(idx.tolist())
        errs.append(np.linalg.norm(Q[idx].mean(axis=0) - c))
    centers = np.array([[b["x"], b["y"]] for b in markers])
    for i in set(range(len(Q))) - claimed:
        errs.append(np.linalg.norm(centers - Q[i], axis=1).min())
    errs = np.array(errs)
    return float(errs.max()), float(errs.mean())
