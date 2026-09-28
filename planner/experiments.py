"""讀取 configs/experiments.yaml、執行實驗，輸出到 runs/<實驗 id>/：

    trajectory.png         與簡報相同樣式的結果圖
    velocity_profile.png   速度與加速度剖面（C、D 組與 S1、S2）
    trajectory.csv         每個軌跡點的座標、走廊或 ESDF、速度等數值
    metrics.json           參數、求解器資訊與各項指標
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

from . import plotting
from .corridor import generate_corridor
from .corridor_qp import solve_trajectory_qp
from .ego_lbfgs import optimize_trajectory
from .esdf import GridESDF
from .metrics import path_metrics, velocity_metrics
from .scenarios import WallChannel, WallPointCloud, clearance, global_path
from .velocity_qp import optimize_velocity_profile_corridor, optimize_velocity_profile_esdf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments.yaml"
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
METHOD = {"A": "DecompROS + OSQP", "D": "DecompROS + OSQP", "B": "ESDF + L-BFGS", "C": "ESDF + L-BFGS"}

log = logging.getLogger("planner")


# ---------------------------------------------------------------- 設定
def load_config(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_dotted(d: dict, key: str, value):
    *parents, last = key.split(".")
    for k in parents:
        d = d.setdefault(k, {})
    d[last] = value


def parse_overrides(items) -> list[tuple[str, object]]:
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set 格式應為 區塊.參數=值，收到：{it}")
        k, v = it.split("=", 1)
        out.append((k.strip(), yaml.safe_load(v)))
    return out


def resolve(exp: dict, cfg: dict, overrides=()) -> dict:
    p = copy.deepcopy(cfg["groups"][exp["group"]]["defaults"])
    for k, v in (exp.get("set") or {}).items():
        _set_dotted(p, k, v)
    for k, v in overrides:
        _set_dotted(p, k, v)
    return p


def find(cfg: dict, exp_id: str) -> dict:
    for e in cfg["experiments"]:
        if e["id"] == exp_id:
            return e
    raise SystemExit(f"找不到實驗 {exp_id}，可用：{[e['id'] for e in cfg['experiments']]}")


# ---------------------------------------------------------------- 建構元件
def build_point_cloud(p: dict) -> WallPointCloud:
    w = p["walls"]
    return WallPointCloud(bottom=tuple(w["bottom"]), top=tuple(w["top"]), x_range=tuple(w["x_range"]),
                          n_x=w["n_x"], n_bottom=w["n_bottom"], n_top=w["n_top"])


def build_esdf(p: dict) -> GridESDF:
    m, w = p["map"], p["walls"]
    return GridESDF(WallChannel(tuple(w["x_range"]), tuple(w["channel"])), m["x_min"], m["x_max"],
                    m["y_min"], m["y_max"], m["resolution"], m["lookup"])


def plan_corridor(p: dict):
    walls = build_point_cloud(p)
    obs = walls.points()
    ref = global_path(p["N"])
    c = p["corridor"]
    boxes = generate_corridor(ref, obs, c["margin"], tuple(c["default_bounds"]), c["search_half_width"])
    res = solve_trajectory_qp(ref, boxes, p["qp"]["w_smooth"], p["qp"]["w_ref"],
                              p["qp"].get("smooth_p_scale", 1.0), p.get("osqp"))
    return walls, obs, ref, boxes, res


def plan_esdf(p: dict):
    esdf_map = build_esdf(p)
    ref = global_path(p["N"])
    e = p["ego"]
    res = optimize_trajectory(ref, esdf_map, e["d_safe"], e["w_smooth"], e["w_obs"], e["w_ref"], p.get("lbfgs"))
    return esdf_map, ref, res


def _velocity_kwargs(v: dict, p: dict) -> dict:
    kw = {k: v[k] for k in ("w_v", "w_a", "enforce_a_max")}
    kw["osqp_settings"] = p.get("osqp")
    return kw


# ---------------------------------------------------------------- 執行
def run_experiment(exp: dict, cfg: dict, overrides=(), out_name: str | None = None) -> dict:
    p = resolve(exp, cfg, overrides)
    group = exp["group"]
    gtype = cfg["groups"][group]["type"]
    title = cfg["groups"][group]["title"]
    if exp.get("suggested"):
        title += f"\n[suggested] {exp['id']}"
    if overrides:
        title += "\n[custom] " + ", ".join(f"{k}={v}" for k, v in overrides)
    out = RUNS / (out_name or exp["id"])
    out.mkdir(parents=True, exist_ok=True)
    figsize = tuple(p.get("plot", {}).get("figsize", (10, 5)))

    custom = bool(overrides) or (out_name is not None and out_name != exp["id"])
    record = {"id": out_name or exp["id"], "base_experiment": exp["id"], "group": group,
              "method": METHOD[group], "type": gtype, "slide": None if custom else exp.get("slide"),
              "suggested": bool(exp.get("suggested", False)), "custom": custom, "desc": exp.get("desc", ""),
              "overrides": {k: v for k, v in overrides}, "params": p}
    rows = []
    vres = None

    if gtype in ("corridor_qp", "corridor_velocity"):
        walls, obs, ref, boxes, res = plan_corridor(p)
        rects = walls.rects()
        record["solver"] = {"name": "OSQP", "status": res.status, "iterations": res.iterations,
                            "run_time_ms": res.run_time_ms, "wall_time_ms": res.wall_time_ms,
                            "objective": res.objective, **res.info}
        Q = res.Q
        if Q is not None and gtype == "corridor_velocity":
            v = p["velocity"]
            vres = optimize_velocity_profile_corridor(Q, boxes, v["v_target"], v["a_max"], v["a_lat_max"],
                                                      v["v_corridor_limit"], v["narrow_width"], **_velocity_kwargs(v, p))
        fig = plotting.plot_corridor_scene(obs, ref, boxes, Q, title, None if vres is None else vres.v,
                                           res.status, figsize, p.get("plot", {}).get("path_alpha", 0.6))
        for i in range(len(ref)):
            row = {"i": i, "x_ref": ref[i, 0], "y_ref": ref[i, 1],
                   "x": Q[i, 0] if Q is not None else "", "y": Q[i, 1] if Q is not None else "",
                   "corridor_y_min": boxes[i, 1], "corridor_y_max": boxes[i, 2],
                   "corridor_width": boxes[i, 2] - boxes[i, 1]}
            rows.append(row)
    else:
        esdf_map, ref, res = plan_esdf(p)
        rects = esdf_map.rects()
        Q = res.Q
        record["solver"] = {"name": "scipy L-BFGS-B", "status": res.message, "converged_normally": res.converged_normally,
                            "iterations": res.nit, "function_evals": res.nfev, "wall_time_ms": res.wall_time_ms,
                            "objective": res.cost, **res.terms}
        record["esdf_grid"] = {"nx": esdf_map.nx, "ny": esdf_map.ny, "min": float(esdf_map.esdf.min()),
                               "max": float(esdf_map.esdf.max()), "lookup": esdf_map.lookup}
        if gtype == "esdf_velocity":
            v = p["velocity"]
            vres = optimize_velocity_profile_esdf(Q, esdf_map, v["v_target"], v["a_max"], v["a_lat_max"],
                                                  v["d_safety_margin"], v["v_obstacle_limit"], **_velocity_kwargs(v, p))
        fig = plotting.plot_esdf_scene(esdf_map, ref, Q, title, None if vres is None else vres.v, figsize)
        dist, _ = esdf_map.query(Q)
        for i in range(len(ref)):
            rows.append({"i": i, "x_ref": ref[i, 0], "y_ref": ref[i, 1], "x": Q[i, 0], "y": Q[i, 1],
                         "esdf": dist[i]})
    plotting.save(fig, out / "trajectory.png")

    if Q is not None:
        cl = clearance(Q, rects)
        for row, c in zip(rows, cl):
            row["clearance_m"] = c
        record["metrics"] = path_metrics(Q, ref, rects)
    else:
        record["metrics"] = {}
    if vres is not None:
        a_max = p["velocity"]["a_max"]
        record["metrics"].update(velocity_metrics(vres, a_max))
        record["velocity_labels"] = [f"{x:.2f}" for x in vres.v]
        s = np.concatenate([[0.0], np.cumsum(vres.ds)])
        for i, row in enumerate(rows):
            row.update({"s": s[i], "kappa": vres.kappa[i], "v_max": vres.v_max[i], "v": vres.v[i],
                        "t": vres.t[i], "a_to_next": vres.a[i] if i < len(vres.a) else ""})
        vtitle = f"{record['id']}  (w_a={p['velocity']['w_a']:g}" + \
                 (", |a| <= a_max enforced)" if p["velocity"]["enforce_a_max"] else ")")
        plotting.save(plotting.plot_velocity_profile(vres, vtitle, a_max), out / "velocity_profile.png")
    record["trajectory"] = None if Q is None else Q.round(6).tolist()

    with open(out / "trajectory.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for row in rows:
            wr.writerow({k: (f"{v:.6f}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=_json_default)

    m = record["metrics"]
    msg = f"{record['id']:<28s} {record['solver']['status'][:34]:<34s} it={record['solver']['iterations']:<4d}"
    if m:
        msg += f" len={m['path_length_m']:.3f} m  min_clear={m['min_clearance_polyline_m']:+.3f} m"
    if vres is not None:
        msg += f"  T={vres.travel_time:.2f} s  v={' '.join(record['velocity_labels'])}"
    log.info(msg)
    return record


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------- CLI
def setup_logging(name: str):
    LOGS.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / f"{name}.log", "w", "utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


def main(groups: list[str], suggested: bool = False, log_name: str = "run"):
    ap = argparse.ArgumentParser(description=f"執行 {'/'.join(groups)} 組實驗")
    ap.add_argument("--exp", nargs="+", help="只跑指定的實驗 id（預設：該組所有簡報實驗）")
    ap.add_argument("--set", action="extend", nargs="+", metavar="區塊.參數=值", help="臨時覆寫參數，可一次給多個")
    ap.add_argument("--name", help="輸出資料夾名稱（runs/<name>），只能搭配單一 --exp")
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    overrides = parse_overrides(args.set)
    if args.name and (not args.exp or len(args.exp) != 1):
        raise SystemExit("--name 需要搭配單一 --exp")
    # 有 --set 但沒給 --name 時自動命名，避免覆蓋簡報實驗的輸出與 log（重現比對會用到）
    tag = "_".join(f"{k.split('.')[-1]}={v}" for k, v in overrides).replace(" ", "").replace("/", "-")
    if args.name:
        setup_logging(f"{log_name}__{args.name}")
    else:
        setup_logging(f"{log_name}__{tag}" if overrides else log_name)
    if args.exp:
        exps = [find(cfg, e) for e in args.exp]
    else:
        exps = [e for e in cfg["experiments"]
                if e["group"] in groups and bool(e.get("suggested", False)) == suggested]
    log.info(f"config={args.config}  overrides={overrides}  experiments={[e['id'] for e in exps]}")
    records = [run_experiment(e, cfg, overrides, args.name or (f"{e['id']}__{tag}" if overrides else None))
               for e in exps]
    log.info(f"完成 {len(records)} 個實驗，輸出於 {RUNS}")
    return records
