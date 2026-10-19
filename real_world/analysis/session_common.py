
from __future__ import annotations

import json
import pathlib
import re

import numpy as np

STOP_OFFSET_M = 0.30      # takes end ~30 cm short of the object
OBJS = ["plant", "fire", "traffic cone"]
NICE = {"plant": "plant", "fire": "fire extinguisher", "traffic cone": "traffic cone"}
COL = {"plant": "#2e7d32", "fire": "#c62828", "traffic cone": "#ef6c00"}
MRK = {"plant": "^", "fire": "s", "traffic cone": "o"}

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_TRAJ = REPO / "real_world/outputs/layout_2026_08_21/trajectories.json"


def load(traj_json: pathlib.Path | str = DEFAULT_TRAJ, stop_offset_m: float = STOP_OFFSET_M):
    """-> (runs, reference start heading in the original odom frame).

    Each run gains X/Y/YAW in the canonical frame plus `obj_xy`, the inferred
    object position (stop pose + stop_offset_m along the final heading).
    """
    data = json.loads(pathlib.Path(traj_json).read_text())
    runs = []
    for r in data["runs"]:
        m = re.match(r"^(\d+)_([abc])_", r["name"])
        if not m or r["prompt"] not in OBJS or r["n_odom"] < 5:
            continue                      # 'walking_*' and the dead 18_c bag land here
        runs.append(dict(name=r["name"], fam=int(m.group(1)), obj=r["prompt"],
                         t0=r["started_unix"], x=np.array(r["x"]),
                         y=np.array(r["y"]), yaw=np.array(r["yaw"])))
    if not runs:
        raise SystemExit(f"no usable takes in {traj_json}")

    h0 = np.array([r["yaw"][0] for r in runs])
    href = float(np.arctan2(np.sin(h0).mean(), np.cos(h0).mean()))
    for r in runs:
        th = np.pi / 2 - r["yaw"][0]              # start heading -> +y
        c, s = np.cos(th), np.sin(th)
        dx, dy = r["x"] - r["x"][0], r["y"] - r["y"][0]
        r["X"], r["Y"] = c * dx - s * dy, s * dx + c * dy
        r["YAW"] = r["yaw"] + th
        r["stop_xy"] = np.array([r["X"][-1], r["Y"][-1]])
        r["obj_xy"] = r["stop_xy"] + stop_offset_m * np.array(
            [np.cos(r["YAW"][-1]), np.sin(r["YAW"][-1])])
        r["path_m"] = float(np.sum(np.hypot(np.diff(r["X"]), np.diff(r["Y"]))))
    runs.sort(key=lambda r: (r["fam"], OBJS.index(r["obj"])))
    return runs, href
