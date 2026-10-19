"""Spatial task: share of rollouts that take the same aisles as A*.

In the 2x3 warehouse every route is characterised by which back aisle (B/C) and
which front aisle (E/F) it passes through. A rollout takes the shortest route if
its aisle set equals that of the A* path, replanned offline for each scene with
the same planner settings that built the evaluation scenes. The robot trace is
resampled to the 0.25 m spacing of the A* path before the aisles are extracted,
so both are counted the same way.

Usage:
    python isaac_sim/analysis/route_optimality.py runs/evaluation/sweeps/<sweep_dir> [...]
"""

import json
import os
import pathlib
import re
import sys
from collections import Counter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omegaconf import OmegaConf  # noqa: E402

from isaac_sim.planning.a_star.offline_gate import path_points  # noqa: E402
from isaac_sim.run_builders.build_2x3 import DataGenerator  # noqa: E402

_ISAAC = pathlib.Path(_REPO_ROOT) / "isaac_sim"
SCENE_DIRS = [_ISAAC / "configs" / "_generated" / "env" / "scene",
              _ISAAC / "configs" / "env" / "scene"]
DATAGEN_CFG = _ISAAC / "configs" / "datagen_2x3.yaml"
SPACING_M = 0.25          # A* path_points default; robot trace resampled to match


def resample_by_arclength(pts, spacing_m=SPACING_M):
    """Thin a dense (x, y) trace to ~uniform `spacing_m` steps along the path.

    Keeps the first point, then emits a point every time cumulative travel since
    the last emitted point exceeds spacing_m.
    e.g. a 30 Hz trace of a 0.9 m/s robot (~3 cm/step) -> every ~8th point.
    """
    if not pts:
        return []
    out = [pts[0]]
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        acc += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if acc >= spacing_m:
            out.append((x1, y1))
            acc = 0.0
    return out


def astar_aisles_for_scene(gen, scene_name):
    """Aisle set of the A* solution for one gridES scene, or None if unavailable.

    Static racks plus the scene's forklifts as rectangles, then A* start -> goal.
    """
    f = next((d / f"{scene_name}.yaml" for d in SCENE_DIRS
              if (d / f"{scene_name}.yaml").exists()), None)
    if f is None:
        return None
    # scene yamls are "# @package env" + a JSON body
    body = "\n".join(l for l in f.read_text().splitlines() if not l.startswith("#"))
    scene = json.loads(body)
    start_xy = tuple(scene["start_xy"])
    goal_xy = tuple(scene["goal_xy"])
    rects = [gen._obstacle_rect(o) for o in scene.get("obstacles", [])]
    pts = path_points(gen.static_rects + rects, start_xy, goal_xy, gen.gate_cfg,
                      spacing_m=SPACING_M)
    if not pts:
        return None
    return set(gen._aisles_on_points(pts))


def robot_aisles(cell_dir, gen):
    """Aisle set the rollout actually threaded (resampled to A* point density)."""
    lp = os.path.join(cell_dir, "log.jsonl")
    if not os.path.exists(lp):
        return None
    pts = []
    for line in open(lp):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "x" in d and "y" in d:
            pts.append((d["x"], d["y"]))
    if not pts:
        return None
    return set(gen._aisles_on_points(resample_by_arclength(pts)))


def analyse(root, gen):
    rows = {}
    for rj in sorted(pathlib.Path(root).glob("warehouse_2x3_*/result.json")):
        cd = rj.parent
        scene = cd.name.split("__pose")[0]          # strip the __pose__..timestamp tail
        r = json.loads(rj.read_text())
        ref = astar_aisles_for_scene(gen, scene)
        got = robot_aisles(str(cd), gen)
        rows[scene] = dict(ref=ref, got=got, outcome=r.get("outcome"),
                           final=r.get("final_dist"))
    return rows


def sealed_of(scene_name):
    """Sealed-aisle set from the scene's trailing `_cf<LETTERS>` token.
    e.g. '..._cfBE' -> {'B','E'};  '..._cfB' -> {'B'};  '..._cf0' -> set()
    """
    m = re.search(r"_cf([A-Z0-9]+)$", scene_name)
    token = m.group(1) if m else "0"
    return set() if token == "0" else set(token)


def report(name, rows):
    n = len(rows)
    usable = {k: v for k, v in rows.items() if v["ref"] and v["got"] is not None}
    match = {k: v for k, v in usable.items() if v["ref"] == v["got"]}
    print(f"\n{'=' * 78}\n{name}   ({n} cells, {len(usable)} with an A* reference)\n{'=' * 78}")
    print(f"took the A*-optimal aisles : {len(match)}/{len(usable)} "
          f"({100 * len(match) / max(1, len(usable)):.0f}%)")

    # A deviation is NOT automatically a failure: taking the OTHER unsealed aisle
    # is a longer-but-legal route (usually still reaches). Only entering a SEALED
    # aisle is a real blockade-reading error. Report the two separately.
    legal = threaded = 0
    for k, v in usable.items():
        if v["ref"] == v["got"]:
            continue
        if v["got"] & sealed_of(k):
            threaded += 1
        else:
            legal += 1
    print(f"suboptimal but LEGAL       : {legal}/{len(usable)}  "
          f"(other unsealed aisle; longer path)")
    print(f"THREADED A SEALED AISLE    : {threaded}/{len(usable)}  "
          f"<-- real blockade-reading failures")

    # reached-vs-route cross-tab: reaching the goal and routing optimally differ
    def reached(v):
        return v["final"] is not None and v["final"] <= 1.0
    tab = Counter()
    for v in usable.values():
        tab[(v["ref"] == v["got"], reached(v))] += 1
    print("\n  route-optimal x reach@1m(final):")
    print(f"    optimal + reached     : {tab[(True, True)]}")
    print(f"    optimal + NOT reached : {tab[(True, False)]}")
    print(f"    OFF-ROUTE + reached   : {tab[(False, True)]}")
    print(f"    OFF-ROUTE + NOT reached: {tab[(False, False)]}")

    off = {k: v for k, v in usable.items() if v["ref"] != v["got"]}
    if off:
        print(f"\n  -- {len(off)} off-route cells --")
        print(f"  {'cell':<34}{'A* aisles':<14}{'robot aisles':<16}{'outcome':<10}{'final'}")
        for k, v in sorted(off.items()):
            a = "".join(sorted(v["ref"])) or "-"
            g = "".join(sorted(v["got"])) or "-"
            fd = f"{v['final']:.2f}" if v["final"] is not None else "?"
            print(f"  {k.replace('warehouse_2x3_gridES_',''):<34}{a:<14}{g:<16}"
                  f"{str(v['outcome']):<10}{fd}")


def main():
    roots = sys.argv[1:]
    if not roots:
        print("usage: isaac_sim/analysis/route_optimality.py <eval_sweep_root> [...]")
        return 1
    gen = DataGenerator(OmegaConf.load(DATAGEN_CFG))
    for r in roots:
        report(os.path.basename(r.rstrip("/")), analyse(r, gen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
