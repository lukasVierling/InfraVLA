
"""Semantic-task success rate and clearance to the shelves.

For every evaluated arm, prints the success rate (closest approach to the goal
within 1 m, the SR reported in the paper) and the mean clearance between the
robot body and the nearest shelf over the rollout.

Usage (after running the semantic evaluations):
    python isaac_sim/analysis/shelf_clearance.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
from isaac_sim.scenes._common import warehouse_2x3_static_rects

OUT_DIR = REPO / "runs/analysis/shelf_clearance"

# Planning inflation radius of the robot, from datagen_2x3_langv2test.yaml
# (robot.inflation_m). Used only to turn a centre distance into a body gap.
ROBOT_RADIUS_M = 0.35

# A rollout is a success if its closest approach to the goal is within this
# radius (not its final distance).
SUCCESS_RADIUS_M = 1.0

# (row label, evaluation output root). The roots are the `recorder.runs_root`
# parents of the matching configs in isaac_sim/configs/experiment/.
ARMS = [
    ("avgpool64 (ours)",     "runs/evaluation/sweeps/langv2test_paper/A0full"),
    ("avgpool16",            "runs/evaluation/sweeps/langv2test_avgpool16_eval"),
    ("shared encoder",       "runs/evaluation/sweeps/langv2test_paper/C1"),
    ("CCTV first",           "runs/evaluation/sweeps/langv2test_paper/B1"),
    ("freeze on-board vis.", "runs/evaluation/sweeps/langv2test_paper/D2"),
    ("freeze vision + LLM",  "runs/evaluation/sweeps/langv2test_frozen_eval"),
    ("no upsampling",        "runs/evaluation/sweeps/langv2test_paper/E1"),
    ("+ stage-2 recovery",   "runs/evaluation/sweeps/langv2_A0_recovft/std"),
    ("ego-only (no CCTV)",   "runs/evaluation/sweeps/langv2_egoonly_baseline/std"),
    # Trained on the held-out split; compare these two rows with each other.
    ("holdout + recovery",   "runs/evaluation/sweeps/langv2test_paper/nfBrec_std"),
    ("holdout, held-out placement", "runs/evaluation/sweeps/langv2test_paper/nfBrec_hfkb"),
]


def point_rect_distances(xy, rects):
    """Distance from each pose to each rectangle; 0 inside.

    xy    : (n, 2) rollout poses, world metres
    rects : (m, 4) rectangles as (xmin, ymin, xmax, ymax)
    -> (n, m) distances

    Standard point-to-AABB form: the gap along each axis is how far the point
    lies OUTSIDE the interval on that axis (0 if within it), and the distance is
    the hypotenuse of the two gaps.
    e.g. point (0, 0), rect (1, 1, 2, 2) -> dx=1, dy=1 -> sqrt(2)=1.414
    """
    px = xy[:, 0][:, None]                      # (n, 1)
    py = xy[:, 1][:, None]                      # (n, 1)
    xmin, ymin, xmax, ymax = (rects[:, i][None, :] for i in range(4))  # (1, m) each

    # How far outside the rect's x-interval the point is (0 when inside it).
    dx = np.maximum(np.maximum(xmin - px, px - xmax), 0.0)   # (n, m)
    dy = np.maximum(np.maximum(ymin - py, py - ymax), 0.0)   # (n, m)
    return np.hypot(dx, dy)                                  # (n, m)


def rollouts(root):
    """scene -> (xy trace (n, 2), closest approach to goal in metres).

    Rollout dirs are named `<scene>__lang__<YYYY-MM-DD_HH-MM-SS>`. If a scene
    was evaluated more than once, the latest rollout is used.
    """
    latest = {}                                    # scene -> (dirname, rollout dir)
    for rj in (REPO / root).glob("**/result.json"):
        scene = rj.parent.name.split("__lang__")[0]
        if not (rj.parent / "log.jsonl").exists():
            continue                               # rollout ended with an exception
        if scene not in latest or rj.parent.name > latest[scene][0]:
            latest[scene] = (rj.parent.name, rj.parent)

    out = {}
    for scene, (_, d) in latest.items():
        recs = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()
                if l.strip()]
        xy = np.asarray([[c["x"], c["y"]] for c in recs if "x" in c], dtype=np.float64)
        goal_d = [c["dist_to_goal"] for c in recs if "dist_to_goal" in c]
        if not len(xy) or not goal_d:
            continue
        # The step that reaches the goal is not logged, so final_dist is folded
        # into the closest approach.
        final_dist = json.loads((d / "result.json").read_text()).get("final_dist")
        closest = min(goal_d)
        if isinstance(final_dist, (int, float)):
            closest = min(closest, float(final_dist))
        out[scene] = (xy, float(closest))
    return out


def main():
    rects = np.asarray(warehouse_2x3_static_rects(), dtype=np.float64)  # (m, 4)
    print(f"{len(rects)} static rack rectangles "
          f"(6 racks tiled into <=4 m segments)\n")

    results = {}
    hdr = (f"{'arm':22s} {'n':>3s} {'SR@1m':>6s} "
           f"{'d_centre':>9s} {'clearance':>10s} {'succ_clear':>11s} "
           f"{'p_incur':>9s} {'dwell%':>7s} {'worst':>7s}")
    print(hdr)
    print("-" * len(hdr))

    for arm, root in ARMS:
        rolls = rollouts(root)
        closest, reached, dwell = [], [], []
        for scene in sorted(rolls):
            xy, goal_closest = rolls[scene]
            # (n_poses, n_rects) -> nearest rack at each pose -> closest approach
            # over the whole rollout. One scalar per rollout, in metres.
            d = point_rect_distances(xy, rects).min(axis=1)   # (n_poses,)
            closest.append(float(d.min()))
            reached.append(bool(goal_closest <= SUCCESS_RADIUS_M))
            # Fraction of this rollout's steps spent inside the rack footprint
            # the planner inflates by ROBOT_RADIUS_M -- i.e. in space A* would
            # have refused to route through. Unlike the closest-approach minimum,
            # this counts HOW LONG the policy stays there, so a policy that
            # grazes once is separated from one that scrapes along a rack face.
            dwell.append(float((d - ROBOT_RADIUS_M < 0.0).mean()))

        closest = np.asarray(closest)
        reached = np.asarray(reached)
        dwell = np.asarray(dwell)
        clearance = closest - ROBOT_RADIUS_M       # body-to-rack gap, metres

        # Successful rollouts only — the non-degenerate version of the metric
        # (a stalled rollout never approaches a rack and would flatter the arm).
        succ_clear = clearance[reached]

        row = dict(
            n=int(len(closest)),
            sr_1m=float(reached.mean()) if len(reached) else float("nan"),
            d_centre_mean=float(closest.mean()),
            clearance_mean=float(clearance.mean()),
            clearance_succ_mean=float(succ_clear.mean()) if len(succ_clear) else float("nan"),
            clearance_succ_n=int(len(succ_clear)),
            # Fraction of rollouts whose body footprint entered the inflated rack
            # footprint at some point, i.e. clearance <= 0.
            p_incursion=float((clearance <= 0.0).mean()),
            # Mean fraction of STEPS spent inside that footprint. This is the
            # column that actually discriminates model quality.
            dwell_frac=float(dwell.mean()),
            worst_clearance=float(clearance.min()),
        )
        results[arm] = row
        print(f"{arm:22s} {row['n']:3d} {row['sr_1m']:6.3f} "
              f"{row['d_centre_mean']:9.2f} {row['clearance_mean']:10.2f} "
              f"{row['clearance_succ_mean']:11.2f} "
              f"{row['p_incursion']:9.3f} {100*row['dwell_frac']:7.2f} "
              f"{row['worst_clearance']:7.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "shelf_clearance.json"
    out_path.write_text(json.dumps(
        {"robot_radius_m": ROBOT_RADIUS_M,
         "success_radius_m": SUCCESS_RADIUS_M,
         "n_rack_rects": int(len(rects)),
         "aisle_clear_width_m": 3.855,
         "centreline_d_centre_m": 3.855 / 2.0,
         "arms": results}, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
