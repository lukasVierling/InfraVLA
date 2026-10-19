
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

_REPO_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from viz._lib.cctv_coverage_panels import C_INK, WALL_T, draw_structure
from viz.paper.wh2x3_example_composite import (
    C_FORKLIFT, FS_LEGEND, _IX0, _IX1, _SY0, _SY1,
    forklift_half_extents, place, rect_local,
)
from viz._lib.warehouse_2x3_schematic import C_ROBOT, warp_y

SWEEP = _REPO_ROOT / "runs/evaluation/sweeps/warehouse_2x3_gridES_2026-07-08"
# Default scene; override with --scene. The companion figure is
# sR_gR_cfF — identical seal and goal, spawn moved to the north-east, where
# BOTH policies route around the seal through E and reach the goal.
# The two panels. Same seal (F) and same goal (R); only the spawn moves.
#   a) spawn north-WEST: the onboard-only policy commits to the sealed aisle.
#   b) spawn north-EAST: both policies route around it and reach the goal.
SCENES = ["warehouse_2x3_gridES_sL_gR_cfF",
          "warehouse_2x3_gridES_sR_gR_cfF"]
OUT_DIR = _REPO_ROOT / "isaac_sim/outputs/gridES_cctv_split"

# (label, arm dir). Order = drawing order; the failing path goes down first so
# the successful one stays legible where they overlap.
ROLLOUTS = [
    ("onboard only", "cfwin_baseline_full"),
    ("with CCTV",    "avgpool_cfwinboost8_step15000"),
]

C_GOAL = "#7570b3"
C_ONBOARD = "#d95f02"       # onboard only — a deep orange, kept clearly
                            # apart from the forklift's golden yellow (#f2b705)
C_CCTV    = "#1a9e5f"       # with CCTV
# Outcome is carried by the GLYPH (checkmark / cross), not by colour, so each
# mark takes its own path's colour. A red "failure" path would editorialise a
# routing choice that is reasonable given what that policy can see.
FS_TAG = 6.4


def load(arm: str, scene: str) -> dict:
    d = next(x for x in (SWEEP / arm).glob("**/*")
             if x.is_dir() and x.name.startswith(scene))
    recs = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if l.strip()]
    recs = [r for r in recs if "x" in r and "dist_to_goal" in r]
    xy = np.asarray([[r["x"], r["y"]] for r in recs], dtype=float)
    goal_d = np.asarray([r["dist_to_goal"] for r in recs], dtype=float)
    fd = json.loads((d / "result.json").read_text()).get("final_dist")
    fd = float(fd) if isinstance(fd, (int, float)) else 9.99

    # The harness keeps logging for a few steps after the robot passes within the
    # success radius, so the raw trace runs on past the goal. Cut a successful
    # path at its closest approach — where the episode naturally terminates.
    if min(fd, goal_d.min()) <= 1.0:
        xy = xy[: int(goal_d.argmin()) + 1]
    env = yaml.safe_load((d / "config.yaml").read_text())["env"]
    return dict(xy=xy, reached=min(fd, goal_d.min()) <= 1.0,
                closest=min(fd, goal_d.min()),
                start=np.asarray(env["start_xy"], dtype=float),
                heading=float(env["start_heading_deg"]),
                goal=np.asarray(env["goal_xy"], dtype=float),
                blockers=[dict(xy=np.array([o["x"], o["y"]], dtype=float),
                               yaw_deg=float(o.get("yaw_deg", 0.0)),
                               variant=int(o.get("variant", 0)))
                          for o in env.get("obstacles", [])])


def draw_panel(ax, rolls, letter: str) -> None:
    ref = rolls[0][1]
    draw_structure(ax)

    # The single seal of this scene, at its measured footprint, turned across F.
    for b in ref["blockers"]:
        hw, hd = forklift_half_extents(b["variant"])
        fake = dict(blocker=b["xy"], blocker_yaw_deg=b["yaw_deg"])
        ax.add_patch(Polygon(place(rect_local(hw, hd), fake), closed=True,
                             facecolor=C_FORKLIFT, edgecolor="black",
                             linewidth=0.7, zorder=5))

    # Above the traces (zorder 7.5/8): a path drawn over the goal marker reads
    # as the robot passing through it rather than arriving at it.
    ax.plot([ref["goal"][0]], [warp_y(ref["goal"][1])], marker="*", ms=10.0,
            color=C_GOAL, markeredgecolor="black", markeredgewidth=0.5,
            zorder=9, linestyle="none")

    # Both solid. The onboard path is drawn THICKER and UNDER the CCTV path, so
    # where the two coincide the reader sees a green core inside a blue casing
    # rather than one path hiding the other.
    for (lab, r), colour, lw, z in zip(rolls, (C_ONBOARD, C_CCTV),
                                       (2.6, 1.5), (7.5, 8.0)):
        xy = r["xy"]
        ax.plot(xy[:, 0], [warp_y(v) for v in xy[:, 1]], color=colour, lw=lw,
                solid_capstyle="round", zorder=z)
        # Outcome mark at the end of the (truncated) path: where it wedged,
        # or where it reached the goal.
        end = xy[-1]
        if r["reached"]:
            ax.plot([end[0] + 1.4], [warp_y(end[1]) + 1.4], marker=r"$\checkmark$",
                    ms=9.0, color=colour, zorder=10, linestyle="none")
        else:
            ax.plot([end[0]], [warp_y(end[1])], marker="X", ms=8.0, color=colour,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=10,
                    linestyle="none")

    # Spawn pose: shared by both rollouts.
    th = np.radians(ref["heading"])
    tip = (ref["start"][0] + 2.2 * np.cos(th), ref["start"][1] + 2.2 * np.sin(th))
    ax.annotate("", xy=(tip[0], warp_y(tip[1])),
                xytext=(ref["start"][0], warp_y(ref["start"][1])),
                arrowprops=dict(arrowstyle="-|>", color=C_ROBOT, lw=1.2,
                                shrinkA=1.5, shrinkB=0.0), zorder=9)
    ax.plot([ref["start"][0]], [warp_y(ref["start"][1])], marker="o", ms=5.0,
            color=C_ROBOT, markeredgecolor="white", markeredgewidth=0.7,
            zorder=9, linestyle="none")

    ax.set_xlim(_IX0 - WALL_T - 0.6, _IX1 + WALL_T + 0.6)
    ax.set_ylim(_SY0 - WALL_T - 0.5, _SY1 + WALL_T + 0.5)
    ax.set_title(f"{letter})", fontsize=8.4, color=C_INK, loc="left", pad=3.0)


def main() -> None:
    argparse.ArgumentParser().parse_args()

    fig = plt.figure(figsize=(5.5, 3.62), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, wspace=0.05,
                  left=0.012, right=0.988, top=0.945, bottom=0.115)
    for i, scene in enumerate(SCENES):
        rolls = [(lab, load(arm, scene)) for lab, arm in ROLLOUTS]
        print(f"{scene}")
        for lab, r in rolls:
            print(f"   {lab:14s} reached={r['reached']}  "
                  f"closest={r['closest']:.2f} m  {len(r['xy'])} steps")
        draw_panel(fig.add_subplot(gs[i]), rolls, "ab"[i])

    handles = [
        Line2D([], [], color=C_ONBOARD, lw=2.6),
        Line2D([], [], color=C_CCTV, lw=1.5),
        Line2D([], [], color=C_ROBOT, lw=1.2, marker="o", ms=4.2),
        Line2D([], [], color=C_GOAL, marker="*", ms=7.5, linestyle="none",
               markeredgecolor="black", markeredgewidth=0.5),
        Patch(facecolor=C_FORKLIFT, edgecolor="black", linewidth=0.7),
    ]
    labels = ["onboard only", "with CCTV", "spawn pose", "goal", "forklift seal"]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=5,
               frameon=False, fontsize=FS_LEGEND, labelcolor=C_INK,
               handlelength=1.5, handletextpad=0.5, labelspacing=0.45,
               columnspacing=1.1, bbox_to_anchor=(0.5, 0.012))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"gridES_cctv_split.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    print(f"\nwrote {OUT_DIR}/gridES_cctv_split.png (+ .pdf)")


if __name__ == "__main__":
    main()
