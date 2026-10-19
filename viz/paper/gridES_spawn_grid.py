
from __future__ import annotations

import collections
import pathlib
import re
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from viz._lib.cctv_coverage_panels import C_INK, C_INK2, WALL_T, draw_structure
from viz.paper.wh2x3_example_composite import (
    C_FORKLIFT, FS_LEGEND, _IX0, _IX1, _SY0, _SY1,
    forklift_half_extents, place, rect_local,
)
from isaac_sim.scenes._common import WAREHOUSE_2X3_ZONES
from viz._lib.warehouse_2x3_schematic import C_ROBOT, warp_y

SWEEP = "runs/evaluation/sweeps/warehouse_2x3_gridES_2026-08-17"
OUT_DIR = _REPO_ROOT / "isaac_sim/outputs/gridES_spawn_grid"

C_GOAL = "#7570b3"
FS_TITLE = 8.4
FS_TAG = 6.4


# Grid-cell run dirs only. Hydra drops its own config.yaml into a `.hydra`
# subdir of every shard, which a bare **/config.yaml glob happily returns as a
# 73rd "scene" — match the cell-name pattern instead of trusting the glob.
_CELL_RE = re.compile(r"^warehouse_2x3_gridES_s([LCR])_g([LCR])_cf([A-F]+)$")


def scenes() -> dict[str, dict]:
    """scene name -> {start, heading, goal, blockers}, one entry per grid cell."""
    out: dict[str, dict] = {}
    for cf in sorted((_REPO_ROOT / SWEEP).glob("**/config.yaml")):
        name = cf.parent.name.split("__pose__")[0]
        if not _CELL_RE.match(name):
            continue                       # .hydra and other non-cell dirs
        if name in out:
            continue                       # same cell re-run by another arm
        env = yaml.safe_load(cf.read_text())["env"]
        out[name] = dict(
            start=np.asarray(env["start_xy"], dtype=float),
            heading=float(env["start_heading_deg"]),
            goal=np.asarray(env["goal_xy"], dtype=float),
            blockers=[dict(xy=np.array([o["x"], o["y"]], dtype=float),
                           yaw_deg=float(o.get("yaw_deg", 0.0)),
                           variant=int(o.get("variant", 0)))
                      for o in env.get("obstacles", [])],
        )
    return out


def draw_arrow(ax, xy, heading_deg, length=2.2):
    th = np.radians(heading_deg)
    tip = (xy[0] + length * np.cos(th), xy[1] + length * np.sin(th))
    ax.annotate("", xy=(tip[0], warp_y(tip[1])), xytext=(xy[0], warp_y(xy[1])),
                arrowprops=dict(arrowstyle="-|>", color=C_ROBOT, lw=1.2,
                                shrinkA=1.5, shrinkB=0.0), zorder=7)


def main() -> None:
    sc = scenes()
    starts = {tuple(np.round(v["start"], 3)) for v in sc.values()}
    goals = {tuple(np.round(v["goal"], 3)) for v in sc.values()}
    seals = {(tuple(np.round(b["xy"], 3)), b["yaw_deg"], b["variant"])
             for v in sc.values() for b in v["blockers"]}
    conds = collections.Counter(_CELL_RE.match(n).group(3) for n in sc)
    n_block = collections.Counter(len(v["blockers"]) for v in sc.values())
    assert len(sc) == 72, f"expected 72 grid cells, got {len(sc)}"
    print(f"   conditions: {dict(sorted(conds.items()))}")
    print(f"{len(sc)} grid cells | {len(starts)} spawn x {len(goals)} goal "
          f"positions | {len(seals)} distinct seal placements")
    print(f"   scenes by seal count: {dict(sorted(n_block.items()))}")

    fig = plt.figure(figsize=(3.35, 3.75), facecolor="white")
    ax = fig.add_axes([0.02, 0.185, 0.96, 0.795])
    draw_structure(ax)

    # Every seal position any condition can use — one forklift footprint per
    # aisle. Drawn at the measured footprint (not the setup figures' 1.9x
    # inflation) because whether the body spans the aisle IS the point here.
    for (xy, yaw, variant) in sorted(seals):
        hw, hd = forklift_half_extents(variant)
        fake = dict(blocker=np.asarray(xy), blocker_yaw_deg=yaw)
        ax.add_patch(Polygon(place(rect_local(hw, hd), fake), closed=True,
                             facecolor=C_FORKLIFT, edgecolor="black",
                             linewidth=0.7, zorder=5))
        # Which aisle this seal closes — the conditions (B, C, E, F, BE, ...)
        # are named by these letters, so the figure has to show them.
        zone = next(z for z, v in WAREHOUSE_2X3_ZONES.items()
                    if v.kind == "aisle"
                    and v.x_range[0] <= xy[0] <= v.x_range[1]
                    and v.y_range[0] <= xy[1] <= v.y_range[1])
        ax.text(xy[0], warp_y(xy[1]) + 1.5, zone, ha="center", va="bottom",
                fontsize=FS_TAG, color=C_INK, zorder=8, fontweight="bold")

    # Goals: south strip, three x-positions.
    for g in sorted(goals):
        ax.plot([g[0]], [warp_y(g[1])], marker="*", ms=10.0, color=C_GOAL,
                markeredgecolor="black", markeredgewidth=0.5, zorder=6,
                linestyle="none")

    # Spawns: north strip, three x-positions, all facing south.
    for s in sorted(starts):
        draw_arrow(ax, np.asarray(s), next(iter(sc.values()))["heading"])
        ax.plot([s[0]], [warp_y(s[1])], marker="o", ms=5.0, color=C_ROBOT,
                markeredgecolor="white", markeredgewidth=0.7, zorder=7,
                linestyle="none")

    ax.set_xlim(_IX0 - WALL_T - 0.6, _IX1 + WALL_T + 0.6)
    ax.set_ylim(_SY0 - WALL_T - 0.5, _SY1 + WALL_T + 0.5)

    handles = [
        Line2D([], [], color=C_ROBOT, lw=1.2, marker="o", ms=4.2),
        Line2D([], [], color=C_GOAL, marker="*", ms=7.5, linestyle="none",
               markeredgecolor="black", markeredgewidth=0.5),
        Patch(facecolor=C_FORKLIFT, edgecolor="black", linewidth=0.7),
    ]
    labels = ["spawn pose", "goal position", "forklift seal"]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=1,
               frameon=False, fontsize=FS_LEGEND, labelcolor=C_INK,
               handlelength=1.5, handletextpad=0.5, labelspacing=0.45,
               bbox_to_anchor=(0.5, 0.005))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"gridES_spawn_grid.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    print(f"\nwrote {OUT_DIR}/gridES_spawn_grid.png (+ .pdf)")


if __name__ == "__main__":
    main()
