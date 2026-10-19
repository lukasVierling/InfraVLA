
from __future__ import annotations

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

from viz._lib.langv2_shelf_clearance import (
    ROBOT_RADIUS_M, point_rect_distances,
)
from isaac_sim.scenes._common import warehouse_2x3_static_rects
from viz._lib.cctv_coverage_panels import (
    C_INK, C_INK2, WALL_T, draw_structure,
)
from viz.paper.wh2x3_example_composite import (
    C_FORKLIFT, FS_LEGEND, FS_MINI, _IX0, _IX1, _SY0, _SY1,
    forklift_half_extents, place, rect_local,
)
from viz.paper.wh2x3_lang_composite import OBJ_STYLE
from viz._lib.warehouse_2x3_schematic import warp_y

SCENE = "warehouse_2x3_langv2_0508_tA"
OUT_DIR = _REPO_ROOT / "isaac_sim/outputs/langv2_derail_triptych"

# (panel title, sweep root). Order = the argument: broken -> ours -> repaired.
# (panel title, sweep root). Titles are two lines on purpose: line 1 is WHICH
# model, line 2 is WHICH TRAINING STAGE, so the stage row reads straight across
# the figure. Panel a is a stage-1 model too -- saying so makes the two
# comparisons explicit: a vs b isolates the architecture, b vs c the data.
PANELS = [
    ("Frozen on-board encoder\nstage 1",
     "runs/evaluation/sweeps/langv2test_paper/D2"),
    ("Ours\nstage 1 (cf. upsampling)",
     "runs/evaluation/sweeps/langv2test_paper/A0full"),
    ("Ours\nstage 2 ($+$ recovery data)",
     "runs/evaluation/sweeps/langv2_A0_recovft/std"),
]

C_TRACE = "#1b1b1b"        # driven path
C_HIT   = "#d1495b"        # rack strike
C_OK    = "#2a9d8f"        # target reached

FS_TITLE = 8.4              # panel titles; FS_MINI (6.6) is too small here

RACKS = np.asarray(warehouse_2x3_static_rects(), dtype=float)


def find_rollout(root: str, scene: str) -> pathlib.Path:
    """The rollout dir for `scene` under `root` (latest timestamp if re-run)."""
    hits = [p.parent for p in (_REPO_ROOT / root).glob("**/result.json")
            if p.parent.name.split("__lang__")[0] == scene]
    if not hits:
        raise FileNotFoundError(f"{scene} not found under {root}")
    return sorted(hits, key=lambda p: p.name)[-1]


def load_rollout(d: pathlib.Path) -> dict:
    """Trace, scene geometry and the derived clearance for one rollout."""
    recs = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if l.strip()]
    xy = np.asarray([[r["x"], r["y"]] for r in recs if "x" in r], dtype=float)

    cfg = yaml.safe_load((d / "config.yaml").read_text())
    env = cfg["env"]
    ep = dict(
        spawn=np.asarray(env["start_xy"], dtype=float),
        spawn_heading_deg=float(env["start_heading_deg"]),
        objects=[dict(kind=o["kind"], variant=int(o.get("variant", 0)),
                      xy=np.array([o["x"], o["y"]], dtype=float),
                      yaw_deg=float(o.get("yaw_deg", 0.0)),
                      role=o.get("role", ""))
                 for o in env.get("obstacles", [])],
    )

    # Per-pose distance to the nearest rack, then the closest approach and where
    # it happened. Clearance subtracts the robot's planning radius, so <= 0 means
    # the body footprint overlapped the rack the planner treats as solid.
    d_pose = point_rect_distances(xy, RACKS).min(axis=1)     # (n_poses,)
    i = int(d_pose.argmin())
    res = json.loads((d / "result.json").read_text())

    # Reached = closest approach to the goal <= 1 m, the same criterion as the
    # SR column of the results table (NOT final distance).
    fd = res.get("final_dist")
    fd = float(fd) if isinstance(fd, (int, float)) else 9.99
    goal_d = [r["dist_to_goal"] for r in recs if "dist_to_goal" in r]
    reached = min([fd] + goal_d) <= 1.0

    return dict(xy=xy, ep=ep, instruction=(cfg.get("task") or {}).get("lang_instruction", ""),
                clearance=float(d_pose[i] - ROBOT_RADIUS_M), hit_xy=xy[i],
                reached=bool(reached), outcome=res.get("outcome"), final_dist=fd)


def draw_objects_true(ax, ep) -> None:
    """Objects by kind, forklift at its MEASURED footprint.

    The shared `wh2x3_lang_composite.draw_objects` inflates the forklift by
    FORKLIFT_THICKEN = 1.9 for legibility at 2.05 in subfigure width. At this
    figure's panel size that exaggeration is unnecessary and it pushes the
    drawn body 0.25 m THROUGH the west rack face (true footprint spans
    x -5.86..-4.64; thickened it spans -6.40..-4.10 against a rack face at
    -6.15). Since this figure is an argument about clearance, an object drawn
    overlapping a rack is exactly the wrong artefact — so draw it true.
    """
    for obj in ep["objects"]:
        if obj["kind"] == "forklift":
            half_w, half_d = forklift_half_extents(obj["variant"])
            fake = dict(blocker=obj["xy"], blocker_yaw_deg=obj["yaw_deg"])
            ax.add_patch(Polygon(place(rect_local(half_w, half_d), fake),
                                 closed=True, facecolor=C_FORKLIFT,
                                 edgecolor="black", linewidth=0.7, zorder=5))
        else:
            st = OBJ_STYLE[obj["kind"]]
            ax.plot([obj["xy"][0]], [warp_y(obj["xy"][1])], marker=st["marker"],
                    ms=st["ms"], color=st["colour"], markeredgecolor="black",
                    markeredgewidth=0.6, zorder=5, linestyle="none")


def draw_panel(ax, roll: dict, title: str, letter: str) -> None:
    draw_structure(ax)
    draw_objects_true(ax, roll["ep"])

    # No in-panel instruction label: the prompt belongs in the LaTeX caption.
    # `tgt` is still needed to position the "reached" tick.
    tgt = next(o for o in roll["ep"]["objects"] if o["role"] == "target")

    xy = roll["xy"]
    ax.plot(xy[:, 0], [warp_y(v) for v in xy[:, 1]],
            color=C_TRACE, lw=1.6, solid_capstyle="round", zorder=8)

    # Two marks only: where the body entered a rack, and whether the named
    # object was reached. No per-panel clearance readout — that lives in the
    # results table; this figure is about the shape of the path.
    if roll["clearance"] <= 0.0:
        ax.plot([roll["hit_xy"][0]], [warp_y(roll["hit_xy"][1])], marker="X",
                ms=8.0, color=C_HIT, markeredgecolor="white",
                markeredgewidth=0.9, zorder=9, linestyle="none")
    if roll["reached"]:
        ax.plot([tgt["xy"][0] + 1.05], [warp_y(tgt["xy"][1]) + 0.35],
                marker=r"$\checkmark$", ms=9.0, color=C_OK, zorder=10,
                linestyle="none")


    ax.set_xlim(_IX0 - WALL_T - 0.6, _IX1 + WALL_T + 0.6)
    ax.set_ylim(_SY0 - WALL_T - 0.5, _SY1 + WALL_T + 0.5)
    ax.set_title(f"{letter}) {title}", fontsize=FS_TITLE, color=C_INK, pad=3.5)


def main() -> None:
    rolls = []
    for title, root in PANELS:
        d = find_rollout(root, SCENE)
        r = load_rollout(d)
        rolls.append((title, r))
        print(f"{title:28s} {d.name}\n"
              f"{'':28s}   clearance {r['clearance']:+.2f} m  "
              f"outcome={r['outcome']}  final_dist={r['final_dist']:.2f}")

    fig = plt.figure(figsize=(6.9, 3.62), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.06,
                  left=0.012, right=0.988, top=0.885, bottom=0.155)
    for i, (title, r) in enumerate(rolls):
        draw_panel(fig.add_subplot(gs[i]), r, title, "abc"[i])

    handles = [
        Line2D([], [], color=C_TRACE, lw=1.6),
        Line2D([], [], color=C_HIT, marker="X", ms=6.5, linestyle="none",
               markeredgecolor="white", markeredgewidth=0.8),
        Line2D([], [], color=C_OK, marker=r"$\checkmark$", ms=8.0,
               linestyle="none"),
        Line2D([], [], color=OBJ_STYLE["barrel"]["colour"], marker="o", ms=5.0,
               linestyle="none", markeredgecolor="black", markeredgewidth=0.6),
        Line2D([], [], color=OBJ_STYLE["traffic_cone"]["colour"], marker="o", ms=5.0,
               linestyle="none", markeredgecolor="black", markeredgewidth=0.6),
        Line2D([], [], color=OBJ_STYLE["box"]["colour"], marker="s", ms=4.8,
               linestyle="none", markeredgecolor="black", markeredgewidth=0.6),
        Patch(facecolor=C_FORKLIFT, edgecolor="black", linewidth=0.7),
    ]
    labels = ["driven path", "drove into rack", "object reached",
              "barrel", "traffic cone", "box", "forklift"]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=4,
               frameon=False, fontsize=FS_LEGEND, labelcolor=C_INK,
               handlelength=1.5, handletextpad=0.45, labelspacing=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, 0.004))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"langv2_derail_triptych.{ext}", dpi=400,
                    facecolor="white")
    plt.close(fig)
    print(f"\nwrote {OUT_DIR}/langv2_derail_triptych.png (+ .pdf)")


if __name__ == "__main__":
    main()
