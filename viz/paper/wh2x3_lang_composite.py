
from __future__ import annotations

import json
import sys
import pathlib
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch, Polygon
from omegaconf import OmegaConf

from isaac_sim.scenes._common import WAREHOUSE_2X3_NAV_X, WAREHOUSE_2X3_ZONES
from viz._lib.cctv_coverage_tikz import (
    CAM_COLOURS,
    CAM_NAMES,
    GRID_RES,
    OWNED_ZONE,
    RACK_RECTS,
    compound_path,
    frame_rings,
    load_bigcam,
    mask_to_rings,
    visibility_mask,
)
from viz._lib.cctv_coverage_panels import (
    place_legend_below,
    place_legends_below,
    CAM_FILL_ALPHA,
    C_INK,
    C_INK2,
    WALL_T,
    WARP_RDP_TOL,
    draw_camera_mark,
    draw_structure,
)
from viz.paper.wh2x3_example_composite import (
    C_FORKLIFT,
    FORKLIFT_THICKEN,
    FS_LEGEND,
    FS_MINI,
    FS_RAIL,
    FS_ZONE,
    SUB_W,
    _IX0,
    _IX1,
    _IY0,
    _IY1,
    _SY0,
    _SY1,
    draw_rails,
    forklift_half_extents,
    place,
    rect_local,
)
from viz._lib.warehouse_2x3_schematic import (
    C_ROBOT,
    EDGE_LW,
    FILL_ALPHA,
    warp_y,
)

EXAMPLE_RUN = (_REPO_ROOT / "runs" / "data_gen" / "warehouse_2x3_langv2_2026-07-22"
               / "langv2_p1"
               / "warehouse_2x3_langv2_0000_tA__pose__2026-07-22_00-54-50")
LANG_CFG = _REPO_ROOT / "isaac_sim" / "configs" / "datagen_2x3_langv2.yaml"
MANIFEST = (_REPO_ROOT / "isaac_sim" / "configs" / "experiment"
            / "datagen_2x3_langv2_all_manifest.json")
FRAME_IDX = 0

C_OBJZONE = "#7570b3"
OBJ_STYLE = {
    "barrel":       dict(colour="#2a78d6", marker="o", ms=6.0),
    "traffic_cone": dict(colour="#eb6834", marker="o", ms=6.0),
    "box":          dict(colour="#8c5a2b", marker="s", ms=5.6),
}
OBJ_LABEL = {"barrel": "barrel", "traffic_cone": "traffic cone",
             "box": "box", "forklift": "forklift"}

OUT_DIR = _REPO_ROOT / "isaac_sim" / "outputs" / "wh2x3_lang_composite"

_NX0, _NX1 = WAREHOUSE_2X3_NAV_X


def object_span(zone_name: str, cfg) -> tuple[tuple[float, float], tuple[float, float]]:
    """Where an object CENTRE may land in `zone_name` — the exact generator rule.

    y: aisle inset by aisle_end_margin_m at both ends, then clipped to the half
    FARTHER from the spawn zone D ('outer_half').
    x: the aisle inset by the object's own cross-aisle half + clearance_margin_m.
       That inset is per-kind, so the span drawn here uses the SMALLEST kind
       (the traffic cone) — i.e. the union over kinds of admissible centres.
    """
    z = WAREHOUSE_2X3_ZONES[zone_name]
    m = float(cfg.placement.aisle_end_margin_m)
    y_lo, y_hi = z.y_range[0] + m, z.y_range[1] - m
    y_mid = 0.5 * (z.y_range[0] + z.y_range[1])
    d_center_y = WAREHOUSE_2X3_ZONES[str(cfg.spawn_zone)].center[1]
    if z.center[1] > d_center_y:
        y_lo = max(y_lo, y_mid)
    else:
        y_hi = min(y_hi, y_mid)
    cone = cfg.footprint_overrides.traffic_cone["0"]
    need = float(cone.half_w) + float(cfg.placement.clearance_margin_m)
    return (z.x_range[0] + need, z.x_range[1] - need), (y_lo, y_hi)


def load_episode() -> dict:
    """Object poses, spawn pose and the instruction of the worked family."""
    scene = yaml.safe_load((EXAMPLE_RUN / "config.yaml").read_text())["env"]
    scene_name = EXAMPLE_RUN.name.split("__pose__")[0]
    entries = json.loads(MANIFEST.read_text())
    entries = entries["scenes"] if isinstance(entries, dict) else entries
    meta = next(e for e in entries if e["name"] == scene_name)
    return dict(
        name=scene_name,
        spawn=np.asarray(scene["start_xy"], dtype=float),
        spawn_heading_deg=float(scene["start_heading_deg"]),
        objects=[dict(kind=o["kind"], variant=int(o.get("variant", 0)),
                      xy=np.array([o["x"], o["y"]], dtype=float),
                      yaw_deg=float(o.get("yaw_deg", 0.0)),
                      role=o.get("role", ""))
                 for o in scene["obstacles"]],
        zone_objects=meta["zone_objects"],
        instruction=meta["instruction"],
        target_kind=meta["target_kind"],
        goal_zone=meta["goal_zone"],
    )


def soft_region(ax, x_range, y_range, colour: str) -> None:
    """Rounded soft fill + hairline full-hue edge (the placement-figure idiom)."""
    x0, x1 = x_range
    y0, y1 = (warp_y(v) for v in y_range)
    for face, edge, lw in ((colour, "none", 0.0), ("none", colour, EDGE_LW)):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor=face, alpha=FILL_ALPHA if face != "none" else 1.0,
            edgecolor=edge, linewidth=lw, zorder=2 if face != "none" else 2.5))


def draw_spawn_pose(ax, ep, ms: float = 5.0) -> None:
    """The robot's single deterministic spawn pose: position + heading arrow."""
    sx, sy = ep["spawn"][0], warp_y(ep["spawn"][1])
    th = np.radians(ep["spawn_heading_deg"])
    tip = np.array([ep["spawn"][0] + 1.7 * np.cos(th),
                    ep["spawn"][1] + 1.7 * np.sin(th)])
    ax.annotate("", xy=(tip[0], warp_y(tip[1])), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=C_ROBOT, lw=1.2,
                                shrinkA=1.5, shrinkB=0.0), zorder=6)
    ax.plot([sx], [sy], marker="o", ms=ms, color=C_ROBOT,
            markeredgecolor="white", markeredgewidth=0.7, zorder=7)


def draw_objects(ax, ep) -> None:
    """All four placed objects, by kind. Small kinds are SYMBOLS (fixed screen
    size, immune to the y-warp); the forklift keeps its measured footprint."""
    for obj in ep["objects"]:
        kind = obj["kind"]
        if kind == "forklift":
            half_w, half_d = forklift_half_extents(obj["variant"])
            fake = dict(blocker=obj["xy"], blocker_yaw_deg=obj["yaw_deg"])
            ax.add_patch(Polygon(
                place(rect_local(half_w * FORKLIFT_THICKEN, half_d), fake),
                closed=True, facecolor=C_FORKLIFT, edgecolor="black",
                linewidth=0.7, zorder=5))
        else:
            st = OBJ_STYLE[kind]
            ax.plot([obj["xy"][0]], [warp_y(obj["xy"][1])], marker=st["marker"],
                    ms=st["ms"], color=st["colour"], markeredgecolor="black",
                    markeredgewidth=0.6, zorder=5, linestyle="none")


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[out]      {OUT_DIR}/{stem}.png (+ .pdf)")


def render_sub_lang_zones(ep, cfg) -> None:
    # Axes sized in INCHES (map as large as SUB_W allows); the figure is created
    # tall and place_legend_below() trims the slack under the legend.
    fig = plt.figure(figsize=(SUB_W, 5.2), facecolor="white")
    ax = fig.add_axes([0.01, 0.99 - 3.17 / 5.2, 0.98, 3.17 / 5.2])
    draw_structure(ax)
    for zname in ("B", "C", "E", "F"):
        soft_region(ax, *object_span(zname, cfg), C_OBJZONE)
    for zname, zone in WAREHOUSE_2X3_ZONES.items():
        cx = 0.5 * (zone.x_range[0] + zone.x_range[1])
        if zname in ("B", "C", "E", "F"):
            # Letter in the INNER half (toward D) — the half `outer_half`
            # placement never uses, so it cannot collide with the span.
            _, (sy0, sy1) = object_span(zname, cfg)
            inner = zone.y_range[0] if sy0 > zone.y_range[0] else zone.y_range[1]
            mid = 0.5 * (zone.y_range[0] + zone.y_range[1])
            cy = 0.5 * (warp_y(inner) + warp_y(mid))
        else:
            cy = 0.5 * (warp_y(zone.y_range[0]) + warp_y(zone.y_range[1]))
        ax.text(cx, cy, zname, fontsize=FS_ZONE, color=C_INK2, ha="center",
                va="center", zorder=6,
                path_effects=[pe.Stroke(linewidth=1.8, foreground="white"),
                              pe.Normal()])
    draw_spawn_pose(ax, ep)
    draw_rails(ax)
    leg = fig.legend(handles=[
        Patch(facecolor=C_OBJZONE, alpha=FILL_ALPHA, edgecolor=C_OBJZONE,
              label="object spawn span"),
        Line2D([], [], color=C_ROBOT, lw=1.2, marker="o", ms=4.5,
               label="robot spawn pose"),
        Patch(facecolor="#41474f", edgecolor="none", label="rack"),
    ], loc="lower center", ncol=1, frameon=False, fontsize=FS_LEGEND,
        labelcolor=C_INK, handlelength=1.4, handletextpad=0.5,
        labelspacing=0.4, bbox_to_anchor=(0.5, 0.0))
    place_legend_below(fig, leg)
    _save(fig, "sub_lang_zones")


def render_sub_lang_objects(rings_per_cam, cams, ep) -> None:
    # Six legend entries in ONE column: at 0.32\textwidth two columns are ~1 in
    # each and clip the long labels. Taller figure is fine — the subfigures are
    # [t]-aligned in LaTeX, so they need not share a height.
    fig = plt.figure(figsize=(SUB_W, 5.2), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.26,
                  left=0.01, right=0.99, top=0.95, bottom=0.95 - 3.375 / 5.2)
    for i, name in enumerate(CAM_NAMES):
        ax = fig.add_subplot(gs[i])
        draw_structure(ax)
        colour = CAM_COLOURS[name]
        cov = compound_path(rings_per_cam[name])
        ax.add_patch(PathPatch(cov, facecolor=colour, alpha=CAM_FILL_ALPHA,
                               edgecolor="none", zorder=2))
        ax.add_patch(PathPatch(cov, facecolor="none", edgecolor=colour,
                               lw=0.7, zorder=2.5))
        draw_camera_mark(ax, cams[name], colour)
        draw_objects(ax, ep)
        draw_spawn_pose(ax, ep, ms=4.0)
        ax.set_xlim(_IX0 - WALL_T - 0.6, _IX1 + WALL_T + 0.6)
        ax.set_ylim(_SY0 - WALL_T - 0.5, _SY1 + WALL_T + 0.5)
        ax.set_title(f"{name}$\\,\\to\\,${OWNED_ZONE[name]}", fontsize=FS_MINI,
                     color=C_INK, pad=2.5)

    obj_handles = [
        Line2D([], [], color=OBJ_STYLE["barrel"]["colour"], marker="o", ms=5.0,
               linestyle="none", markeredgecolor="black", markeredgewidth=0.6),
        Line2D([], [], color=OBJ_STYLE["traffic_cone"]["colour"], marker="o",
               ms=5.0, linestyle="none", markeredgecolor="black",
               markeredgewidth=0.6),
        Line2D([], [], color=OBJ_STYLE["box"]["colour"], marker="s", ms=4.8,
               linestyle="none", markeredgecolor="black", markeredgewidth=0.6),
        Patch(facecolor=C_FORKLIFT, edgecolor="black", linewidth=0.7),
    ]
    swatch = tuple(Patch(facecolor=CAM_COLOURS[n], alpha=CAM_FILL_ALPHA,
                         edgecolor=CAM_COLOURS[n], linewidth=0.7)
                   for n in CAM_NAMES)
    # Two stacked legend blocks: the four object kinds as a 2x2 (they are peers,
    # and one column of six entries made the figure needlessly tall), then the
    # two full-width entries under it.
    leg_objects = fig.legend(
        handles=obj_handles,
        labels=["barrel", "traffic cone", "box", "forklift"],
        loc="lower left", ncol=2, frameon=False, fontsize=FS_LEGEND,
        labelcolor=C_INK, handlelength=1.6, handletextpad=0.5,
        labelspacing=0.4, columnspacing=1.1, bbox_to_anchor=(0.0, 0.0))
    leg_rest = fig.legend(
        handles=[swatch,
                 Line2D([], [], color=C_ROBOT, lw=1.2, marker="o", ms=4.0)],
        labels=["floor observed by the camera", "robot spawn pose"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="lower left", ncol=1, frameon=False, fontsize=FS_LEGEND,
        labelcolor=C_INK, handlelength=1.6, handletextpad=0.5,
        labelspacing=0.4, bbox_to_anchor=(0.0, 0.0))
    place_legends_below(fig, [leg_rest, leg_objects], align="left")
    _save(fig, "sub_lang_objects")


def render_sub_lang_cctv() -> None:
    fig = plt.figure(figsize=(SUB_W, 2.45), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, wspace=0.07, hspace=0.22,
                  left=0.015, right=0.985, top=0.925, bottom=0.015)
    for i, name in enumerate(CAM_NAMES):
        img = EXAMPLE_RUN / f"cctv_{name}" / f"{FRAME_IDX:03d}.png"
        assert img.exists(), f"{img} missing"
        ax = fig.add_subplot(gs[i])
        ax.imshow(mpimg.imread(img))
        ax.set_xticks([])
        ax.set_yticks([])
        # Border in the camera's colour — the same hue its coverage wears in
        # (b), so a reader can pair each view with its footprint.
        for spine in ax.spines.values():
            spine.set_edgecolor(CAM_COLOURS[name])
            spine.set_linewidth(1.8)
        ax.set_title(name, fontsize=FS_MINI, color=C_INK, pad=2.5)
    _save(fig, "sub_lang_cctv")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert EXAMPLE_RUN.exists(), f"episode not found: {EXAMPLE_RUN}"
    cfg = OmegaConf.load(LANG_CFG)
    ep = load_episode()
    cams = {name: load_bigcam(name) for name in CAM_NAMES}

    xs = np.arange(_IX0 + GRID_RES / 2, _IX1, GRID_RES)
    ys = np.arange(_IY0 + GRID_RES / 2, _IY1, GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    masks = {n: visibility_mask(cams[n], X, Y) for n in CAM_NAMES}
    rings = {n: frame_rings(mask_to_rings(masks[n], xs, ys), "warped", WARP_RDP_TOL)
             for n in CAM_NAMES}

    print(f"[episode]  {ep['name']}   instruction: {ep['instruction']!r}")
    print(f"[episode]  spawn ({ep['spawn'][0]:.2f}, {ep['spawn'][1]:.2f}) "
          f"heading {ep['spawn_heading_deg']:.0f} deg")
    for zname in ("B", "C", "E", "F"):
        (x0, x1), (y0, y1) = object_span(zname, cfg)
        obj = next(o for o in ep["objects"]
                   if o["kind"] == ep["zone_objects"][zname])
        inside = (x0 <= obj["xy"][0] <= x1) and (y0 <= obj["xy"][1] <= y1)
        assert inside, f"{obj['kind']} at {obj['xy']} outside {zname} span"
        seen = [n for n in CAM_NAMES
                if masks[n][int((obj["xy"][1] - _IY0) / GRID_RES),
                            int((obj["xy"][0] - _IX0) / GRID_RES)]]
        print(f"[episode]  {zname}: {ep['zone_objects'][zname]:12s} "
              f"at ({obj['xy'][0]:6.2f}, {obj['xy'][1]:6.2f})  "
              f"CCTV seeing it: {','.join(seen) or 'none'}")
    print("[self-check] every object lies inside its generator span")

    render_sub_lang_zones(ep, cfg)
    render_sub_lang_objects(rings, cams, ep)
    render_sub_lang_cctv()

    print()
    print("--- LaTeX skeleton (subfigure package) ---")
    print(r"""\begin{figure}[t]
  \centering
  \begin{subfigure}[t]{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/sub_lang_zones.pdf}
    \caption{Placement.}\label{fig:lang-zones}
  \end{subfigure}\hfill
  \begin{subfigure}[t]{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/sub_lang_objects.pdf}
    \caption{One family.}\label{fig:lang-objects}
  \end{subfigure}\hfill
  \begin{subfigure}[t]{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/sub_lang_cctv.pdf}
    \caption{Its CCTV views.}\label{fig:lang-cctv}
  \end{subfigure}
  \caption{<paste the caption clause below>}\label{fig:lang}
\end{figure}""")
    print()
    print("--- caption clause (edit to taste) ---")
    zo = ep["zone_objects"]
    print(f"The language-goal task. (a) Placement: exactly one object is placed "
          f"in each of the four aisles, confined to the aisle half farther from "
          f"the cross-aisle D — the half nearer that aisle's CCTV — so no target "
          f"is visible from the onboard camera at the start. The robot always "
          f"spawns at the same pose against D's west edge, facing east; only the "
          f"instruction changes. (b) One recorded family ({ep['name']}): "
          f"{zo['B']} in B, {zo['C']} in C, {zo['E']} in E, {zo['F']} in F, with "
          f"each panel repeating the four objects over one camera's observed "
          f"floor — every object falls in its own camera's aisle, so the fixed "
          f"cameras jointly see all four candidates while the robot at spawn "
          f"sees none. Small objects are drawn as symbols; the forklift at its "
          f"measured footprint. (c) The four CCTV views of that scene. This "
          f"family's four twins share these placements byte-for-byte and differ "
          f"only in the instruction — here \"{ep['instruction']}\", whose target "
          f"is the {ep['target_kind']} in aisle {ep['goal_zone']}. Band depths "
          f"are compressed as in the setup figure; widths are to scale.")


if __name__ == "__main__":
    main()
