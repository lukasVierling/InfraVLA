
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
from matplotlib.patches import (
    FancyBboxPatch, Patch, PathPatch, Polygon, Rectangle,
)

from isaac_sim.scenes._common import (
    WAREHOUSE_2X3_AISLE_CLEAR_W,
    WAREHOUSE_2X3_INTERIOR_X,
    WAREHOUSE_2X3_INTERIOR_Y,
    WAREHOUSE_2X3_NAV_X,
    WAREHOUSE_2X3_NAV_Y,
    WAREHOUSE_2X3_ZONES,
)
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
    CAM_FILL_ALPHA,
    C_INK,
    C_INK2,
    C_MARGIN,
    C_RACK,
    WALL_T,
    WARP_RDP_TOL,
    draw_camera_mark,
    draw_structure,
    zone_letter_pos,
)
from viz._lib.env_panel_figure import (
    C_SEAL as C_SEAL_PAPER,
)
from viz._lib.warehouse_2x3_schematic import (
    C_GOAL,
    C_ROBOT,
    C_SEAL,
    EDGE_LW,
    FILL_ALPHA,
    PT_MARGIN,
    _TRUE_BAND_M,
    _Y_EDGES_TRUE,
    seal_span,
    warp_y,
)

EXAMPLE_RUN = (_REPO_ROOT / "runs" / "data_gen" / "warehouse_2x3_cfv4_2026-07-10"
               / "cfv4_p40" / "warehouse_2x3_1698_cfC__pose__2026-07-10_18-06-52")
FRAME_IDX = 20

MEASURED_FOOTPRINTS = (_REPO_ROOT / "isaac_sim" / "utils" / "scene_prep"
                       / "measured_footprints.json")

# How the sealing forklift is drawn. Candidates rendered side by side to
# outputs/.../proposals/blocker_styles.png on every run:
#   "footprint"  CHOSEN. Measured AABB in forklift yellow with a black edge,
#                thickened along the aisle only (see C_FORKLIFT below).
#   "cspace"     that AABB plus the planner's inflation halo — the region the
#                robot's CENTRE cannot enter, i.e. literally why it is a seal
#   "glyph"      forklift pictogram, true aspect but stretched along the
#                band-compressed axis so it stays legible (symbol, not to size)
#   "paper"      the convention the paper figures already use
#                (results_visualization/env_panel_figure.py): the measured
#                footprint in ITS red + black edge, plus a wash over the whole
#                sealed corridor. Its own comment gives the reason we hit here
#                -- "so 'this route is closed' reads from the back of the room,
#                not just the 1.2 m forklift footprint".
BLOCKER_STYLE = "footprint"
BLOCKER_STYLES = ("footprint", "cspace", "glyph", "paper")
SEAL_TINT_ALPHA = 0.14        # env_panel_figure.py's sealed-corridor wash
SEALED_AISLE = "C"            # which aisle this episode's blocker sits in

# Forklift yellow, matching the asset as it appears in the CCTV frames of (c).
# Thickened along the aisle so a 1.2 m-deep prop survives the band compression
# (factor 0.432) -- the drawn bar is FORKLIFT_THICKEN x its true depth; its
# width across the aisle stays exact, and that is the dimension that seals.
C_FORKLIFT = "#f2b705"
FORKLIFT_THICKEN = 1.9

C_SEAM = "#b8b6ad"

SUB_W = 2.05
FS_ZONE, FS_RAIL, FS_MINI, FS_LEGEND = 7.5, 5.8, 6.6, 7.4
RAIL_GAP, RAIL_BAND = 0.6, 1.2

OUT_DIR = _REPO_ROOT / "isaac_sim" / "outputs" / "wh2x3_example_composite"

_IX0, _IX1 = WAREHOUSE_2X3_INTERIOR_X
_IY0, _IY1 = WAREHOUSE_2X3_INTERIOR_Y
_NX0, _NX1 = WAREHOUSE_2X3_NAV_X
_SY0, _SY1 = warp_y(_IY0), warp_y(_IY1)

_YLIM = (_SY0 - WALL_T - RAIL_GAP - RAIL_BAND, _SY1 + WALL_T + 2.4)


def forklift_half_extents(variant: int) -> tuple[float, float]:
    """Measured yaw=0 AABB half extents (half_w along x, half_d along y), metres.

    From measure_prop_footprints.py's recorded measurements, so the drawn
    rectangle is the asset's real footprint (1.21 x 3.49 m), not an aspect
    ratio invented to look forklift-shaped.
    """
    data = json.loads(MEASURED_FOOTPRINTS.read_text())
    for entry in data["per_variant"]:
        if entry["kind"] == "forklift" and entry["variant_index"] == variant:
            aabb = entry["aabb_xy"]
            return float(aabb["half_w"]), float(aabb["half_d"])
    raise KeyError(f"no measured footprint for forklift variant {variant}")


def load_episode() -> dict:
    """Start, goal, blocker pose and the recorded path of the worked example."""
    cfg = yaml.safe_load((EXAMPLE_RUN / "config.yaml").read_text())
    scene = cfg["env"]
    nodes = json.loads((EXAMPLE_RUN / "frames.json").read_text())["nodes"]
    blockers = [o for o in scene["obstacles"] if o.get("role") == "blocker"]
    assert len(blockers) == 1, f"expected exactly one blocker, got {len(blockers)}"
    b = blockers[0]
    return dict(
        start=np.asarray(scene["start_xy"], dtype=float),
        goal=np.asarray(scene["goal_xy"], dtype=float),
        blocker=np.array([b["x"], b["y"]], dtype=float),
        blocker_yaw_deg=float(b["yaw_deg"]),
        blocker_half=forklift_half_extents(int(b.get("variant", 0))),
        inflation_m=float(scene["robot"]["inflation_m"]),
        path=np.array([n["xy"] for n in nodes], dtype=float),
    )


def draw_rails(ax) -> None:
    """The two true-metre dimension rails (the not-to-scale disclosure)."""
    xr = _IX0 - WALL_T - RAIL_GAP
    yr = _SY0 - WALL_T - RAIL_GAP
    y_edges = [warp_y(y) for y in _Y_EDGES_TRUE]
    ax.plot([xr, xr], [_SY0, _SY1], color=C_INK2, lw=0.7, zorder=4)
    for ys in y_edges:
        ax.plot([xr - 0.12, xr + 0.12], [ys] * 2, color=C_INK2, lw=0.7, zorder=4)
    for i, m in enumerate(_TRUE_BAND_M):
        cy = 0.5 * (y_edges[i] + y_edges[i + 1])
        ax.text(xr - 0.34, cy, f"{m:g}", fontsize=FS_RAIL, color=C_INK2,
                ha="center", va="center", rotation=90, zorder=4)

    B, C = WAREHOUSE_2X3_ZONES["B"], WAREHOUSE_2X3_ZONES["C"]
    x_edges = [_IX0, _NX0, B.x_range[0], B.x_range[1],
               C.x_range[0], C.x_range[1], _NX1, _IX1]
    ax.plot([_IX0, _IX1], [yr, yr], color=C_INK2, lw=0.7, zorder=4)
    for xe in x_edges:
        ax.plot([xe] * 2, [yr - 0.12, yr + 0.12], color=C_INK2, lw=0.7, zorder=4)
    for i in range(len(x_edges) - 1):
        cx = 0.5 * (x_edges[i] + x_edges[i + 1])
        ax.text(cx, yr - 0.32, f"{x_edges[i + 1] - x_edges[i]:.1f}",
                fontsize=FS_RAIL, color=C_INK2, ha="center", va="top", zorder=4)
    ax.text(xr, yr - 0.32, "m", fontsize=FS_RAIL + 0.5, color=C_INK2,
            ha="center", va="top", fontstyle="italic", zorder=4)
    ax.set_xlim(xr - 1.8, _IX1 + WALL_T + 0.4)
    ax.set_ylim(*_YLIM)


def soft_region(ax, x_range, y_range, colour: str) -> None:
    """Rounded soft fill + hairline full-hue edge (the placement-figure idiom)."""
    x0, x1 = x_range
    y0, y1 = (warp_y(v) for v in y_range)
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.22",
        facecolor=colour, alpha=FILL_ALPHA, edgecolor="none", zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.22",
        facecolor="none", edgecolor=colour, linewidth=EDGE_LW, zorder=2.5))


def render_sub_zones() -> None:
    # Axes box sized to the DATA aspect, so equal-aspect leaves no slack inside
    # it; the legend then sits directly under the bottom rail.
    # Axes sized in INCHES (map as large as SUB_W allows); the figure is created
    # tall and place_legend_below() trims the slack under the legend.
    fig = plt.figure(figsize=(SUB_W, 5.2), facecolor="white")
    ax = fig.add_axes([0.01, 0.99 - 3.17 / 5.2, 0.98, 3.17 / 5.2])
    draw_structure(ax)
    for yb in _Y_EDGES_TRUE[1:-1]:
        ax.plot([_NX0, _NX1], [warp_y(yb)] * 2, color=C_SEAM, lw=1.0,
                ls=(0, (1.2, 2.4)), zorder=1.5)

    A, G = WAREHOUSE_2X3_ZONES["A"], WAREHOUSE_2X3_ZONES["G"]
    soft_region(ax, (A.x_range[0] + PT_MARGIN, A.x_range[1] - PT_MARGIN),
                (A.y_range[0] + PT_MARGIN, A.y_range[1] - PT_MARGIN), C_ROBOT)
    soft_region(ax, (G.x_range[0] + PT_MARGIN, G.x_range[1] - PT_MARGIN),
                (G.y_range[0] + PT_MARGIN, G.y_range[1] - PT_MARGIN), C_GOAL)
    for name in ("B", "C", "E", "F"):
        x_range, y_span = seal_span(name)
        soft_region(ax, x_range, y_span, C_SEAL)

    for zname in WAREHOUSE_2X3_ZONES:
        zx, zy = zone_letter_pos(zname)
        ax.text(zx, zy, zname, fontsize=FS_ZONE, color=C_INK2,
                ha="center", va="center", zorder=6)
    draw_rails(ax)

    leg = fig.legend(handles=[
        Patch(facecolor=C_ROBOT, alpha=FILL_ALPHA, edgecolor=C_ROBOT,
              label="robot spawn region"),
        Patch(facecolor=C_GOAL, alpha=FILL_ALPHA, edgecolor=C_GOAL,
              label="goal region"),
        Patch(facecolor=C_SEAL, alpha=FILL_ALPHA, edgecolor=C_SEAL,
              label="forklift spawn span"),
        Patch(facecolor=C_RACK, edgecolor="none", label="rack"),
    ], loc="lower center", ncol=1, frameon=False, fontsize=FS_LEGEND,
        labelcolor=C_INK, handlelength=1.4, handletextpad=0.5,
        labelspacing=0.4, bbox_to_anchor=(0.5, 0.0))
    place_legend_below(fig, leg)
    _save(fig, "sub_zones")


def warp_scale(y: float) -> float:
    """Local dy_schematic/dy_true of the band warp, for drawing true-metre discs."""
    eps = 1e-4
    return (warp_y(y + eps) - warp_y(y - eps)) / (2.0 * eps)


def place(local: np.ndarray, ep, y_stretch: float = 1.0) -> np.ndarray:
    """Local (N, 2) metres -> drawing frame: rotate by yaw, translate, warp y.

    The warp is affine WITHIN a band and the blocker lies wholly inside the
    aisle band, so warping vertices is exact, not an approximation.
    y_stretch > 1 exaggerates extent along the compressed axis (symbol styles
    only: legibility bought with true size).
    """
    th = np.radians(ep["blocker_yaw_deg"])
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    world = np.asarray(local, dtype=float) @ rot.T + ep["blocker"]
    if y_stretch != 1.0:
        world[:, 1] = ep["blocker"][1] + (world[:, 1] - ep["blocker"][1]) * y_stretch
    return np.column_stack([world[:, 0], [warp_y(v) for v in world[:, 1]]])


def rect_local(half_w: float, half_d: float, dx: float = 0.0,
               dy: float = 0.0) -> np.ndarray:
    return np.array([[-half_w, -half_d], [half_w, -half_d],
                     [half_w, half_d], [-half_w, half_d]]) + np.array([dx, dy])


def inflated_local(half_w: float, half_d: float, r: float,
                   n_arc: int = 8) -> np.ndarray:
    """Rect (+) disc of radius r — the planner's C-space obstacle (rounded)."""
    pts = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        cx, cy = sx * half_w, sy * half_d
        a0 = np.arctan2(sy, sx) - np.pi / 4
        for a in np.linspace(a0, a0 + np.pi / 2, n_arc):
            pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.array(pts)


def forklift_glyph_local(half_w: float, half_d: float) -> list[np.ndarray]:
    """Top-down forklift pictogram, normalised to fill the measured AABB.

    +y is the forklift's long axis at yaw 0, forks forward: counterweight body,
    mast bar, two fork tines.
    """
    w, d = half_w, half_d
    return [
        rect_local(w, d * 0.52, dy=-d * 0.44),           # body / counterweight
        rect_local(w * 0.92, d * 0.07, dy=d * 0.14),     # mast
        rect_local(w * 0.22, d * 0.38, -w * 0.46, d * 0.58),   # left tine
        rect_local(w * 0.22, d * 0.38, w * 0.46, d * 0.58),    # right tine
    ]


def owned_letter_pos(zname: str) -> tuple[float, float]:
    """Aisle letter placed in the seal-FREE north half, so it never collides
    with a forklift (the generator confines seals to the south half)."""
    zone = WAREHOUSE_2X3_ZONES[zname]
    _, (_, seal_hi) = seal_span(zname)
    cx = 0.5 * (zone.x_range[0] + zone.x_range[1])
    cy = 0.5 * (warp_y(seal_hi) + warp_y(zone.y_range[1]))
    return cx, cy


def draw_blocker(ax, ep, style: str) -> None:
    """The sealing forklift, in one of three candidate styles."""
    half_w, half_d = ep["blocker_half"]
    if style == "footprint":
        ax.add_patch(Polygon(
            place(rect_local(half_w * FORKLIFT_THICKEN, half_d), ep),
            closed=True, facecolor=C_FORKLIFT, edgecolor="black",
            linewidth=0.8, zorder=4.5))
    elif style == "cspace":
        ax.add_patch(Polygon(
            place(inflated_local(half_w, half_d, ep["inflation_m"]), ep),
            closed=True, facecolor=C_SEAL, alpha=0.30, edgecolor=C_SEAL,
            linewidth=0.7, zorder=4.3))
        ax.add_patch(Polygon(place(rect_local(half_w, half_d), ep), closed=True,
                             facecolor=C_SEAL, edgecolor="black",
                             linewidth=0.8, zorder=4.5))
    elif style == "glyph":
        stretch = 1.0 / warp_scale(ep["blocker"][1])
        for part in forklift_glyph_local(half_w, half_d):
            ax.add_patch(Polygon(place(part, ep, y_stretch=stretch), closed=True,
                                 facecolor=C_SEAL, edgecolor="black",
                                 linewidth=0.6, zorder=4.5))
    elif style == "paper":
        zone = WAREHOUSE_2X3_ZONES[SEALED_AISLE]
        y0, y1 = (warp_y(v) for v in zone.y_range)
        ax.add_patch(Rectangle((zone.x_range[0], y0),
                               zone.x_range[1] - zone.x_range[0], y1 - y0,
                               facecolor=C_SEAL_PAPER, alpha=SEAL_TINT_ALPHA,
                               edgecolor="none", zorder=1.8))
        ax.add_patch(Polygon(place(rect_local(half_w, half_d), ep), closed=True,
                             facecolor=C_SEAL_PAPER, edgecolor="black",
                             linewidth=0.8, zorder=4.5))
    else:
        raise ValueError(f"unknown blocker style {style!r}")


def draw_episode_overlay(ax, ep, style: str | None = None) -> None:
    """The sealing forklift — identical in every panel."""
    draw_blocker(ax, ep, style or BLOCKER_STYLE)


def render_sub_example(rings_per_cam, cams, ep) -> None:
    # Grid sized to the panel data aspect and the legend given only the height
    # it needs, so no slack is left between the panels and the legend.
    fig = plt.figure(figsize=(SUB_W, 5.2), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.26,
                  left=0.01, right=0.99, top=0.95, bottom=0.95 - 3.375 / 5.2)
    for i, name in enumerate(CAM_NAMES):
        ax = fig.add_subplot(gs[i])
        draw_structure(ax)
        colour = CAM_COLOURS[name]
        path = compound_path(rings_per_cam[name])
        ax.add_patch(PathPatch(path, facecolor=colour, alpha=CAM_FILL_ALPHA,
                               edgecolor="none", zorder=2))
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=colour,
                               lw=0.7, zorder=2.5))
        draw_camera_mark(ax, cams[name], colour)
        draw_episode_overlay(ax, ep)
        owned = OWNED_ZONE[name]
        zx, zy = owned_letter_pos(owned)
        ax.text(zx, zy, owned, fontsize=FS_ZONE, color=colour,
                fontweight="bold", ha="center", va="center", zorder=6,
                path_effects=[pe.Stroke(linewidth=1.8, foreground="white"),
                              pe.Normal()])
        ax.set_xlim(_IX0 - WALL_T - 0.6, _IX1 + WALL_T + 0.6)
        ax.set_ylim(_SY0 - WALL_T - 0.5, _SY1 + WALL_T + 0.5)
        ax.set_title(f"{name}$\\,\\to\\,${owned}", fontsize=FS_MINI,
                     color=C_INK, pad=2.5)

    observed_swatch = tuple(
        Patch(facecolor=CAM_COLOURS[n], alpha=CAM_FILL_ALPHA,
              edgecolor=CAM_COLOURS[n], linewidth=0.7)
        for n in CAM_NAMES
    )
    leg = fig.legend(
        handles=[Line2D([], [], color=C_INK, marker="o", ms=3.6, lw=1.0),
                 observed_swatch,
                 Patch(facecolor=C_FORKLIFT, edgecolor="black", linewidth=0.8)],
        labels=["CCTV mount + view direction", "floor observed by the camera",
                "forklift sealing aisle C"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="lower center", ncol=1, frameon=False, fontsize=FS_LEGEND,
        labelcolor=C_INK, handlelength=1.8, handletextpad=0.5,
        labelspacing=0.4, bbox_to_anchor=(0.5, 0.0))
    place_legend_below(fig, leg)
    _save(fig, "sub_example")


def render_blocker_proposals(rings_per_cam, cams, ep) -> None:
    """The NE panel (the one that owns the sealed aisle) in all three styles."""
    fig, axes = plt.subplots(1, len(BLOCKER_STYLES), figsize=(6.0, 3.1),
                             facecolor="white")
    colour = CAM_COLOURS["NE"]
    for ax, style in zip(axes, BLOCKER_STYLES):
        draw_structure(ax)
        path = compound_path(rings_per_cam["NE"])
        ax.add_patch(PathPatch(path, facecolor=colour, alpha=CAM_FILL_ALPHA,
                               edgecolor="none", zorder=2))
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=colour,
                               lw=0.7, zorder=2.5))
        draw_camera_mark(ax, cams["NE"], colour)
        draw_blocker(ax, ep, style)
        zx, zy = owned_letter_pos("C")
        ax.text(zx, zy, "C", fontsize=FS_ZONE, color=colour, fontweight="bold",
                ha="center", va="center", zorder=6,
                path_effects=[pe.Stroke(linewidth=1.8, foreground="white"),
                              pe.Normal()])
        ax.set_xlim(_IX0 - WALL_T - 0.6, _IX1 + WALL_T + 0.6)
        ax.set_ylim(_SY0 - WALL_T - 0.5, _SY1 + WALL_T + 0.5)
        marker = "  <- current" if style == BLOCKER_STYLE else ""
        ax.set_title(f"{style}{marker}", fontsize=8.0, color=C_INK, pad=4.0)
    fig.tight_layout()
    out = OUT_DIR / "proposals"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "blocker_styles.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[out]      {out}/blocker_styles.png  "
          f"(pick one -> set BLOCKER_STYLE)")


def render_sub_example_cctv() -> None:
    fig = plt.figure(figsize=(SUB_W, 2.45), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, wspace=0.07, hspace=0.22,
                  left=0.015, right=0.985, top=0.925, bottom=0.015)
    for i, name in enumerate(CAM_NAMES):
        img_path = EXAMPLE_RUN / f"cctv_{name}" / f"{FRAME_IDX:03d}.png"
        assert img_path.exists(), f"{img_path} missing"
        ax = fig.add_subplot(gs[i])
        ax.imshow(mpimg.imread(img_path))
        ax.set_xticks([])
        ax.set_yticks([])
        # Border in the camera's colour — the same hue its coverage wears in
        # (b), so a reader can pair each view with its footprint.
        for spine in ax.spines.values():
            spine.set_edgecolor(CAM_COLOURS[name])
            spine.set_linewidth(1.8)
        ax.set_title(name, fontsize=FS_MINI, color=C_INK, pad=2.5)
    _save(fig, "sub_example_cctv")


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[out]      {OUT_DIR}/{stem}.png (+ .pdf)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert EXAMPLE_RUN.exists(), f"episode not found: {EXAMPLE_RUN}"
    ep = load_episode()
    cams = {name: load_bigcam(name) for name in CAM_NAMES}

    xs = np.arange(_IX0 + GRID_RES / 2, _IX1, GRID_RES)
    ys = np.arange(_IY0 + GRID_RES / 2, _IY1, GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    masks = {name: visibility_mask(cams[name], X, Y) for name in CAM_NAMES}
    rings_per_cam = {
        name: frame_rings(mask_to_rings(masks[name], xs, ys), "warped", WARP_RDP_TOL)
        for name in CAM_NAMES
    }
    union = np.zeros_like(X, dtype=bool)
    for m in masks.values():
        union |= m

    in_rack = np.zeros_like(X, dtype=bool)
    for (x0, y0, x1, y1) in RACK_RECTS:
        in_rack |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    nav = ((X >= _NX0) & (X <= _NX1)
           & (Y >= WAREHOUSE_2X3_NAV_Y[0]) & (Y <= WAREHOUSE_2X3_NAV_Y[1])
           & ~in_rack)
    pct_union = 100.0 * (union & nav).sum() / int(nav.sum())

    seg = np.linalg.norm(np.diff(ep["path"], axis=0), axis=1).sum()
    print(f"[episode]  {EXAMPLE_RUN.name}")
    print(f"[episode]  start ({ep['start'][0]:.2f}, {ep['start'][1]:.2f})  "
          f"goal ({ep['goal'][0]:.2f}, {ep['goal'][1]:.2f})  "
          f"forklift ({ep['blocker'][0]:.2f}, {ep['blocker'][1]:.2f})")
    print(f"[episode]  {len(ep['path'])} recorded steps, path {seg:.1f} m, "
          f"CCTV frame {FRAME_IDX}")
    print(f"[coverage] union {pct_union:.1f} % of navigable floor (static rig)")

    render_sub_zones()
    render_sub_example(rings_per_cam, cams, ep)
    render_sub_example_cctv()
    render_blocker_proposals(rings_per_cam, cams, ep)

    print()
    print("--- LaTeX skeleton (subfigure package) ---")
    print(r"""\begin{figure}[t]
  \centering
  \begin{subfigure}[t]{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/sub_zones.pdf}
    \caption{Placement regions.}\label{fig:ex-zones}
  \end{subfigure}\hfill
  \begin{subfigure}[t]{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/sub_example.pdf}
    \caption{One episode.}\label{fig:ex-path}
  \end{subfigure}\hfill
  \begin{subfigure}[t]{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/sub_example_cctv.pdf}
    \caption{Its CCTV views.}\label{fig:ex-cctv}
  \end{subfigure}
  \caption{<paste the caption clause below>}\label{fig:example}
\end{figure}""")
    print()
    print("--- caption clause (edit to taste) ---")
    print(f"A training episode. (a) Where the generator places things: the robot "
          f"spawns in zone A, the goal is sampled in zone G, and one forklift "
          f"seals the south part of one aisle; regions bound the sampled centre "
          f"points, so the gaps to racks and walls are the generator's "
          f"{PT_MARGIN:g} m sampling margins. (b) One recorded episode "
          f"(scene {EXAMPLE_RUN.name.split('__')[0]}): the robot starts top-right "
          f"and must reach the bottom-right goal, but a forklift seals the "
          f"north-east aisle C, so the planner routes it through the north-west "
          f"aisle B, across the cross-aisle D and down the south-east aisle F "
          f"({seg:.0f} m in {len(ep['path'])} steps). The forklift is drawn at "
          f"its measured {2 * ep['blocker_half'][1]:.1f} x "
          f"{2 * ep['blocker_half'][0]:.1f} m footprint, lying across a "
          f"{WAREHOUSE_2X3_AISLE_CLEAR_W:.1f} m clear aisle — hence a seal. Each "
          f"panel repeats it over one camera's observed floor (jointly "
          f"{pct_union:.0f}% of the navigable floor; static rig, the forklift is "
          f"not modelled as an occluder), showing that only the NE camera, which "
          f"owns aisle C, can see the blockade. "
          f"(c) The four CCTV views at the routing decision (step {FRAME_IDX}): "
          f"the NE camera shows both the robot in A and the forklift sealing C, "
          f"the information the onboard camera lacks at that moment. Band depths "
          f"are compressed as in the setup figure; widths are to scale.")


if __name__ == "__main__":
    main()
