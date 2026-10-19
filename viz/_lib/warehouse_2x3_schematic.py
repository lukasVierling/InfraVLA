
from __future__ import annotations

import sys
import pathlib
from pathlib import Path

import numpy as np

_REPO_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from omegaconf import OmegaConf

from isaac_sim.scenes._common import (
    WAREHOUSE_2X3_ZONES,
    WAREHOUSE_2X3_INTERIOR_X,
    WAREHOUSE_2X3_INTERIOR_Y,
    WAREHOUSE_2X3_NAV_X,
    warehouse_2x3_rack_rects,
)

# ---------------------------------------------------------------------------
# Palette -- same tokens as planner_ablation_panel.py for cross-figure
# consistency (categorical slots 1-3, validated).
# ---------------------------------------------------------------------------
SURFACE = "#ffffff"
OBSTACLE = "#3c4450"     # racks + perimeter walls
C_ROBOT = "#2a78d6"      # slot 1, blue   — robot spawn
C_SEAL = "#eb6834"       # slot 2, orange — forklift seal (the threat)
C_GOAL = "#1baf7a"       # slot 3, aqua   — goal
FLOOR = "#f3f6f9"        # navigable floor, faint blue-grey
MARGIN_TINT = "#e6eaee"  # outside the A* navigable bounds (one step darker)
INK = "#0b0b0b"          # legend text, north arrow
INK_2 = "#52514e"        # zone letters, dimension rails
SEAM = "#b6bcc4"         # dotted zone-boundary lines (general figure)

FILL_ALPHA = 0.38        # soft fills are light tints of the hue over white
EDGE_LW = 0.9            # hairline full-hue edge on each fill (identity cue)
FS_LEGEND, FS_ZONE, FS_RAIL = 9.2, 12.5, 7.0

FIG_W = 3.3              # inches; close to printed width in a thesis column
WALL_T = 0.35            # drawn wall thickness (schematic units)
RAIL_GAP = 0.60          # wall outer face -> rail line
RAIL_W = 1.55            # extra canvas reserved per rail side

# ---------------------------------------------------------------------------
# Piecewise y-warp: true y -> schematic y. Each band keeps its internal
# proportions; only the band HEIGHTS are reassigned. The left dimension rail
# prints the TRUE metres per band, so the warp is disclosed on the canvas.
# ---------------------------------------------------------------------------
_ZONES = WAREHOUSE_2X3_ZONES
_Y_EDGES_TRUE = [
    WAREHOUSE_2X3_INTERIOR_Y[0],      # -20.5  south wall inner face
    _ZONES["G"].y_range[1],           # -15.2  G / front-band boundary
    _ZONES["D"].y_range[0],           #   1.0  front-band / D
    _ZONES["D"].y_range[1],           #   8.8  D / back-band
    _ZONES["A"].y_range[0],           #  25.0  back-band / A
    WAREHOUSE_2X3_INTERIOR_Y[1],      #  30.3  north wall inner face
]
# Taller bands (G, front aisles, D, back aisles, A): aisle bands dominate, as
# they do in reality (each is 32% of the true hall length). Aspect ~1:1.6.
_BAND_HEIGHTS = [2.6, 7.0, 3.4, 7.0, 2.6]
_Y_EDGES_SCHEM = np.concatenate([[0.0], np.cumsum(_BAND_HEIGHTS)])
_TRUE_BAND_M = [_Y_EDGES_TRUE[i + 1] - _Y_EDGES_TRUE[i] for i in range(5)]

_IX0, _IX1 = WAREHOUSE_2X3_INTERIOR_X
_IY0, _IY1 = WAREHOUSE_2X3_INTERIOR_Y
_NX0, _NX1 = WAREHOUSE_2X3_NAV_X

_PLACEMENT = OmegaConf.load(
    _REPO_ROOT / "isaac_sim" / "configs" / "datagen_2x3.yaml").placement
PT_MARGIN = float(_PLACEMENT.point_margin_m)
END_MARGIN = float(_PLACEMENT.blocker_end_margin_m)
F_LO, F_HI = (float(v) for v in _PLACEMENT.blocker_y_span_frac)

OUT_DIR = _REPO_ROOT / "isaac_sim" / "outputs" / "warehouse_2x3_schematic"


def warp_y(y: float) -> float:
    """True world y -> schematic y (piecewise linear, band proportions kept)."""
    return float(np.interp(y, _Y_EDGES_TRUE, _Y_EDGES_SCHEM))


def rect_schem(x0, y0, x1, y1, **kw) -> Rectangle:
    """Rectangle in (true x, warped y) coordinates."""
    return Rectangle((x0, warp_y(y0)), x1 - x0, warp_y(y1) - warp_y(y0), **kw)


def seal_span(name: str):
    """Sampling span of the forklift CENTRE in aisle `name`: (x_range, (lo, hi))."""
    z = _ZONES[name]
    y0, y1 = z.y_range
    lo = max(y0 + END_MARGIN, y0 + F_LO * (y1 - y0))
    hi = min(y1 - END_MARGIN, y0 + F_HI * (y1 - y0))
    return z.x_range, (lo, hi)


def build_canvas(legend_h_in: float):
    """Figure + map axes with everything task-agnostic EXCEPT seams/letters:
    walls, floor tints, racks, dimension rails, north arrow."""
    sy0, sy1 = warp_y(_IY0), warp_y(_IY1)
    map_w_units = (_IX1 - _IX0) + 2 * WALL_T + RAIL_W          # left rail
    map_h_units = (sy1 - sy0) + 2 * WALL_T + RAIL_W            # bottom rail
    map_h_in = FIG_W * map_h_units / map_w_units
    fig_h = map_h_in + legend_h_in + 0.06
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=SURFACE)
    ax = fig.add_axes([0.0, legend_h_in / fig_h, 1.0, map_h_in / fig_h])
    ax.set_xlim(_IX0 - WALL_T - RAIL_W, _IX1 + WALL_T)
    ax.set_ylim(sy0 - WALL_T - RAIL_W, sy1 + WALL_T)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(SURFACE)

    # Perimeter wall ring (drawn OUTSIDE the interior faces).
    for (x0, y0, x1, y1) in (
        (_IX0 - WALL_T, sy0 - WALL_T, _IX1 + WALL_T, sy0),        # south
        (_IX0 - WALL_T, sy1, _IX1 + WALL_T, sy1 + WALL_T),        # north
        (_IX0 - WALL_T, sy0, _IX0, sy1),                          # west
        (_IX1, sy0, _IX1 + WALL_T, sy1),                          # east
    ):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor=OBSTACLE, edgecolor="none", zorder=3))

    # Floor: faint tint over the navigable box, one step darker over the
    # non-navigable rack<->wall margins (outside the A* navigable_bounds mask).
    ax.add_patch(Rectangle((_NX0, sy0), _NX1 - _NX0, sy1 - sy0,
                           facecolor=FLOOR, edgecolor="none", zorder=0.8))
    for (x0, x1) in ((_IX0, _NX0), (_NX1, _IX1)):
        ax.add_patch(Rectangle((x0, sy0), x1 - x0, sy1 - sy0,
                               facecolor=MARGIN_TINT, edgecolor="none", zorder=1))

    # The six racks (3 columns x 2 rows), from the shared derived geometry.
    for (x0, y0, x1, y1) in warehouse_2x3_rack_rects(seg_len=99.0):
        ax.add_patch(rect_schem(x0, y0, x1, y1,
                                facecolor=OBSTACLE, edgecolor="none", zorder=3))

    # ---------------------------------------------------------- dimension rails
    # TRUE metres on both axes. Depth (left): the per-band heights the y-warp
    # compresses -- this rail IS the not-to-scale disclosure. Width (bottom):
    # margin / rack / aisle / rack / aisle / rack / margin; x is true scale.
    B, C = _ZONES["B"], _ZONES["C"]
    x_edges = [_IX0, _NX0, B.x_range[0], B.x_range[1],
               C.x_range[0], C.x_range[1], _NX1, _IX1]
    xr = _IX0 - WALL_T - RAIL_GAP      # depth-rail line x
    yr = sy0 - WALL_T - RAIL_GAP       # width-rail line y

    ax.plot([xr, xr], [sy0, sy1], color=INK_2, lw=0.8, zorder=4)
    for ys in _Y_EDGES_SCHEM:
        ax.plot([xr - 0.13, xr + 0.13], [ys] * 2, color=INK_2, lw=0.8, zorder=4)
    for i, m in enumerate(_TRUE_BAND_M):
        cy = 0.5 * (_Y_EDGES_SCHEM[i] + _Y_EDGES_SCHEM[i + 1])
        ax.text(xr - 0.32, cy, f"{m:g}", fontsize=FS_RAIL, color=INK_2,
                ha="center", va="center", rotation=90, zorder=4)

    ax.plot([_IX0, _IX1], [yr, yr], color=INK_2, lw=0.8, zorder=4)
    for xe in x_edges:
        ax.plot([xe] * 2, [yr - 0.13, yr + 0.13], color=INK_2, lw=0.8, zorder=4)
    for i in range(len(x_edges) - 1):
        cx = 0.5 * (x_edges[i] + x_edges[i + 1])
        w = x_edges[i + 1] - x_edges[i]
        ax.text(cx, yr - 0.30, f"{w:.1f}", fontsize=FS_RAIL, color=INK_2,
                ha="center", va="top", zorder=4)

    # Shared unit tag at the rail crossing (both rails are metres).
    ax.text(xr, yr - 0.30, "m", fontsize=FS_RAIL + 0.5, color=INK_2,
            ha="center", va="top", fontstyle="italic", zorder=4)

    # North indicator: the robot travels A (top) -> G (bottom).
    ax.annotate("N", xy=(_IX1 - 0.55, sy1 - 0.35), xytext=(_IX1 - 0.55, sy1 - 1.45),
                fontsize=9, color=INK, ha="center", va="top", fontweight="semibold",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2), zorder=6)
    return fig, ax, fig_h, legend_h_in, x_edges


def zone_letters(ax, *, avoid_seal: bool):
    """A-G labels. avoid_seal centres aisle letters in the seal-free span
    (task figure); otherwise in the whole aisle (general figure)."""
    for name, z in _ZONES.items():
        cx = 0.5 * (z.x_range[0] + z.x_range[1])
        if name in ("B", "C", "E", "F") and avoid_seal:
            _, (_, hi) = seal_span(name)
            cy = 0.5 * (warp_y(hi) + warp_y(z.y_range[1]))
        else:
            cy = 0.5 * (warp_y(z.y_range[0]) + warp_y(z.y_range[1]))
        ax.text(cx, cy, name, fontsize=FS_ZONE, color=INK_2, ha="center",
                va="center", zorder=6)


def add_legend(fig, handles, fig_h, legend_h_in):
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.02, (legend_h_in - 0.02) / fig_h),
               ncol=1, frameon=False, fontsize=FS_LEGEND, labelcolor=INK,
               handlelength=1.6, handletextpad=0.7, labelspacing=0.45,
               borderpad=0.0)


def save_fig(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=400, facecolor=SURFACE)
    plt.close(fig)
    print(f"[out]      {OUT_DIR}/{stem}.png")


def render_general():
    """Task-agnostic setup figure: dotted zone boundaries, no placement info."""
    fig, ax, fig_h, leg_h, _ = build_canvas(legend_h_in=0.58)
    # Zone boundaries: the four interior band edges, dotted, navigable width
    # only (the boundaries through the rack bands are physical, not zonal).
    for yb in _Y_EDGES_TRUE[1:-1]:
        ax.plot([_NX0, _NX1], [warp_y(yb)] * 2, color=SEAM, lw=1.1,
                ls=(0, (1.2, 2.6)), zorder=1.5)
    zone_letters(ax, avoid_seal=False)
    add_legend(fig, [
        Line2D([], [], color=SEAM, lw=1.1, ls=(0, (1.2, 2.6)),
               label="zone boundary"),
        Patch(facecolor=MARGIN_TINT, edgecolor="none",
              label="outside navigable bounds"),
    ], fig_h, leg_h)
    save_fig(fig, "wh2x3_layout_general")


def render_datagen():
    """cfv4 training data-gen placement figure (soft fills)."""
    fig, ax, fig_h, leg_h, _ = build_canvas(legend_h_in=0.92)

    def soft(zone_rect, colour):
        (x0, x1), (y0, y1) = zone_rect
        for face, edge, lw in ((colour, "none", 0.0), ("none", colour, EDGE_LW)):
            ax.add_patch(FancyBboxPatch(
                (x0, warp_y(y0)), x1 - x0, warp_y(y1) - warp_y(y0),
                boxstyle="round,pad=0,rounding_size=0.28",
                facecolor=face, alpha=FILL_ALPHA if face != "none" else 1.0,
                edgecolor=edge, linewidth=lw, zorder=2 if face != "none" else 2.5))

    A, G = _ZONES["A"], _ZONES["G"]
    soft(((A.x_range[0] + PT_MARGIN, A.x_range[1] - PT_MARGIN),
          (A.y_range[0] + PT_MARGIN, A.y_range[1] - PT_MARGIN)), C_ROBOT)
    soft(((G.x_range[0] + PT_MARGIN, G.x_range[1] - PT_MARGIN),
          (G.y_range[0] + PT_MARGIN, G.y_range[1] - PT_MARGIN)), C_GOAL)
    for name in ("B", "C", "E", "F"):
        x_range, (lo, hi) = seal_span(name)
        soft((x_range, (lo, hi)), C_SEAL)

    zone_letters(ax, avoid_seal=True)
    add_legend(fig, [
        Patch(facecolor=C_ROBOT, alpha=FILL_ALPHA, edgecolor=C_ROBOT,
              label="robot spawn region"),
        Patch(facecolor=C_GOAL, alpha=FILL_ALPHA, edgecolor=C_GOAL,
              label="goal region"),
        Patch(facecolor=C_SEAL, alpha=FILL_ALPHA, edgecolor=C_SEAL,
              label="forklift seal span"),
        Patch(facecolor=MARGIN_TINT, edgecolor="none",
              label="outside navigable bounds"),
    ], fig_h, leg_h)
    save_fig(fig, "wh2x3_placement_datagen")


def main():
    render_general()
    render_datagen()

    ix_w, iy_h = _IX1 - _IX0, _IY1 - _IY0
    print(f"[depth m]  {' / '.join(f'{m:g}' for m in _TRUE_BAND_M)}  (true band heights, left rail)")
    print(f"[rules]    spawn/goal margin {PT_MARGIN} m; forklift span frac "
          f"[{F_LO}, {F_HI}] of aisle length, end margin {END_MARGIN} m")
    print()
    print("--- caption clause: GENERAL layout figure ---")
    print(f"Schematic of the 2x3 warehouse (north up). Dimensions in metres are given "
          f"by the rails: widths (bottom) are drawn to scale, band depths (left) are "
          f"compressed per band for legibility -- the true hall measures "
          f"{ix_w:.0f} x {iy_h:.0f} m. Zones A/D/G are open strips, B/C/E/F the four "
          f"rack aisles (dotted lines mark zone boundaries); together they form the "
          f"digital-8 topology in which B/C and E/F are parallel alternatives. Darker "
          f"side margins lie outside the planner's navigable bounds.")
    print()
    print("--- caption clause: DATAGEN placement figure ---")
    print(f"Training-data placement in the 2x3 warehouse (layout and rails as in the "
          f"setup figure). The robot spawns in A and the goal is sampled in G; a "
          f"sealing forklift is confined to the south third of its aisle "
          f"({END_MARGIN:g} m end margin). Regions bound the sampled centre points, "
          f"not the vehicle extents -- the gaps to racks and walls are the "
          f"{PT_MARGIN:g} m sampling margins of the generator. On the robot's "
          f"north-to-south route every seal is therefore hidden from the onboard "
          f"camera at the routing decision while remaining visible to the CCTVs; the "
          f"remainder of each aisle is guaranteed forklift-free.")


if __name__ == "__main__":
    main()
