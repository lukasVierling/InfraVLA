
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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch

from isaac_sim.scenes._common import (
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
    _shoelace,
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
    C_MUTED,
    C_RACK,
    WALL_T,
    WARP_RDP_TOL,
    draw_camera_mark,
    draw_structure,
    zone_letter_pos,
)
from viz._lib.warehouse_2x3_schematic import (
    _TRUE_BAND_M,
    _Y_EDGES_TRUE,
    warp_y,
)

C_SEAM = "#b8b6ad"

# SUB_W = the PRINTED width of one 0.32\textwidth subfigure slot (inches), so
# nominal font sizes survive \includegraphics[width=\linewidth] unscaled.
SUB_W = 2.05
FS_PANEL, FS_ZONE, FS_RAIL, FS_MINI, FS_LEGEND = 9.0, 7.5, 5.8, 6.6, 7.4
RAIL_GAP, RAIL_BAND = 0.6, 2.0

OUT_DIR = _REPO_ROOT / "isaac_sim" / "outputs" / "wh2x3_setup_composite"
FRAME_DIR = _REPO_ROOT / "isaac_sim" / "outputs" / "cctv_observation_panel"

_IX0, _IX1 = WAREHOUSE_2X3_INTERIOR_X
_IY0, _IY1 = WAREHOUSE_2X3_INTERIOR_Y
_NX0, _NX1 = WAREHOUSE_2X3_NAV_X
_SY0, _SY1 = warp_y(_IY0), warp_y(_IY1)

_YLIM = (_SY0 - WALL_T - RAIL_GAP - RAIL_BAND, _SY1 + WALL_T + 2.4)


def draw_layout_panel(ax) -> None:
    """Zones, dotted boundaries, dimension rails, north arrow (no title)."""
    draw_structure(ax)
    for yb in _Y_EDGES_TRUE[1:-1]:
        ax.plot([_NX0, _NX1], [warp_y(yb)] * 2, color=C_SEAM, lw=1.0,
                ls=(0, (1.2, 2.4)), zorder=1.5)
    for zname in WAREHOUSE_2X3_ZONES:
        zx, zy = zone_letter_pos(zname)
        ax.text(zx, zy, zname, fontsize=FS_ZONE, color=C_INK2,
                ha="center", va="center", zorder=6)

    xr = _IX0 - WALL_T - RAIL_GAP
    yr = _SY0 - WALL_T - RAIL_GAP
    y_edges_schem = [warp_y(y) for y in _Y_EDGES_TRUE]
    ax.plot([xr, xr], [_SY0, _SY1], color=C_INK2, lw=0.7, zorder=4)
    for ys in y_edges_schem:
        ax.plot([xr - 0.12, xr + 0.12], [ys] * 2, color=C_INK2, lw=0.7, zorder=4)
    for i, m in enumerate(_TRUE_BAND_M):
        cy = 0.5 * (y_edges_schem[i] + y_edges_schem[i + 1])
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

    ax.annotate("N", xy=(_IX1 - 0.8, _SY1 + WALL_T + 2.1),
                xytext=(_IX1 - 0.8, _SY1 + WALL_T + 0.3),
                fontsize=FS_PANEL - 1.5, color=C_INK, ha="center", va="bottom",
                fontweight="semibold",
                arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.0), zorder=6)
    ax.set_xlim(xr - 1.8, _IX1 + WALL_T + 0.4)
    ax.set_ylim(*_YLIM)


def draw_mini_panel(ax, name: str, rings, cam, pct: float) -> None:
    """One coverage cell: this camera's observed floor, compact."""
    draw_structure(ax)
    colour = CAM_COLOURS[name]
    path = compound_path(rings)
    ax.add_patch(PathPatch(path, facecolor=colour, alpha=CAM_FILL_ALPHA,
                           edgecolor="none", zorder=2))
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=colour,
                           lw=0.7, zorder=2.5))
    draw_camera_mark(ax, cam, colour)
    owned = OWNED_ZONE[name]
    zx, zy = zone_letter_pos(owned)
    ax.text(zx, zy, owned, fontsize=FS_ZONE, color=colour, fontweight="bold",
            ha="center", va="center", zorder=6)
    ax.set_xlim(_IX0 - WALL_T - 0.6, _IX1 + WALL_T + 0.6)
    ax.set_ylim(_SY0 - WALL_T - 0.5, _SY1 + WALL_T + 0.5)
    ax.set_title(f"{name}$\\,\\to\\,${owned}", fontsize=FS_MINI, color=C_INK,
                 pad=2.5)


def draw_frame_panel(ax, name: str) -> None:
    """One observation cell: the camera's rendered view, framed in its colour."""
    img_path = FRAME_DIR / f"cctv_{name}.png"
    assert img_path.exists(), (
        f"{img_path} missing — render the views first: "
        f"viz/paper/cctv_observation_panel.py (needs Isaac)")
    ax.imshow(mpimg.imread(img_path))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(CAM_COLOURS[name])
        spine.set_linewidth(1.8)
    ax.set_title(name, fontsize=FS_MINI, color=C_INK, pad=2.5)


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[out]      {OUT_DIR}/{stem}.png (+ .pdf)")


# ---------------------------------------------------------------------------
# The three subfigure singles (no headers; own legends; print-size fonts)
# ---------------------------------------------------------------------------

def render_sub_layout() -> None:
    # Figure created tall on purpose: the axes is sized in INCHES so the map is
    # as large as SUB_W allows, and place_legend_below() trims the slack.
    fig = plt.figure(figsize=(SUB_W, 5.2), facecolor="white")
    ax = fig.add_axes([0.01, 0.99 - 3.24 / 5.2, 0.98, 3.24 / 5.2])
    draw_layout_panel(ax)
    leg = fig.legend(handles=[
        Line2D([], [], color=C_SEAM, lw=1.0, ls=(0, (1.2, 2.4)),
               label="zone boundary"),
        Patch(facecolor=C_MARGIN, edgecolor="none",
              label="outside navigable bounds"),
        Patch(facecolor=C_RACK, edgecolor="none", label="rack"),
    ], loc="lower center", ncol=1, frameon=False, fontsize=FS_LEGEND,
        labelcolor=C_INK, handlelength=1.4, handletextpad=0.5,
        labelspacing=0.4, bbox_to_anchor=(0.5, 0.0))
    place_legend_below(fig, leg)
    _save(fig, "sub_layout")


def render_sub_coverage(rings, cams, pcts) -> None:
    fig = plt.figure(figsize=(SUB_W, 5.2), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.26,
                  left=0.01, right=0.99, top=0.955, bottom=0.955 - 3.375 / 5.2)
    for i, name in enumerate(CAM_NAMES):
        ax = fig.add_subplot(gs[i])
        draw_mini_panel(ax, name, rings[name], cams[name], pcts[name])
    # The observed-floor swatch is the four camera hues side by side, because
    # the fill colour differs per panel and a single neutral swatch would not
    # match anything on the canvas.
    observed_swatch = tuple(
        Patch(facecolor=CAM_COLOURS[n], alpha=CAM_FILL_ALPHA,
              edgecolor=CAM_COLOURS[n], linewidth=0.7)
        for n in CAM_NAMES
    )
    leg = fig.legend(
        handles=[Line2D([], [], color=C_INK, marker="o", ms=4.0, lw=1.0),
                 observed_swatch],
        labels=["CCTV mount + view direction", "floor observed by the camera"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="lower center", ncol=1, frameon=False, fontsize=FS_LEGEND,
        labelcolor=C_INK, handlelength=1.8, handletextpad=0.5,
        labelspacing=0.4, bbox_to_anchor=(0.5, 0.0))
    place_legend_below(fig, leg)
    _save(fig, "sub_coverage")


def render_sub_observations() -> None:
    fig = plt.figure(figsize=(SUB_W, 2.45), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, wspace=0.07, hspace=0.22,
                  left=0.015, right=0.985, top=0.925, bottom=0.015)
    for i, name in enumerate(CAM_NAMES):
        draw_frame_panel(fig.add_subplot(gs[i]), name)
    _save(fig, "sub_observations")


# ---------------------------------------------------------------------------
# The all-in-one composite (kept for quick eyeballing)
# ---------------------------------------------------------------------------

def render_composite(rings, cams, pcts) -> None:
    fig = plt.figure(figsize=(8.2, 4.35), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.10, 1.10, 1.30],
                  wspace=0.08, left=0.005, right=0.995, top=0.90, bottom=0.11)
    ax_layout = fig.add_subplot(gs[0])
    gs_minis = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1],
                                       wspace=0.04, hspace=0.22)
    mini_axes = [fig.add_subplot(gs_minis[i]) for i in range(4)]
    gs_frames = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2],
                                        wspace=0.06, hspace=0.22)
    frame_axes = [fig.add_subplot(gs_frames[i]) for i in range(4)]

    draw_layout_panel(ax_layout)
    for ax, name in zip(mini_axes, CAM_NAMES):
        draw_mini_panel(ax, name, rings[name], cams[name], pcts[name])
    for ax, name in zip(frame_axes, CAM_NAMES):
        draw_frame_panel(ax, name)

    def region_title(axes, text):
        boxes = [a.get_position() for a in axes]
        fig.text(0.5 * (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)),
                 ax_layout.get_position().y1 + 0.045, text,
                 fontsize=FS_PANEL, color=C_INK, ha="center", va="bottom")

    region_title([ax_layout], "(a) warehouse layout")
    region_title(mini_axes, "(b) observed floor per camera")
    region_title(frame_axes, "(c) CCTV observations")

    fig.legend(handles=[
        Line2D([], [], color=C_SEAM, lw=1.0, ls=(0, (1.2, 2.4)),
               label="zone boundary"),
        Line2D([], [], color=C_INK, marker="o", ms=4.5, lw=1.1,
               label="CCTV mount + view direction"),
        Patch(facecolor=C_MARGIN, edgecolor="none",
              label="outside navigable bounds"),
        Patch(facecolor=C_RACK, edgecolor="none",
              label="rack (opaque occluder model)"),
    ], loc="lower center", ncol=4, frameon=False, fontsize=7.0,
        labelcolor=C_INK, handlelength=1.5, handletextpad=0.6,
        columnspacing=1.3, bbox_to_anchor=(0.5, 0.0))
    _save(fig, "wh2x3_setup_composite")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cams = {name: load_bigcam(name) for name in CAM_NAMES}

    xs = np.arange(_IX0 + GRID_RES / 2, _IX1, GRID_RES)
    ys = np.arange(_IY0 + GRID_RES / 2, _IY1, GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    masks = {name: visibility_mask(cams[name], X, Y) for name in CAM_NAMES}
    count = np.sum(np.stack([masks[n] for n in CAM_NAMES]), axis=0)

    in_rack = np.zeros_like(X, dtype=bool)
    for (x0, y0, x1, y1) in RACK_RECTS:
        in_rack |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    nav = ((X >= _NX0) & (X <= _NX1)
           & (Y >= WAREHOUSE_2X3_NAV_Y[0]) & (Y <= WAREHOUSE_2X3_NAV_Y[1])
           & ~in_rack)
    n_nav = int(nav.sum())
    pcts = {n: 100.0 * (masks[n] & nav).sum() / n_nav for n in CAM_NAMES}
    pct_union = 100.0 * ((count >= 1) & nav).sum() / n_nav
    pct_multi = 100.0 * ((count >= 2) & nav).sum() / n_nav
    print(f"[coverage] per-cam {'/'.join(f'{pcts[n]:.1f}' for n in CAM_NAMES)} %  "
          f"union {pct_union:.1f} %  (>=2: {pct_multi:.1f} %)")

    for name, zname in OWNED_ZONE.items():
        zone = WAREHOUSE_2X3_ZONES[zname]
        zmask = (nav & (X >= zone.x_range[0]) & (X <= zone.x_range[1])
                 & (Y >= zone.y_range[0]) & (Y <= zone.y_range[1]))
        own = 100.0 * (masks[name] & zmask).sum() / max(int(zmask.sum()), 1)
        assert own >= 60.0, f"{name} covers only {own:.1f}% of its aisle {zname}"

    rings = {}
    for name in CAM_NAMES:
        dense = mask_to_rings(masks[name], xs, ys)
        poly_area = sum(_shoelace(r) for r in dense)
        raster_area = masks[name].sum() * GRID_RES**2
        err = abs(abs(poly_area) - raster_area) / raster_area
        assert err < 0.02, f"{name}: polygon/raster area mismatch {err:.1%}"
        rings[name] = frame_rings(dense, "warped", WARP_RDP_TOL)
    print("[self-check] own-aisle >= 60% and polygon areas match rasters (<2%)")

    render_sub_layout()
    render_sub_coverage(rings, cams, pcts)
    render_sub_observations()
    render_composite(rings, cams, pcts)

    print()
    print("--- LaTeX skeleton (subfigure package) ---")
    print(r"""\begin{figure}[t]
  \centering
  \begin{subfigure}[t]{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/sub_layout.pdf}
    \caption{Warehouse layout.}\label{fig:setup-layout}
  \end{subfigure}\hfill
  \begin{subfigure}[t]{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/sub_coverage.pdf}
    \caption{Observed floor per camera.}\label{fig:setup-coverage}
  \end{subfigure}\hfill
  \begin{subfigure}[t]{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/sub_observations.pdf}
    \caption{CCTV observations.}\label{fig:setup-observations}
  \end{subfigure}
  \caption{<paste the caption clause below>}\label{fig:setup}
\end{figure}""")
    print()
    print("--- caption clause (edit to taste) ---")
    print(f"Simulation testbed. (a) The 2x3 warehouse: zones A-G form the "
          f"digital-8 topology in which the aisle pairs B/C and E/F are parallel "
          f"alternatives; dimension rails give true metres — band depths (left) "
          f"are compressed per band for legibility, widths (bottom) are to scale. "
          f"(b) The floor observed by each of the four ceiling CCTVs (16 mm, "
          f"720x720, z = {cams['NE'].t[2]:.1f} m; panels arranged like the "
          f"mounts), computed by projecting every floor point through the "
          f"deployed intrinsics and poses with racks as opaque full-height "
          f"occluders — a conservative analytical model. Each camera is aimed "
          f"down one rack aisle (NW->B, NE->C, SW->E, SE->F) and observes "
          f"~{np.mean(list(pcts.values())):.0f}% of the navigable floor; jointly "
          f"{pct_union:.0f}% ({pct_multi:.0f}% by two or more), covering all "
          f"four aisles completely. (c) The corresponding camera views (frame "
          f"colour = camera), rendered in the empty hall with the robot at its "
          f"spawn. Coverage regions in (b) are warped with the frame and remain "
          f"point-wise exact.")


if __name__ == "__main__":
    main()
