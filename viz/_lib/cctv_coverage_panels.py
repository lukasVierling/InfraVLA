
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
from matplotlib.patches import Patch, PathPatch, Rectangle

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
    RDP_TOL_M,
    _shoelace,
    camera_axis_xy,
    compound_path,
    frame_rings,
    load_bigcam,
    mask_to_rings,
    to_frame,
    visibility_mask,
)
from viz._lib.warehouse_2x3_schematic import warp_y

C_WALL = "#41474f"
C_RACK = "#41474f"
C_FLOOR = "#ffffff"
C_MARGIN = "#eceae5"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#9a988f"

WALL_T = 0.35
CAM_FILL_ALPHA = 0.32
AXIS_ARROW_M = 1.8
WARP_RDP_TOL = RDP_TOL_M * 0.445
FOOTER_U = 2.6
HEADROOM_U = 0.9

FS_TITLE, FS_ZONE, FS_SMALL = 9.0, 8.0, 7.0

OUT_DIR = _REPO_ROOT / "isaac_sim" / "outputs" / "cctv_coverage_panels"

_IX0, _IX1 = WAREHOUSE_2X3_INTERIOR_X
_IY0, _IY1 = WAREHOUSE_2X3_INTERIOR_Y
_NX0, _NX1 = WAREHOUSE_2X3_NAV_X
_SY0, _SY1 = warp_y(_IY0), warp_y(_IY1)


LEGEND_GAP_IN = 0.07     # the ONE standard plot->legend gap, all subfigures


LEGEND_ROW_GAP_IN = 0.03   # between stacked legend blocks of one figure


def place_legends_below(fig, legends, gap_in: float = LEGEND_GAP_IN,
                        inner_gap_in: float = LEGEND_ROW_GAP_IN,
                        align: str = "center") -> float:
    """Stack `legends` (BOTTOM-FIRST) under the plot with the standard gap.

    Legend height is not known until it is drawn, so figures are created with
    slack and trimmed here: every axes keeps its size IN INCHES (equal-aspect
    panels untouched) and only the figure height changes. Returns the new
    figure height.

    align="center" — each block centred; legends must use loc="lower center".
    align="left"   — every block's LEFT edge on the same x (the widest block
                     stays centred in the figure); legends must use
                     loc="lower left", else the anchor means something else.
    """
    fig.canvas.draw()
    w_in, h_in = fig.get_size_inches()
    # (axes, x0, distance from figure TOP to axes top, width, height): the top
    # offset must stay fixed so titles keep their room.
    geom = []
    for ax in fig.axes:
        p = ax.get_position()
        geom.append((ax, p.x0, (1.0 - p.y1) * h_in, p.width, p.height * h_in))
    content_bottom_in = max(top + h for _, _, top, _, h in geom)

    boxes = [lg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
             for lg in legends]
    heights = [b.height for b in boxes]
    block = sum(heights) + inner_gap_in * (len(heights) - 1)
    new_h = content_bottom_in + gap_in + block

    fig.set_size_inches(w_in, new_h, forward=True)
    for ax, x0, top, w, h in geom:
        ax.set_position([x0, 1.0 - (top + h) / new_h, w, h / new_h])
    # Legend widths are set by point-sized text, so they are unchanged by the
    # resize; left-aligning means anchoring every block at the widest block's
    # left edge, which keeps the stack centred as a whole.
    anchor_x = 0.5
    if align == "left":
        anchor_x = 0.5 * (w_in - max(b.width for b in boxes)) / w_in
    y = 0.0
    for lg, hh in zip(legends, heights):
        lg.set_bbox_to_anchor((anchor_x, y / new_h))
        y += hh + inner_gap_in
    return new_h


def place_legend_below(fig, legend, gap_in: float = LEGEND_GAP_IN) -> float:
    """Single-legend case of place_legends_below()."""
    return place_legends_below(fig, [legend], gap_in=gap_in)


def draw_structure(ax) -> None:
    """Walls, floor, margin tint, racks — one panel's skeleton, warped frame."""
    ax.add_patch(Rectangle((_IX0, _SY0), _IX1 - _IX0, _SY1 - _SY0,
                           facecolor=C_FLOOR, edgecolor="none", zorder=0.8))
    for (x0, x1) in ((_IX0, _NX0), (_NX1, _IX1)):
        ax.add_patch(Rectangle((x0, _SY0), x1 - x0, _SY1 - _SY0,
                               facecolor=C_MARGIN, edgecolor="none", zorder=1))
    for (x0, y0, x1, y1) in (
        (_IX0 - WALL_T, _SY0 - WALL_T, _IX1 + WALL_T, _SY0),
        (_IX0 - WALL_T, _SY1, _IX1 + WALL_T, _SY1 + WALL_T),
        (_IX0 - WALL_T, _SY0, _IX0, _SY1),
        (_IX1, _SY0, _IX1 + WALL_T, _SY1),
    ):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor=C_WALL, edgecolor="none", zorder=3))
    for (x0, y0, x1, y1) in RACK_RECTS:
        sy0, sy1 = warp_y(y0), warp_y(y1)
        ax.add_patch(Rectangle((x0, sy0), x1 - x0, sy1 - sy0,
                               facecolor=C_RACK, edgecolor="none", zorder=3))
    ax.set_xlim(_IX0 - WALL_T - 0.5, _IX1 + WALL_T + 0.5)
    ax.set_ylim(_SY0 - WALL_T - FOOTER_U, _SY1 + WALL_T + HEADROOM_U)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_camera_mark(ax, cam, colour: str) -> None:
    """Mount dot + optical-axis arrow in the camera's colour, warped frame."""
    cx, cy = cam.t[0], warp_y(cam.t[1])
    az = camera_axis_xy(cam)
    tip = to_frame(np.array([cam.t[:2] + AXIS_ARROW_M * az]), "warped")[0]
    ax.annotate("", xy=(tip[0], tip[1]), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.3,
                                shrinkA=2.0, shrinkB=0.0), zorder=6)
    ax.plot([cx], [cy], marker="o", ms=5.5, color=colour,
            markeredgecolor="white", markeredgewidth=0.7, zorder=7)


def zone_letter_pos(zname: str) -> tuple[float, float]:
    zone = WAREHOUSE_2X3_ZONES[zname]
    cx = 0.5 * (zone.x_range[0] + zone.x_range[1])
    cy = 0.5 * (warp_y(zone.y_range[0]) + warp_y(zone.y_range[1]))
    return cx, cy


def draw_camera_panel(ax, name: str, rings, cam, pct: float) -> None:
    """One panel: this camera's observed floor, mount, and owned aisle."""
    draw_structure(ax)
    colour = CAM_COLOURS[name]
    path = compound_path(rings)
    ax.add_patch(PathPatch(path, facecolor=colour, alpha=CAM_FILL_ALPHA,
                           edgecolor="none", zorder=2))
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=colour,
                           lw=0.9, zorder=2.5))
    draw_camera_mark(ax, cam, colour)
    owned = OWNED_ZONE[name]
    for zname in WAREHOUSE_2X3_ZONES:
        zx, zy = zone_letter_pos(zname)
        if zname == owned:
            ax.text(zx, zy, zname, fontsize=FS_ZONE + 1.5, color=colour,
                    fontweight="bold", ha="center", va="center", zorder=6)
        else:
            ax.text(zx, zy, zname, fontsize=FS_ZONE - 1.0, color=C_MUTED,
                    ha="center", va="center", zorder=6)
    ax.set_title(f"{name} $\\rightarrow$ aisle {owned}  ({pct:.0f}% of floor)",
                 fontsize=FS_TITLE, color=C_INK, pad=4.0)


def draw_footer(ax_bl, ax_br) -> None:
    """5 m bar (x is true scale) on bottom-left; north arrow on bottom-right."""
    by = _SY0 - WALL_T - 1.5
    ax_bl.plot([_IX0, _IX0 + 5.0], [by, by], color=C_INK2, lw=0.9, zorder=4)
    for xe in (_IX0, _IX0 + 5.0):
        ax_bl.plot([xe, xe], [by - 0.22, by + 0.22], color=C_INK2, lw=0.9,
                   zorder=4)
    ax_bl.text(_IX0 + 2.5, by - 0.5, "5 m", fontsize=FS_SMALL, color=C_INK2,
               ha="center", va="top", zorder=4)
    ax_br.annotate("N", xy=(_IX1 - 0.9, by + 1.1), xytext=(_IX1 - 0.9, by - 0.9),
                   fontsize=FS_TITLE - 1.0, color=C_INK, ha="center", va="top",
                   fontweight="semibold",
                   arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.1),
                   zorder=6)


def render_panels(rings_per_cam, cams, pcts) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 7.6), facecolor="white",
                             gridspec_kw=dict(wspace=0.06, hspace=0.10,
                                              left=0.01, right=0.99,
                                              top=0.955, bottom=0.055))
    for ax, name in zip(axes.ravel(), CAM_NAMES):
        draw_camera_panel(ax, name, rings_per_cam[name], cams[name], pcts[name])
    draw_footer(axes[1, 0], axes[1, 1])

    handles = [
        Patch(facecolor="white", edgecolor=C_INK2, lw=0.5,
              label="floor not observed by this camera"),
        Patch(facecolor=C_MARGIN, edgecolor="none",
              label="outside navigable bounds"),
        Patch(facecolor=C_RACK, edgecolor="none",
              label="rack (opaque occluder model)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=FS_SMALL, labelcolor=C_INK,
               handlelength=1.4, handletextpad=0.6, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"wh2x3_cctv_panels.{ext}", dpi=400,
                    facecolor="white")
    plt.close(fig)
    print(f"[out]      {OUT_DIR}/wh2x3_cctv_panels.png (+ .pdf)")


# ---------------------------------------------------------------------------
# TikZ transcription of the same composition
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _ring(ring: np.ndarray) -> str:
    return " -- ".join(f"({_fmt(p[0])},{_fmt(p[1])})" for p in ring[:-1]) + " -- cycle"


def emit_tikz(rings_per_cam, cams, pcts) -> None:
    unit = 0.28
    pitch_x = (_IX1 - _IX0) + 2 * WALL_T + 2.2
    pitch_y = (_SY1 - _SY0) + 2 * WALL_T + FOOTER_U + HEADROOM_U + 2.6

    L: list[str] = []
    L.append("% AUTO-GENERATED by viz/_lib/cctv_coverage_panels.py -- do not hand-edit")
    L.append("% 2x2 per-camera observed floor, band-compressed y (as the layout schematic;")
    L.append("% disclose in the caption). x is true scale; the 5 m bar is horizontal for that reason.")
    L.append("% Occlusion model: rack footprints are opaque, infinitely high prisms (conservative).")
    L.append(r"\documentclass[tikz,border=2mm]{standalone}")
    L.append(r"\usetikzlibrary{arrows.meta}")
    for tok, hexc in (("wall", C_WALL), ("margin", C_MARGIN), ("ink", C_INK),
                      ("inktwo", C_INK2), ("muted", C_MUTED)):
        L.append(rf"\definecolor{{{tok}}}{{HTML}}{{{hexc.lstrip('#').upper()}}}")
    for name, colour in CAM_COLOURS.items():
        L.append(rf"\definecolor{{cam{name.lower()}}}{{HTML}}{{{colour.lstrip('#').upper()}}}")
    L.append(r"\begin{document}")
    L.append(rf"\begin{{tikzpicture}}[x={unit}cm,y={unit}cm]")

    def structure() -> list[str]:
        s = [rf"\fill[white] ({_fmt(_IX0)},{_fmt(_SY0)}) rectangle ({_fmt(_IX1)},{_fmt(_SY1)});"]
        for (x0, x1) in ((_IX0, _NX0), (_NX1, _IX1)):
            s.append(rf"\fill[margin] ({_fmt(x0)},{_fmt(_SY0)}) rectangle ({_fmt(x1)},{_fmt(_SY1)});")
        s.append(rf"\fill[wall, even odd rule] ({_fmt(_IX0 - WALL_T)},{_fmt(_SY0 - WALL_T)}) "
                 rf"rectangle ({_fmt(_IX1 + WALL_T)},{_fmt(_SY1 + WALL_T)}) "
                 rf"({_fmt(_IX0)},{_fmt(_SY0)}) rectangle ({_fmt(_IX1)},{_fmt(_SY1)});")
        for (x0, y0, x1, y1) in RACK_RECTS:
            s.append(rf"\fill[wall] ({_fmt(x0)},{_fmt(warp_y(y0))}) rectangle "
                     rf"({_fmt(x1)},{_fmt(warp_y(y1))});")
        return s

    title_y = _SY1 + WALL_T + 1.6
    for i, name in enumerate(CAM_NAMES):
        row, col = divmod(i, 2)
        style = f"cam{name.lower()}"
        owned = OWNED_ZONE[name]
        L.append(rf"\begin{{scope}}[xshift={_fmt(col * pitch_x * unit)}cm, "
                 rf"yshift={_fmt(-row * pitch_y * unit)}cm]")
        L.append(f"% ---- panel {name} ----")
        L.extend(structure())
        for r in rings_per_cam[name]:
            L.append(rf"\fill[{style}, fill opacity={CAM_FILL_ALPHA}] {_ring(r)};")
            L.append(rf"\draw[{style}, line width=0.5pt] {_ring(r)};")
        cam = cams[name]
        cx, cy = cam.t[0], warp_y(cam.t[1])
        az = camera_axis_xy(cam)
        tip = to_frame(np.array([cam.t[:2] + AXIS_ARROW_M * az]), "warped")[0]
        L.append(rf"\draw[{style}, -{{Stealth[length=3pt]}}, line width=0.9pt] "
                 rf"({_fmt(cx)},{_fmt(cy)}) -- ({_fmt(tip[0])},{_fmt(tip[1])});")
        L.append(rf"\fill[{style}] ({_fmt(cx)},{_fmt(cy)}) circle[radius=2.2pt];")
        for zname in WAREHOUSE_2X3_ZONES:
            zx, zy = zone_letter_pos(zname)
            if zname == owned:
                L.append(rf"\node[{style}, font=\small\bfseries] at "
                         rf"({_fmt(zx)},{_fmt(zy)}) {{{zname}}};")
            else:
                L.append(rf"\node[muted, font=\tiny] at ({_fmt(zx)},{_fmt(zy)}) {{{zname}}};")
        L.append(rf"\node[ink, font=\scriptsize] at "
                 rf"({_fmt(0.5 * (_IX0 + _IX1))},{_fmt(title_y)}) "
                 rf"{{\textbf{{{name} $\to$ aisle {owned}}} ({pcts[name]:.0f}\% of floor)}};")
        if row == 1 and col == 0:
            by = _SY0 - WALL_T - 1.5
            L.append(rf"\draw[inktwo, line width=0.8pt] ({_fmt(_IX0)},{_fmt(by)}) -- "
                     rf"({_fmt(_IX0 + 5.0)},{_fmt(by)}) node[midway, below=1pt, font=\tiny] {{5\,m}};")
            L.append(rf"\foreach \p in {{{_fmt(_IX0)},{_fmt(_IX0 + 5.0)}}} "
                     rf"\draw[inktwo, line width=0.8pt] (\p,{_fmt(by - 0.22)}) -- (\p,{_fmt(by + 0.22)});")
        if row == 1 and col == 1:
            by = _SY0 - WALL_T - 1.5
            L.append(rf"\draw[ink, -{{Stealth[length=4pt]}}, line width=0.9pt] "
                     rf"({_fmt(_IX1 - 0.9)},{_fmt(by - 0.9)}) -- ({_fmt(_IX1 - 0.9)},{_fmt(by + 1.1)}) "
                     rf"node[right=1pt, font=\scriptsize] {{N}};")
        L.append(r"\end{scope}")

    L.append(r"\end{tikzpicture}")
    L.append(r"\end{document}")
    out_path = OUT_DIR / "wh2x3_cctv_panels.tex"
    out_path.write_text("\n".join(L) + "\n")
    print(f"[out]      {out_path}")


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
    print(f"[coverage] navigable floor: {n_nav * GRID_RES**2:.1f} m^2")
    for n in CAM_NAMES:
        print(f"[coverage]   {n}: {pcts[n]:5.1f} %")
    print(f"[coverage]   union: {pct_union:5.1f} %   (>=2 cameras: {pct_multi:4.1f} %)")

    zone_pct = {}
    for zname, zone in WAREHOUSE_2X3_ZONES.items():
        zmask = (nav & (X >= zone.x_range[0]) & (X <= zone.x_range[1])
                 & (Y >= zone.y_range[0]) & (Y <= zone.y_range[1]))
        zone_pct[zname] = 100.0 * ((count >= 1) & zmask).sum() / max(int(zmask.sum()), 1)
    print("[coverage]   union per zone: "
          + "  ".join(f"{z}={zone_pct[z]:.0f}%" for z in "ABCDEFG"))

    for name, zname in OWNED_ZONE.items():
        zone = WAREHOUSE_2X3_ZONES[zname]
        zmask = (nav & (X >= zone.x_range[0]) & (X <= zone.x_range[1])
                 & (Y >= zone.y_range[0]) & (Y <= zone.y_range[1]))
        own = 100.0 * (masks[name] & zmask).sum() / max(int(zmask.sum()), 1)
        assert own >= 60.0, f"{name} covers only {own:.1f}% of its aisle {zname}"

    rings_per_cam = {}
    for name in CAM_NAMES:
        dense = mask_to_rings(masks[name], xs, ys)
        poly_area = sum(_shoelace(r) for r in dense)
        raster_area = masks[name].sum() * GRID_RES**2
        err = abs(abs(poly_area) - raster_area) / raster_area
        assert err < 0.02, f"{name}: polygon/raster area mismatch {err:.1%}"
        rings_per_cam[name] = frame_rings(dense, "warped", WARP_RDP_TOL)
    print("[self-check] own-aisle >= 60% and polygon areas match rasters (<2%)")

    render_panels(rings_per_cam, cams, pcts)
    emit_tikz(rings_per_cam, cams, pcts)

    print()
    print("--- caption clause (edit to taste) ---")
    print(f"Per-camera ground coverage of the 2x3 warehouse CCTV rig (panels "
          f"arranged like the mounts: NW/NE north wall, SW/SE south wall; north "
          f"up). Each panel shades the floor observed by one ceiling camera "
          f"(16 mm, 720x720, mounted at z = {cams['NE'].t[2]:.1f} m), with its "
          f"mount, optical axis and owned aisle; each camera is aimed down one "
          f"rack aisle (NW->B, NE->C, SW->E, SE->F) and observes "
          f"~{np.mean(list(pcts.values())):.0f}% of the navigable floor. "
          f"Visibility is computed by projecting every floor point through the "
          f"deployed camera intrinsics and poses, with racks simplified as "
          f"opaque, infinitely high prisms (a conservative analytical model). "
          f"Jointly the cameras observe {pct_union:.0f}% of the navigable floor "
          f"({pct_multi:.0f}% by two or more) — all four aisles completely, open "
          f"zones A/D/G at {zone_pct['A']:.0f}/{zone_pct['D']:.0f}/"
          f"{zone_pct['G']:.0f}%. Band depths are compressed as in the setup "
          f"figure (widths true to scale); the observed regions are warped with "
          f"the frame and remain point-wise exact.")


if __name__ == "__main__":
    main()
