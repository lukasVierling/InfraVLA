
from __future__ import annotations

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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

from isaac_sim.cam_to_map.camera_model import CameraModel, intrinsics_matrix
from isaac_sim.scenes._common import (
    WAREHOUSE_2X3_AISLE_CLEAR_W,
    WAREHOUSE_2X3_COLUMN_XS,
    WAREHOUSE_2X3_INTERIOR_X,
    WAREHOUSE_2X3_INTERIOR_Y,
    WAREHOUSE_2X3_NAV_X,
    WAREHOUSE_2X3_NAV_Y,
    WAREHOUSE_2X3_ZONES,
)
from viz._lib.warehouse_2x3_schematic import (
    EDGE_LW,
    FS_LEGEND,
    INK,
    INK_2,
    MARGIN_TINT,
    SEAM,
    SURFACE,
    _Y_EDGES_TRUE,
    add_legend,
    build_canvas,
    warp_y,
    zone_letters,
)

GRID_RES = 0.025
RDP_TOL_M = 0.025
MIN_RING_AREA = 0.05
OWNED_ZONE = {"NW": "B", "NE": "C", "SW": "E", "SE": "F"}
CAM_NAMES = ("NW", "NE", "SW", "SE")

C_COV = "#f7c948"
C_COV_EDGE = "#c79a18"
COV_ALPHA = 0.30
AXIS_ARROW_M = 1.8

WALL_T = 0.35
COL_RACK, COL_FLOOR, COL_MARGIN = "3C4450", "F3F6F9", "E6EAEE"
COL_INK, COL_INK2 = "0B0B0B", "52514E"
COL_COV, COL_COVEDGE = "F7C948", "C79A18"

# Sequential redundancy colours make overlap count directly readable. Camera
# outlines use the colourblind-safe Okabe--Ito categorical palette.
COUNT_COLOURS = {
    1: "#fff3c4",
    2: "#f7c948",
    3: "#e59a17",
    4: "#a95c00",
}
CAM_COLOURS = {
    "NW": "#0072b2",
    "NE": "#009e73",
    "SW": "#cc79a7",
    "SE": "#d55e00",
}

OUT_DIR = _REPO_ROOT / "isaac_sim" / "outputs" / "cctv_coverage_tikz"

_RACK_HALF_DEPTH = (
    abs(WAREHOUSE_2X3_COLUMN_XS[0] - WAREHOUSE_2X3_COLUMN_XS[1])
    - WAREHOUSE_2X3_AISLE_CLEAR_W
) / 2.0
_ROW_SPANS = (WAREHOUSE_2X3_ZONES["B"].y_range, WAREHOUSE_2X3_ZONES["E"].y_range)
RACK_RECTS = [
    (col - _RACK_HALF_DEPTH, y0, col + _RACK_HALF_DEPTH, y1)
    for col in WAREHOUSE_2X3_COLUMN_XS
    for (y0, y1) in _ROW_SPANS
]


def load_bigcam(name: str) -> CameraModel:
    """One cctv_2x3_<name>_bigcam.yaml -> CameraModel."""
    path = (_REPO_ROOT / "isaac_sim" / "configs" / "env" / "camera"
            / f"cctv_2x3_{name.lower()}_bigcam.yaml")
    cfg = yaml.safe_load(path.read_text())
    w_px, h_px = cfg["resolution"]
    K = intrinsics_matrix(
        cfg["intrinsics"]["focal_length_mm"],
        cfg["intrinsics"]["horizontal_aperture_mm"],
        cfg["intrinsics"]["vertical_aperture_mm"],
        w_px, h_px,
    )
    M = np.array(cfg["extrinsics"]["world_transform_matrix"], dtype=np.float64)
    # USD Gf.Matrix4d is ROW-vector convention: rows 0..2 = local axes in world.
    R_world_from_usd = M[:3, :3].T
    R_wc = R_world_from_usd @ np.diag([1.0, -1.0, -1.0])   # USD cam -> OpenCV cam
    cam = CameraModel(K=K, R_wc=R_wc, t=M[3, :3].copy())
    assert cam.R_wc[2, 2] < 0, f"{name}: camera does not look down"
    assert cam.t[2] > 5.0, f"{name}: implausible mount height {cam.t[2]:.2f} m"
    return cam


def segment_blocked_by_rects(cam_xy: np.ndarray, pts_xy: np.ndarray, rects) -> np.ndarray:
    """Liang-Barsky slab test: True where segment cam->pt crosses any rack rect."""
    d = pts_xy - cam_xy[None, :]
    blocked = np.zeros(len(pts_xy), dtype=bool)
    for (x0, y0, x1, y1) in rects:
        t_lo = np.zeros(len(pts_xy))
        t_hi = np.ones(len(pts_xy))
        for axis, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
            da = d[:, axis]
            pa = cam_xy[axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (lo - pa) / da
                tb = (hi - pa) / da
            slab_lo = np.minimum(ta, tb)
            slab_hi = np.maximum(ta, tb)
            parallel = da == 0.0
            inside = (lo <= pa) & (pa <= hi)
            slab_lo = np.where(parallel, (-np.inf if inside else np.inf), slab_lo)
            slab_hi = np.where(parallel, (np.inf if inside else -np.inf), slab_hi)
            t_lo = np.maximum(t_lo, slab_lo)
            t_hi = np.minimum(t_hi, slab_hi)
        blocked |= t_lo < t_hi
    return blocked


def visibility_mask(cam: CameraModel, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Boolean (H, W) floor mask per the 3-part definition in the module docstring."""
    h_img = w_img = 720
    pts = np.stack([X, Y, np.zeros_like(X)], axis=-1)     # (H, W, 3), floor z=0
    uv, z_cam = cam.project_points(pts)
    in_image = (
        (z_cam > 1e-6)
        & (uv[..., 0] >= 0.0) & (uv[..., 0] < w_img)
        & (uv[..., 1] >= 0.0) & (uv[..., 1] < h_img)
    )
    flat = np.stack([X.ravel(), Y.ravel()], axis=-1)      # (N, 2)
    occluded = segment_blocked_by_rects(cam.t[:2], flat, RACK_RECTS)
    return in_image & ~occluded.reshape(X.shape)


def _shoelace(ring: np.ndarray) -> float:
    """Signed polygon area (positive = counter-clockwise)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def mask_to_rings(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> list[np.ndarray]:
    """Boolean raster -> closed contour rings in world coords (dense vertices)."""
    if not mask.any():
        return []
    res = xs[1] - xs[0]
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2))
    padded[1:-1, 1:-1] = mask.astype(float)
    xs_p = np.concatenate([[xs[0] - res], xs, [xs[-1] + res]])
    ys_p = np.concatenate([[ys[0] - res], ys, [ys[-1] + res]])
    fig, ax = plt.subplots()
    cs = ax.contour(xs_p, ys_p, padded, levels=[0.5])
    rings = [seg for seg in cs.allsegs[0] if abs(_shoelace(seg)) >= MIN_RING_AREA]
    plt.close(fig)
    return rings


def rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker polyline simplification (iterative, endpoints kept)."""
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        seg = points[i1] - points[i0]
        seg_len = np.hypot(*seg)
        rel = points[i0 + 1:i1] - points[i0]
        if seg_len < 1e-12:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            dist = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / seg_len
        k = int(np.argmax(dist))
        if dist[k] > tol:
            split = i0 + 1 + k
            keep[split] = True
            stack.append((i0, split))
            stack.append((split, i1))
    return points[keep]


def to_frame(pts: np.ndarray, frame: str) -> np.ndarray:
    """(N, 2) world points -> drawing coords ('true' = identity, 'warped' = warp_y)."""
    if frame == "true":
        return pts
    out = np.asarray(pts, dtype=float).copy()
    out[:, 1] = [warp_y(y) for y in out[:, 1]]
    return out


def frame_rings(rings: list[np.ndarray], frame: str, tol: float) -> list[np.ndarray]:
    """Warp then simplify each dense ring, so band-boundary kinks land exactly."""
    return [rdp(to_frame(r, frame), tol) for r in rings]


def compound_path(rings: list[np.ndarray]) -> MplPath:
    """Rings -> one matplotlib Path; opposite winding makes shadow islands holes."""
    verts, codes = [], []
    for r in rings:
        verts.extend(r)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)


def camera_axis_xy(cam: CameraModel) -> np.ndarray:
    """Ground-plane unit azimuth of the optical axis (3rd column of R_wc)."""
    fwd = cam.R_wc[:, 2]
    return fwd[:2] / np.linalg.norm(fwd[:2])


# ---------------------------------------------------------------------------
# The thesis figure: schematic canvas + coverage + camera marks
# ---------------------------------------------------------------------------

def render_coverage_figure(rings_per_cam, rings_per_count, cams, stem: str) -> None:
    """Render the intuitive warped figure: exact redundancy + camera identity."""
    fig, ax, fig_h, leg_h, _ = build_canvas(legend_h_in=1.22)

    for yb in _Y_EDGES_TRUE[1:-1]:
        ax.plot(WAREHOUSE_2X3_NAV_X, [warp_y(yb)] * 2, color=SEAM, lw=1.1,
                ls=(0, (1.2, 2.6)), zorder=1.5)

    # Draw exact camera count as solid sequential colour. This is substantially
    # easier to decode than estimating overlap from alpha-blended amber.
    for count in range(1, 5):
        if not rings_per_count[count]:
            continue
        ax.add_patch(PathPatch(
            compound_path(rings_per_count[count]),
            facecolor=COUNT_COLOURS[count],
            edgecolor="none",
            zorder=2,
        ))

    # A thin categorical outline preserves camera identity without turning the
    # large filled areas into four competing hues.
    for name in CAM_NAMES:
        ax.add_patch(PathPatch(
            compound_path(rings_per_cam[name]),
            facecolor="none",
            edgecolor=CAM_COLOURS[name],
            lw=EDGE_LW * 0.85,
            alpha=0.9,
            zorder=2.5,
        ))

    zone_letters(ax, avoid_seal=False)

    for name in CAM_NAMES:
        cam = cams[name]
        cx, cy = cam.t[0], warp_y(cam.t[1])
        az = camera_axis_xy(cam)
        tip = to_frame(np.array([cam.t[:2] + AXIS_ARROW_M * az]), "warped")[0]
        cam_colour = CAM_COLOURS[name]
        ax.annotate("", xy=(tip[0], tip[1]), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=cam_colour, lw=1.3,
                                    shrinkA=2.0, shrinkB=0.0), zorder=6)
        ax.plot([cx], [cy], marker="o", ms=5.0, color=cam_colour,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=7)
        west = cam.t[0] < 0.5 * sum(WAREHOUSE_2X3_NAV_X)
        ax.annotate(name, xy=(cx, cy), xytext=(-9 if west else 9, 0),
                    textcoords="offset points", fontsize=FS_LEGEND, color=cam_colour,
                    fontweight="semibold", va="center",
                    ha="right" if west else "left", zorder=7)

    handles = [
        Patch(facecolor=COUNT_COLOURS[count], edgecolor=INK_2,
              label=f"{count} camera" + ("" if count == 1 else "s"))
        for count in range(1, 5)
    ]
    handles.extend(
        Line2D([], [], color=CAM_COLOURS[name], lw=1.5,
               marker="o", ms=4.0, label=f"{name} footprint")
        for name in CAM_NAMES
    )
    handles.extend([
        Patch(facecolor=MARGIN_TINT, edgecolor="none",
              label="outside nav. bounds"),
        Patch(facecolor="#" + COL_RACK, edgecolor="none",
              label="rack (infinite-height)"),
    ])
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, (leg_h - 0.02) / fig_h),
        ncol=2,
        frameon=False,
        fontsize=FS_LEGEND - 0.6,
        labelcolor=INK,
        handlelength=1.5,
        handletextpad=0.55,
        columnspacing=1.05,
        labelspacing=0.28,
        borderpad=0.0,
        title="Analytical ground visibility\n(racks: opaque, infinitely high)",
        title_fontsize=FS_LEGEND - 0.2,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=400, facecolor=SURFACE)
    plt.close(fig)
    print(f"[out]      {OUT_DIR}/{stem}.png (+ .pdf)")


# ---------------------------------------------------------------------------
# TikZ emission
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _tikz_ring(ring: np.ndarray) -> str:
    pts = " -- ".join(f"({_fmt(p[0])},{_fmt(p[1])})" for p in ring[:-1])
    return f"{pts} -- cycle"


def emit_tikz(out_path: Path, frame: str, rings_per_cam, rings_per_count,
              cams, unit_cm: float) -> None:
    ix0, ix1 = WAREHOUSE_2X3_INTERIOR_X
    iy0, iy1 = WAREHOUSE_2X3_INTERIOR_Y
    nx0, nx1 = WAREHOUSE_2X3_NAV_X

    def F(x: float, y: float) -> str:
        p = to_frame(np.array([[x, y]]), frame)[0]
        return f"({_fmt(p[0])},{_fmt(p[1])})"

    L: list[str] = []
    L.append("% AUTO-GENERATED by viz/_lib/cctv_coverage_tikz.py -- do not hand-edit")
    L.append(f"% variant: {frame} frame"
             + ("" if frame == "true"
                else " (per-band y-compression of warehouse_2x3_schematic.py; disclose in caption)"))
    L.append("% Compile standalone, or lift the tikzpicture into the thesis.")
    L.append("% Occlusion model: rack footprints are opaque, infinitely high prisms.")
    L.append("% Coverage is a conservative analytical approximation, not mesh ray tracing.")
    L.append("% Resize by editing x=/y= below (1 unit = 1 m"
             + (" true scale)" if frame == "true" else " in x; y is band-warped)"))
    L.append(r"\documentclass[tikz,border=2mm]{standalone}")
    L.append(r"\usetikzlibrary{arrows.meta}")
    for tok, hexc in (("rack", COL_RACK), ("floor", COL_FLOOR), ("margin", COL_MARGIN),
                      ("ink", COL_INK), ("inktwo", COL_INK2),
                      ("cov", COL_COV), ("covedge", COL_COVEDGE)):
        L.append(rf"\definecolor{{{tok}}}{{HTML}}{{{hexc}}}")
    for count, colour in COUNT_COLOURS.items():
        L.append(rf"\definecolor{{vis{count}}}{{HTML}}{{{colour.lstrip('#').upper()}}}")
    for name, colour in CAM_COLOURS.items():
        L.append(rf"\definecolor{{cam{name.lower()}}}{{HTML}}{{{colour.lstrip('#').upper()}}}")
    L.append(r"\begin{document}")
    L.append(rf"\begin{{tikzpicture}}[x={unit_cm}cm,y={unit_cm}cm]")

    L.append("% ---- layer 1: floor + non-navigable margins + perimeter walls ----")
    L.append(rf"\fill[floor] {F(ix0, iy0)} rectangle {F(ix1, iy1)};")
    L.append(rf"\fill[margin] {F(ix0, iy0)} rectangle {F(nx0, iy1)};")
    L.append(rf"\fill[margin] {F(nx1, iy0)} rectangle {F(ix1, iy1)};")
    L.append(rf"\fill[rack, even odd rule] {F(ix0 - WALL_T, iy0 - WALL_T)} rectangle "
             rf"{F(ix1 + WALL_T, iy1 + WALL_T)} {F(ix0, iy0)} rectangle {F(ix1, iy1)};")

    L.append("% ---- layer 2: analytical redundancy, filled by exact camera count ----")
    L.append("% Delete this layer for a pure camera-position figure.")
    for count in range(1, 5):
        if not rings_per_count[count]:
            continue
        body = "  ".join(_tikz_ring(r) for r in rings_per_count[count])
        L.append(rf"\fill[vis{count}, even odd rule] % visible by exactly {count} camera(s)")
        L.append(f"  {body};")

    L.append("% ---- layer 3: individual camera-footprint outlines ----")
    for name in CAM_NAMES:
        for r in rings_per_cam[name]:
            L.append(rf"\draw[cam{name.lower()}, line width=0.45pt, opacity=0.9] "
                     rf"{_tikz_ring(r)};")

    L.append("% ---- layer 4: racks ----")
    for (x0, y0, x1, y1) in RACK_RECTS:
        L.append(rf"\fill[rack, rounded corners=1.5pt] {F(x0, y0)} rectangle {F(x1, y1)};")

    L.append("% ---- layer 5: zone letters ----")
    for zname, zone in WAREHOUSE_2X3_ZONES.items():
        cx, cy = zone.center
        L.append(rf"\node[inktwo, font=\scriptsize] at {F(cx, cy)} {{{zname}}};")

    L.append("% ---- layer 6: camera mounts, view directions, labels ----")
    for name in CAM_NAMES:
        cam = cams[name]
        cx, cy = cam.t[0], cam.t[1]
        az = camera_axis_xy(cam)
        tip_x, tip_y = cx + AXIS_ARROW_M * az[0], cy + AXIS_ARROW_M * az[1]
        north = cy > 0
        anchor = "south" if north else "north"
        ly = iy1 + WALL_T + 0.35 if north else iy0 - WALL_T - 0.35
        cam_style = f"cam{name.lower()}"
        L.append(rf"\draw[{cam_style}, -{{Stealth[length=3.5pt]}}, line width=0.9pt] "
                 rf"{F(cx, cy)} -- {F(tip_x, tip_y)};   % {name} optical axis")
        L.append(rf"\fill[{cam_style}] {F(cx, cy)} circle[radius=2.4pt];")
        L.append(rf"\node[{cam_style}, font=\small\bfseries, anchor={anchor}] "
                 rf"at {F(cx, ly)} {{{name}}};")

    L.append("% ---- layer 7: north arrow" + (" + scale bar ----" if frame == "true" else " ----"))
    if frame == "true":
        ax_, ay = ix0 - WALL_T, iy1 + WALL_T + 0.9
        arrow_start = F(ax_, ay)
        arrow_end = F(ax_, ay + 2.2)
        arrow_label = F(ax_ + 0.25, ay + 1.1)
    else:
        # `warp_y` intentionally clamps outside the warehouse bands, so draw
        # this decoration directly in warped-frame coordinates inside the map.
        sy1 = to_frame(np.array([[0.0, iy1]]), frame)[0, 1]
        arrow_start = f"({_fmt(ix1 - 0.55)},{_fmt(sy1 - 1.45)})"
        arrow_end = f"({_fmt(ix1 - 0.55)},{_fmt(sy1 - 0.35)})"
        arrow_label = f"({_fmt(ix1 - 0.30)},{_fmt(sy1 - 0.90)})"
    L.append(rf"\draw[ink, -{{Stealth[length=4pt]}}, line width=0.8pt] "
             rf"{arrow_start} -- {arrow_end};")
    L.append(rf"\node[ink, font=\scriptsize, anchor=west] at {arrow_label} {{N}};")
    if frame == "true":
        bx, by = ix0, iy0 - WALL_T - 1.4
        L.append(rf"\draw[inktwo, line width=0.8pt] {F(bx, by)} -- {F(bx + 5.0, by)} "
                 rf"node[midway, below=1pt, font=\scriptsize] {{5\,m}};")
        L.append(rf"\foreach \p in {{{_fmt(bx)},{_fmt(bx + 5.0)}}} "
                 rf"\draw[inktwo, line width=0.8pt] (\p,{_fmt(by - 0.13)}) -- (\p,{_fmt(by + 0.13)});")

    # Explicit legend below the map. Drawing the primitives directly keeps the
    # generated TikZ self-contained and robust when copied outside the thesis.
    sy0 = to_frame(np.array([[0.0, iy0]]), frame)[0, 1]
    panel_x0, panel_x1 = ix0, ix1
    panel_y1, panel_y0 = sy0 - 3.0, sy0 - 7.2
    item_xs = [panel_x0 + 0.55, panel_x0 + 3.85,
               panel_x0 + 7.15, panel_x0 + 10.45]
    L.append("% ---- layer 8: legend ----")
    L.append(rf"\filldraw[fill=white, draw=inktwo, line width=0.35pt, rounded corners=2pt] "
             rf"({_fmt(panel_x0)},{_fmt(panel_y0)}) rectangle "
             rf"({_fmt(panel_x1)},{_fmt(panel_y1)});")
    L.append(rf"\node[ink, anchor=west, font=\scriptsize\bfseries] at "
             rf"({_fmt(panel_x0 + 0.45)},{_fmt(panel_y1 - 0.55)}) "
             rf"{{Analytical ground visibility}};")
    count_y = panel_y1 - 1.50
    for count, item_x in zip(range(1, 5), item_xs):
        L.append(rf"\filldraw[vis{count}, draw=inktwo, line width=0.25pt] "
                 rf"({_fmt(item_x)},{_fmt(count_y - 0.22)}) rectangle "
                 rf"({_fmt(item_x + 0.48)},{_fmt(count_y + 0.22)});")
        suffix = "camera" if count == 1 else "cameras"
        L.append(rf"\node[inktwo, anchor=west, font=\tiny] at "
                 rf"({_fmt(item_x + 0.65)},{_fmt(count_y)}) {{{count} {suffix}}};")
    cam_y = panel_y1 - 2.52
    for name, item_x in zip(CAM_NAMES, item_xs):
        L.append(rf"\draw[cam{name.lower()}, line width=0.9pt] "
                 rf"({_fmt(item_x)},{_fmt(cam_y)}) -- "
                 rf"({_fmt(item_x + 0.5)},{_fmt(cam_y)});")
        L.append(rf"\node[inktwo, anchor=west, font=\tiny] at "
                 rf"({_fmt(item_x + 0.65)},{_fmt(cam_y)}) {{{name} footprint}};")
    rack_y = panel_y1 - 3.48
    L.append(rf"\fill[rack] ({_fmt(panel_x0 + 0.55)},{_fmt(rack_y - 0.22)}) rectangle "
             rf"({_fmt(panel_x0 + 1.03)},{_fmt(rack_y + 0.22)});")
    L.append(rf"\node[inktwo, anchor=west, font=\tiny] at "
             rf"({_fmt(panel_x0 + 1.2)},{_fmt(rack_y)}) "
             rf"{{Rack footprint (opaque, infinitely high occluder)}};")

    L.append(r"\end{tikzpicture}")
    L.append(r"\end{document}")
    out_path.write_text("\n".join(L) + "\n")
    print(f"[out]      {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cams = {name: load_bigcam(name) for name in CAM_NAMES}

    ix0, ix1 = WAREHOUSE_2X3_INTERIOR_X
    iy0, iy1 = WAREHOUSE_2X3_INTERIOR_Y
    xs = np.arange(ix0 + GRID_RES / 2, ix1, GRID_RES)
    ys = np.arange(iy0 + GRID_RES / 2, iy1, GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    masks = {name: visibility_mask(cams[name], X, Y) for name in CAM_NAMES}
    union = np.zeros_like(X, dtype=bool)
    for m in masks.values():
        union |= m

    in_rack = np.zeros_like(X, dtype=bool)
    for (x0, y0, x1, y1) in RACK_RECTS:
        in_rack |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    nav = ((X >= WAREHOUSE_2X3_NAV_X[0]) & (X <= WAREHOUSE_2X3_NAV_X[1])
           & (Y >= WAREHOUSE_2X3_NAV_Y[0]) & (Y <= WAREHOUSE_2X3_NAV_Y[1])
           & ~in_rack)
    n_nav = int(nav.sum())
    print(f"[coverage] navigable floor: {n_nav * GRID_RES**2:.1f} m^2 "
          f"({n_nav} cells @ {GRID_RES} m)")
    for name in CAM_NAMES:
        print(f"[coverage]   {name}: {100.0 * (masks[name] & nav).sum() / n_nav:5.1f} %")
    pct_union = 100.0 * (union & nav).sum() / n_nav
    n_multi = int((nav & (sum(m.astype(int) for m in masks.values()) >= 2)).sum())
    print(f"[coverage]   union: {pct_union:5.1f} %   "
          f"(>=2 cameras: {100.0 * n_multi / n_nav:4.1f} %)")

    zone_pct = {}
    for zname, zone in WAREHOUSE_2X3_ZONES.items():
        zmask = (nav & (X >= zone.x_range[0]) & (X <= zone.x_range[1])
                 & (Y >= zone.y_range[0]) & (Y <= zone.y_range[1]))
        zone_pct[zname] = 100.0 * (union & zmask).sum() / max(int(zmask.sum()), 1)
    print("[coverage]   union per zone: "
          + "  ".join(f"{z}={zone_pct[z]:.0f}%" for z in "ABCDEFG"))

    for name, zname in OWNED_ZONE.items():
        zone = WAREHOUSE_2X3_ZONES[zname]
        zmask = (nav & (X >= zone.x_range[0]) & (X <= zone.x_range[1])
                 & (Y >= zone.y_range[0]) & (Y <= zone.y_range[1]))
        own = 100.0 * (masks[name] & zmask).sum() / max(int(zmask.sum()), 1)
        print(f"[own-aisle] {name} -> {zname}: {own:5.1f} %")
        assert own >= 60.0, f"{name} covers only {own:.1f}% of its aisle {zname}"

    dense_rings = {name: mask_to_rings(masks[name], xs, ys) for name in CAM_NAMES}
    coverage_count = np.sum(np.stack([masks[name] for name in CAM_NAMES]), axis=0)
    dense_count_rings = {
        count: mask_to_rings(coverage_count == count, xs, ys)
        for count in range(1, 5)
    }
    for name in CAM_NAMES:
        poly_area = sum(_shoelace(r) for r in dense_rings[name])
        raster_area = masks[name].sum() * GRID_RES**2
        err = abs(abs(poly_area) - raster_area) / raster_area
        assert err < 0.02, f"{name}: polygon/raster area mismatch {err:.1%}"
    print("[self-check] polygon areas match rasters (<2%) for all 4 cameras")

    warped = {n: frame_rings(dense_rings[n], "warped", RDP_TOL_M * 0.445)
              for n in CAM_NAMES}
    true = {n: frame_rings(dense_rings[n], "true", RDP_TOL_M) for n in CAM_NAMES}
    warped_count = {
        count: frame_rings(dense_count_rings[count], "warped", RDP_TOL_M * 0.445)
        for count in range(1, 5)
    }
    true_count = {
        count: frame_rings(dense_count_rings[count], "true", RDP_TOL_M)
        for count in range(1, 5)
    }

    render_coverage_figure(
        warped,
        warped_count,
        cams,
        "wh2x3_cctv_redundancy",
    )
    emit_tikz(
        OUT_DIR / "wh2x3_cctv_redundancy_true.tex",
        "true",
        true,
        true_count,
        cams,
        0.30,
    )
    emit_tikz(
        OUT_DIR / "wh2x3_cctv_redundancy_warped.tex",
        "warped",
        warped,
        warped_count,
        cams,
        0.55,
    )

    print()
    print("--- caption clause (edit to taste) ---")
    print(f"Fixed-camera rig of the 2x3 warehouse (layout, rails and zone letters as "
          f"in the setup figure). Four ceiling CCTVs (16 mm, 720x720, mounted at "
          f"z = {cams['NE'].t[2]:.1f} m) each look down one rack aisle "
          f"(NW->B, NE->C, SW->E, SE->F); coloured arrows and outlines identify "
          f"each camera and its footprint. Sequential shading gives the exact number "
          f"of cameras covering each floor region. Visibility is computed from the "
          f"deployed intrinsics and poses while simplifying racks as opaque, "
          f"infinitely high rectangular prisms. Under this conservative analytical "
          f"model the cameras jointly cover {pct_union:.0f}% of the navigable floor — "
          f"all four aisles B/C/E/F completely "
          f"({zone_pct['B']:.0f}/{zone_pct['C']:.0f}/{zone_pct['E']:.0f}/"
          f"{zone_pct['F']:.0f}%), open zones A/D/G at {zone_pct['A']:.0f}/"
          f"{zone_pct['D']:.0f}/{zone_pct['G']:.0f}%. Band depths are compressed as "
          f"in the setup figure (widths true to scale); the observed regions are "
          f"warped with the frame and remain point-wise exact.")


if __name__ == "__main__":
    main()
