
from __future__ import annotations

import sys
import pathlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Repo root on the path so `isaac_sim.*` imports resolve when run from anywhere.
_REPO_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf

from isaac_sim.planning.a_star.offline_gate import build_grid_map
from isaac_sim.planning.a_star.planner import A_star, PolylinePath

# ---------------------------------------------------------------------------
# Palette. Restrained on purpose: white ground, one dark neutral for occupied
# space, and exactly ONE hue per pipeline stage (validated categorical slots
# 1-3, assigned in the order the stages are introduced and never cycled).
#
# The "previous stage" ghost wears the PREVIOUS stage's own hue, lightened --
# colour follows the entity. That also leaves grey unused by any path, so the
# configuration-space boundary can be the one grey thing on the panel and never
# be mistaken for a route.
# ---------------------------------------------------------------------------
SURFACE = "#ffffff"
OBSTACLE = "#3c4450"   # occupied cells — dark slate, softer than pure black

C_A = "#2a78d6"        # (a) A*             — slot 1, blue
C_B = "#eb6834"        # (b) + EDF         — slot 2, orange
C_C = "#1baf7a"        # (c) + Cubic Splines— slot 3, aqua
C_A_GHOST = "#9fc2ed"  # C_A at ~45 % over white
C_B_GHOST = "#f6bba4"  # C_B at ~45 % over white

CSPACE = "#a8b0ba"     # configuration-space obstacle boundary — the only grey
INK = "#0b0b0b"        # legend text, start/goal marks

# ---------------------------------------------------------------------------
# The toy scene, defined PORTRAIT (6 m wide x 12 m tall). Hand-built, NOT a real
# warehouse layout -- its only job is to exercise the two behaviours the EDF
# term is there for:
#   1. a 1.8 m doorway  -> shortest-path clips a jamb, clearance-weighted centres
#   2. a 1.8 m corridor past a block -> shortest-path grazes both its corners
# Rects are world AABBs (xmin, ymin, xmax, ymax) in metres, i.e. exactly what
# offline_gate rasterises. Walls are fed as SEGMENTS because a rect whose
# half-extent exceeds cfg.max_obstacle_half_extent_m (3.0 m) is silently dropped
# by the real MapBuilder -- and by its offline mirror.
# ---------------------------------------------------------------------------
ARENA_W, ARENA_H = 6.0, 12.0
T = 0.3  # wall thickness

SCENE_RECTS = [
    # --- outer walls, split so no half-extent exceeds 3.0 m ---
    (-T, -T, 0.0, 4.0), (-T, 4.0, 0.0, 8.0), (-T, 8.0, 0.0, ARENA_H + T),     # left
    (ARENA_W, -T, ARENA_W + T, 4.0), (ARENA_W, 4.0, ARENA_W + T, 8.0),
    (ARENA_W, 8.0, ARENA_W + T, ARENA_H + T),                                 # right
    (0.0, -T, 3.0, 0.0), (3.0, -T, ARENA_W, 0.0),                             # bottom
    (0.0, ARENA_H, 3.0, ARENA_H + T), (3.0, ARENA_H, ARENA_W, ARENA_H + T),   # top
    # --- divider with a 1.8 m doorway centred at x = 3.0 ---
    (0.0, 4.0, 2.1, 4.4),
    (3.9, 4.0, ARENA_W, 4.4),
    # --- wide block leaving a 1.8 m corridor along the right wall ---
    (0.0, 7.2, 4.2, 8.6),
]

START_XY = (1.3, 1.2)
GOAL_XY = (1.6, 11.0)

# Arc-length spacing the drawn paths are sampled at. The DEPLOYED value
# (policy.densify_spacing_m = 0.05 m) -- what the oracle's pure-pursuit tracker
# actually consumes.
DENSIFY_M = 0.05

# Spline KNOT spacing -- the one deliberate departure from the deployed
# pipeline, disclosed in the figure CAPTION (printed to stdout by this script).
#
# The deployed `_cubic_spline_interpolation` fits a knot at EVERY raw A* vertex,
# i.e. every grid_resolution_m = 0.1 m. At that density the spline is an
# interpolant through the entire 8-connected staircase and has essentially no
# freedom to smooth: it moves the path by ~1 cm and only drops peak |dpsi| per
# 5 cm from 45 deg to 28 deg. That is the honest operating point, but it makes
# the MECHANISM of the stage invisible in a figure.
#
# 1.0 m is a round number on the scale of the scene's own features (the 1.8 m
# doorway, the 1.8 m corridor), and it coincides with the robot's cruise turn
# radius v/w = 1.0 m/s / 1.0 rad/s = 1.0 m -- the finest geometry the vehicle
# can actually track at full speed, hence the natural scale at which to show
# what smoothing is FOR. Nothing else about the stage changes: same natural
# cubic, same arc-length parameterisation.
SPLINE_KNOT_M = 1.0

# Figure geometry. FIG_W is chosen to sit close to the printed width of a
# 0.32\textwidth subfigure (~2.1 in on a 6.5 in text block), so LaTeX rescales
# by only ~0.75x and the font sizes below survive into print.
FIG_W = 2.8
PAD_IN = 0.04          # left/right margin, inches
FS_LEGEND = 9.2


def load_deployed_astar_cfg():
    """The real `env/astar/default.yaml`, with the robot interpolation resolved.

    `robot_inflation_m: ${env.robot.inflation_m}` can't resolve standalone (the
    full hydra tree isn't composed here), so we read the robot yaml and stitch
    the one value in. Everything else -- resolution, alpha, buffers, the size cap
    -- is taken verbatim from the deployed config.
    """
    cfg_dir = _REPO_ROOT / "isaac_sim" / "configs"
    astar = OmegaConf.load(cfg_dir / "env" / "astar" / "default.yaml")
    robot = OmegaConf.load(cfg_dir / "env" / "robot" / "nova_carter.yaml")
    astar.robot_inflation_m = float(robot.inflation_m)
    return astar


def search_cfg(base):
    """The deployed cfg as a plain namespace; pipeline stages are run by hand.

    hill_climbing is irrelevant here: at the shipped knobs
    (hill_climb_window_m = 0.1 = grid_resolution_m) the pass is a numerical
    no-op -- half_cells = int(round(0.5)) = 0 gives a 1x1 search window, so the
    argmax is the waypoint itself. Every dataset we have generated therefore
    used inflation + EDF + spline and nothing else.
    """
    return SimpleNamespace(
        grid_resolution_m=float(base.grid_resolution_m),
        robot_inflation_m=float(base.robot_inflation_m),
        world_buffer_m=float(base.world_buffer_m),
        max_obstacle_half_extent_m=float(base.max_obstacle_half_extent_m),
        alpha_clearance=float(base.alpha_clearance),
        hill_climb_window_m=float(base.hill_climb_window_m),
        pipeline=SimpleNamespace(sdf_weighted=True, hill_climbing=False,
                                 cubic_spline=False),
    )


def clearance_along(gmap, samples):
    """EDF value (metres to nearest obstacle) at each sampled (x, y).

    Nearest-cell lookup -- the same discretisation the planner reasons over, so
    the number is the quantity the edge cost actually saw.
    """
    out = []
    for xy in samples:
        r, c = gmap.get_map_coords(xy)
        r = min(max(r, 0), gmap.sdf.shape[0] - 1)
        c = min(max(c, 0), gmap.sdf.shape[1] - 1)
        out.append(float(gmap.sdf[r, c]))
    return np.asarray(out)


def heading_steps_deg(path):
    """|heading change| per DENSIFY_M step, in degrees. shape: (n-1,)"""
    _, yaws = path.sample_with_heading(DENSIFY_M)
    return np.abs(np.diff(np.degrees(np.unwrap(np.asarray(yaws)))))


def render_stage(gmap, cfg, out_dir, *, stem, xy, colour, label,
                 ghost_xy=None, ghost_colour=None, ghost_label=None):
    """Render ONE self-contained portrait figure for a single pipeline stage.

    Layout, top to bottom: the map, then the legend. The stage name is NOT
    drawn -- it belongs in the subfigure's \\caption{}. Each figure does carry its
    own legend so it stands alone as a subfigure --
    which is also why the ghost legend entry can use the previous stage's REAL
    colour here (within one figure there is exactly one ghost, so there is no
    "it varies per panel" caveat to make).
    """
    H, W = gmap.sdf.shape
    xs = gmap.origin[0] + np.arange(W) * gmap.resolution
    ys = gmap.origin[1] + np.arange(H) * gmap.resolution

    # Layout is computed in INCHES and only converted to figure fractions at the
    # end -- axes-relative text placement put the title outside the canvas.
    # N_LEGEND is FIXED at 5 (the most any stage needs) even for (a), which has
    # no ghost entry: all three figures must come out the SAME height or they
    # will not line up as a row of subfigures.
    N_LEGEND = 5
    map_w = FIG_W - 2 * PAD_IN
    map_h = map_w * (ARENA_H + 2 * T) / (ARENA_W + 2 * T)   # equal aspect
    head_h = PAD_IN                                          # no title; just breathing room
    legend_h = 0.12 + 0.19 * N_LEGEND
    fig_h = head_h + map_h + legend_h

    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=SURFACE)
    ax = fig.add_axes([PAD_IN / FIG_W, legend_h / fig_h,
                       map_w / FIG_W, map_h / fig_h])

    # The traversability boundary = {sdf == robot_inflation_m}: the obstacles
    # grown by the robot radius. The point-robot may not enter it.
    ax.contour(xs, ys, gmap.sdf, levels=[cfg.robot_inflation_m],
               colors=[CSPACE], linewidths=1.0, linestyles=[(0, (1.4, 2.4))],
               zorder=2)

    for (x0, y0, x1, y1) in SCENE_RECTS:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor=OBSTACLE, edgecolor="none", zorder=3))

    if ghost_xy is not None:
        ax.plot(ghost_xy[:, 0], ghost_xy[:, 1], color=ghost_colour, linewidth=1.8,
                linestyle=(0, (4.5, 2.5)), zorder=5)
    ax.plot(xy[:, 0], xy[:, 1], color=colour, linewidth=2.2,
            solid_capstyle="round", zorder=6)

    # White ring so the endpoints stay readable wherever a path runs under them.
    ax.plot(*START_XY, marker="o", markersize=6.5, color=INK, linestyle="none",
            markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=8)
    ax.plot(*GOAL_XY, marker="*", markersize=12, color=INK, linestyle="none",
            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=8)

    # The dark wall ring IS the panel frame -- no spines, no ticks, no grid.
    ax.set_xlim(-T, ARENA_W + T)
    ax.set_ylim(-T, ARENA_H + T)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(SURFACE)

    handles = [plt.Line2D([], [], color=colour, lw=2.2, label=label)]
    if ghost_xy is not None:
        handles.append(plt.Line2D([], [], color=ghost_colour, lw=1.8,
                                  ls=(0, (4.5, 2.5)), label=ghost_label))
    handles += [
        plt.Line2D([], [], color=CSPACE, lw=1.0, ls=(0, (1.4, 2.4)),
                   label="inflated obstacles"),
        plt.Line2D([], [], color=INK, marker="o", ls="none", ms=5, label="start"),
        plt.Line2D([], [], color=INK, marker="*", ls="none", ms=9, label="goal"),
    ]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(PAD_IN / FIG_W, (legend_h - 0.03) / fig_h),
               ncol=1, frameon=False, fontsize=FS_LEGEND, labelcolor=INK,
               handlelength=2.6, handletextpad=0.7, labelspacing=0.42,
               borderpad=0.0)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=400, facecolor=SURFACE)
    plt.close(fig)
    return f"{stem}.pdf"


def main():
    base = load_deployed_astar_cfg()
    cfg = search_cfg(base)
    gmap = build_grid_map(SCENE_RECTS, [START_XY, GOAL_XY], cfg)
    planner = A_star(cfg, gmap)

    # _a_star_search is the same call get_a_star_path makes; invoking it directly
    # (rather than flipping the yaml twice) means both searches run against ONE
    # map, so the only difference between (a) and (b) is the edge-cost term.
    raw_plain = planner._a_star_search(START_XY, GOAL_XY, sdf_weighted=False)
    raw_sdf = planner._a_star_search(START_XY, GOAL_XY, sdf_weighted=True)

    path_plain = PolylinePath(raw_plain)
    path_sdf = PolylinePath(raw_sdf)
    spline_knots = path_sdf.sample(SPLINE_KNOT_M)      # see SPLINE_KNOT_M note
    path_spline = planner._cubic_spline_interpolation(spline_knots)

    xy_plain = np.asarray(path_plain.sample(DENSIFY_M))
    xy_sdf = np.asarray(path_sdf.sample(DENSIFY_M))
    xy_spl = np.asarray(path_spline.sample(DENSIFY_M))

    out_dir = _REPO_ROOT / "isaac_sim" / "outputs" / "planner_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (file, subfigure caption, label suffix) -- the stage names live here and
    # are emitted into the LaTeX skeleton, not drawn on the canvas.
    stages = [
        (render_stage(gmap, cfg, out_dir, stem="planner_stage_a_astar",
                      xy=xy_plain, colour=C_A, label="A* path"),
         "A*", "astar"),
        (render_stage(gmap, cfg, out_dir, stem="planner_stage_b_edf",
                      xy=xy_sdf, colour=C_B, label="A* + EDF path",
                      ghost_xy=xy_plain, ghost_colour=C_A_GHOST,
                      ghost_label="A* path"),
         "A* + EDF", "edf"),
        (render_stage(gmap, cfg, out_dir, stem="planner_stage_c_spline",
                      xy=xy_spl, colour=C_C, label="A* + EDF + spline path",
                      ghost_xy=xy_sdf, ghost_colour=C_B_GHOST,
                      ghost_label="A* + EDF path"),
         "A* + EDF + Cubic Splines", "spline"),
    ]
    files = [f for f, _, _ in stages]

    # --- numbers for the caption / body text (not drawn on the figures) ----
    clr_plain = clearance_along(gmap, xy_plain)
    clr_sdf = clearance_along(gmap, xy_sdf)
    clr_spl = clearance_along(gmap, xy_spl)
    d_sdf, d_spl = heading_steps_deg(path_sdf), heading_steps_deg(path_spline)

    print(f"[cfg]     δ={cfg.grid_resolution_m} m  r={cfg.robot_inflation_m} m  "
          f"α={cfg.alpha_clearance}  densify={DENSIFY_M} m")
    print(f"[length]  A* {path_plain.total_length:.2f} m  →  +EDF {path_sdf.total_length:.2f} m  "
          f"(+{100*(path_sdf.total_length/path_plain.total_length-1):.1f}%)")
    print(f"[clear]   min   A* {clr_plain.min():.3f} m  →  +EDF {clr_sdf.min():.3f} m")
    print(f"[clear]   mean  A* {clr_plain.mean():.3f} m  →  +EDF {clr_sdf.mean():.3f} m")
    print(f"[spline]  {len(spline_knots)} knots at {SPLINE_KNOT_M} m "
          f"(FIGURE-ONLY; deployed = {len(raw_sdf)} knots, one per A* vertex)")
    print(f"[heading] peak |Δψ| per {DENSIFY_M} m   polyline {d_sdf.max():.1f}°  →  "
          f"spline {d_spl.max():.1f}°")
    print(f"[heading] RMS  |Δψ|                    polyline {np.sqrt((d_sdf**2).mean()):.2f}°  →  "
          f"spline {np.sqrt((d_spl**2).mean()):.2f}°")

    # Sparser knots let the spline cut corners -- assert it never cuts INSIDE
    # the inflated obstacle set, which would make the drawn path nonsense.
    ok = clr_spl.min() > cfg.robot_inflation_m
    print(f"[spline]  min clearance {clr_spl.min():.3f} m  vs  r = {cfg.robot_inflation_m:g} m  "
          f"→  {'OK' if ok else 'COLLIDES'}")
    if not ok:
        raise SystemExit(f"spline at SPLINE_KNOT_M={SPLINE_KNOT_M} cuts inside the "
                         f"inflated obstacle set — lower SPLINE_KNOT_M")
    for f in files:
        print(f"[out]     {out_dir/f}")

    print()
    print("--- LaTeX skeleton -------------------------------------------------")
    print(r"\begin{figure}[t]\centering")
    for f, name, lab in stages:
        print(rf"  \begin{{subfigure}}{{0.32\textwidth}}\centering"
              rf"\includegraphics[width=\linewidth]{{{f}}}"
              rf"\caption{{{name}}}\label{{fig:plan-{lab}}}\end{{subfigure}}\hfill")
    print(r"  \caption{Stages of the A* oracle path pipeline. " + _caption(cfg) + r"}")
    print(r"\end{figure}")


def _caption(cfg):
    """The disclosure clause, emitted here so caption and code can't drift."""
    return (f"Obstacles are inflated by the robot radius $r = {cfg.robot_inflation_m:g}$\\,m so the "
            f"planner can treat the robot as a point (dotted outline); the clearance term in "
            f"(b) uses $\\alpha = {cfg.alpha_clearance:g}$; the spline in (c) is a natural cubic. "
            f"In (c) the spline is fitted through knots every {SPLINE_KNOT_M:g}\\,m so that the "
            f"stage's effect is legible; the deployed planner places a knot at every A* "
            f"vertex (grid $\\delta = {cfg.grid_resolution_m:g}$\\,m), where the same stage still "
            f"yields a continuous heading but moves the path by only ${{\\sim}}1$\\,cm. Knot "
            f"spacing is a parameter of the spline stage alone --- (a) and (b) show the A* "
            f"polyline through every grid vertex. All other settings are the deployed "
            f"configuration.")


if __name__ == "__main__":
    main()
