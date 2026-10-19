"""Path / map visualisation for the A* + rollout pipeline.

One core renderer — `render_path_map` — draws the env occupancy/SDF map plus
any subset of optional layers (executed robot path, A* plan, per-node heading
arrows, start/goal markers) and then either saves a PNG, returns an RGB array,
or returns the matplotlib Figure (the caller decides via `out_path`/`as_array`).

Both diagnostics in the pipeline are thin callers of it:
  • data-gen's "planning" check  → `save_planning_viz` (the env-unpacking adapter),
  • eval's "rollout" check        → RunRecorder calls `render_path_map` directly,
  • the per-step behaviour panel  → RunRecorder embeds the `as_array` output when
                                    a step has no goal image (pose-conditioned).

All inputs are plain data (a duck-typed map + (x, y) lists), so callers that
hold no live env — the RunRecorder, unit tests — render without booting Isaac
Sim. ALL cosmetics come from the `style` mapping (cfg.viz.path_map — see
configs/viz/default.yaml); there are no hardcoded colours/sizes here, so
every visual knob is tunable from config. Structural drawing constants (zorder
layering, marker shapes, colormap + vmin, quiver geometry) stay in code.
"""
from __future__ import annotations

import math
from pathlib import Path as _PathlibPath


def render_path_map(
    occ_map,
    style,
    *,
    robot_path_xy=None,
    astar_path_xy=None,
    astar_headings=None,
    show_heading_arrows: bool = False,
    start_xy=None,
    goal_xy=None,
    title: str | None = None,
    out_path=None,
    as_array: bool = False,
):
    """Render the map + any subset of {robot path, A* plan, headings, start,
    goal}, then dispose of the result per `out_path` / `as_array`.

    Args:
        occ_map: duck-typed map — needs .occupancy (HxW bool ndarray), .sdf
            (HxW float ndarray or None), .origin ((x0, y0)), .resolution (m/cell).
        style: mapping of ALL cosmetics (cfg.viz.path_map). Required keys —
            backdrop, astar_linestyle, crop_pad_m, figsize, dpi, arrow_len_m,
            occupancy_alpha, astar_linewidth, astar_marker_size,
            astar_marker_edgewidth, quiver_{linewidth,width,headwidth,headlength},
            robot_linewidth, robot_end_size, start_size, goal_size,
            legend_{fontsize,framealpha,loc}, title_fontsize, label_fontsize,
            colors{astar,start,goal,arrow,robot_path,robot_end,marker_edge}.
        robot_path_xy: executed (x, y) path → solid line + end marker. None omits.
        astar_path_xy: A* plan (x, y) polyline. None omits.
        astar_headings: per-point yaws (same length as astar_path_xy); only used
            when show_heading_arrows is True (drawn as a quiver).
        show_heading_arrows: draw the per-node heading quiver (needs astar_headings).
        start_xy / goal_xy: (x, y) → markers. None omits either.
        title: full title string (caller composes). None → no title.
        out_path: if not None → save a PNG there and RETURN the path.
        as_array: elif True → RETURN an RGB np.ndarray (H, W, 3) uint8, for
            embedding into another figure. Else → RETURN the Figure (caller
            owns closing it).

    Returns: the saved path, or an RGB ndarray, or the Figure (see above).
    """
    # Deferred + Agg so importing this module is cheap and we stay headless.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # All cosmetics from config — pulled into locals once for legibility.
    colors = style["colors"]
    backdrop = style["backdrop"]
    astar_linestyle = style["astar_linestyle"]
    crop_pad_m = float(style["crop_pad_m"])
    edge = colors["marker_edge"]

    m = occ_map
    # `is not None and len(...)` rather than truthiness — astar_path_xy / the
    # robot path may arrive as numpy arrays, whose bare truth value is ambiguous.
    has_astar = astar_path_xy is not None and len(astar_path_xy) > 0
    has_robot = robot_path_xy is not None and len(robot_path_xy) > 0

    fig, ax = plt.subplots(1, 1, figsize=tuple(style["figsize"]))

    # ---- Backdrop ----
    H, W = m.occupancy.shape
    extent = (m.origin[0], m.origin[0] + W * m.resolution,
              m.origin[1], m.origin[1] + H * m.resolution)
    use_sdf = (backdrop == "sdf") and (m.sdf is not None)
    if use_sdf:
        ax.imshow(m.sdf, origin="lower", extent=extent, cmap="viridis", vmin=0)
        ax.imshow(m.occupancy, origin="lower", extent=extent, cmap="gray",
                  alpha=float(style["occupancy_alpha"]))
    else:
        # Binary obstacle mask only (occupancy True → dark).
        ax.imshow(m.occupancy, origin="lower", extent=extent, cmap="gray_r")

    # ---- A* plan (+ optional sample dots when solid, + optional heading quiver) ----
    plan_xs: list = []
    plan_ys: list = []
    if has_astar:
        plan_xs = [float(p[0]) for p in astar_path_xy]
        plan_ys = [float(p[1]) for p in astar_path_xy]
        plan_len = float(np.hypot(np.diff(plan_xs), np.diff(plan_ys)).sum())
        ax.plot(plan_xs, plan_ys, color=colors["astar"],
                linewidth=float(style["astar_linewidth"]),
                linestyle=astar_linestyle, zorder=6,
                label=f"A* plan ({plan_len:.1f} m)")
        # When the A* path is the SUBJECT (solid) it gets per-node dots — that's
        # the planning-diagnostic look; as a dashed reference it stays clean.
        if astar_linestyle == "-":
            ax.scatter(plan_xs, plan_ys, color=colors["astar"],
                       s=float(style["astar_marker_size"]), zorder=7,
                       edgecolors=edge, linewidths=float(style["astar_marker_edgewidth"]))
        if show_heading_arrows and astar_headings is not None:
            arrow_len = float(style["arrow_len_m"])
            dx = [arrow_len * math.cos(y) for y in astar_headings]
            dy = [arrow_len * math.sin(y) for y in astar_headings]
            ax.quiver(plan_xs, plan_ys, dx, dy,
                      color=colors["arrow"], edgecolor=edge,
                      linewidth=float(style["quiver_linewidth"]),
                      scale_units="xy", angles="xy", scale=1.0,
                      width=float(style["quiver_width"]),
                      headwidth=float(style["quiver_headwidth"]),
                      headlength=float(style["quiver_headlength"]), zorder=8)

    # ---- Robot executed path (the subject when present) ----
    robot_xs: list = []
    robot_ys: list = []
    if has_robot:
        robot_xs = [float(p[0]) for p in robot_path_xy]
        robot_ys = [float(p[1]) for p in robot_path_xy]
        robot_len = float(np.hypot(np.diff(robot_xs), np.diff(robot_ys)).sum())
        ax.plot(robot_xs, robot_ys, color=colors["robot_path"],
                linewidth=float(style["robot_linewidth"]), zorder=9,
                label=f"robot rollout ({robot_len:.1f} m)")
        ax.scatter([robot_xs[-1]], [robot_ys[-1]], c=colors["robot_end"],
                   s=float(style["robot_end_size"]), marker="X",
                   edgecolors=edge, zorder=10, label="robot end")

    # ---- Start / goal markers ----
    if start_xy is not None:
        ax.scatter([start_xy[0]], [start_xy[1]], c=colors["start"],
                   s=float(style["start_size"]), marker="o",
                   edgecolors=edge, zorder=7, label="start")
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], c=colors["goal"],
                   s=float(style["goal_size"]), marker="*",
                   edgecolors=edge, zorder=7, label="goal")

    # ---- Crop: obstacles + everything drawn, plus pad ----
    occ_r, occ_c = np.where(m.occupancy)
    xmin = m.origin[0] + W * m.resolution
    xmax = m.origin[0]
    ymin = m.origin[1] + H * m.resolution
    ymax = m.origin[1]
    if occ_r.size > 0:
        xmin = min(xmin, m.origin[0] + occ_c.min() * m.resolution)
        xmax = max(xmax, m.origin[0] + occ_c.max() * m.resolution)
        ymin = min(ymin, m.origin[1] + occ_r.min() * m.resolution)
        ymax = max(ymax, m.origin[1] + occ_r.max() * m.resolution)
    xs_all = plan_xs + robot_xs
    ys_all = plan_ys + robot_ys
    if start_xy is not None:
        xs_all.append(float(start_xy[0]))
        ys_all.append(float(start_xy[1]))
    if goal_xy is not None:
        xs_all.append(float(goal_xy[0]))
        ys_all.append(float(goal_xy[1]))
    if xs_all:
        xmin = min([xmin] + xs_all) - crop_pad_m
        xmax = max([xmax] + xs_all) + crop_pad_m
        ymin = min([ymin] + ys_all) - crop_pad_m
        ymax = max([ymax] + ys_all) + crop_pad_m
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=float(style["label_fontsize"]))
    ax.set_ylabel("y (m)", fontsize=float(style["label_fontsize"]))
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        # Legend OUTSIDE the axes (to the right), vertically centred, so it never
        # overlaps the trajectory/map. bbox_to_anchor is axes-relative: (1.02, 0.5)
        # sits just past the right edge. style["legend_loc"] is not used: the
        # outside placement is unconditional so the plot view is never clipped.
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=float(style["legend_fontsize"]),
                  framealpha=float(style["legend_framealpha"]))
    if title:
        ax.set_title(title, fontsize=float(style["title_fontsize"]))
    # Reserve the right ~18% of the figure for the outside legend so it stays
    # inside the figure bounds (the as_array/canvas path below can't use
    # savefig's bbox_inches='tight').
    fig.tight_layout(rect=(0, 0, 0.82, 1))

    # ---- Output: save PNG / return RGB array / return Figure ----
    if out_path is not None:
        out_path = _PathlibPath(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=float(style["dpi"]), bbox_inches="tight")
        plt.close(fig)
        return out_path
    if as_array:
        # Rasterise the canvas to an (H, W, 3) uint8 array (Agg → RGBA buffer,
        # drop alpha). Lets save_robot_behavior imshow the map into a sub-panel.
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        arr = rgba[..., :3].copy()
        plt.close(fig)
        return arr
    return fig


def save_planning_viz(env, out_path, style) -> _PathlibPath:
    """Data-gen adapter: unpack the live env (map, start, goal, planned Path +
    sampled headings) and render the planning diagnostic via `render_path_map`.

    Kept as a named function because data_gen calls it with an `env`; the actual
    drawing lives in render_path_map so the planning and rollout views can't
    drift apart. Headings are sampled at `style.astar_sample_spacing_m` and drawn
    as the headings `_record_image_graph` would attach
    (analytic tangent interior / next-node secant at endpoints — see SplinePath
    in a_star/planner.py). The A* path is drawn SOLID here (it's the subject —
    no rollout to compare against yet), per style.astar_linestyle.

    `style` is cfg.viz.path_map; `style.heading_arrows` toggles the quiver.
    """
    m = env.a_star.map
    sx, sy = env.template_cfg.start_xy
    gx, gy = env.get_effective_goal_pose()["xy"]

    planned_path = env.get_a_start_path((sx, sy), (gx, gy))
    positions, yaws = planned_path.sample_with_heading(float(style["astar_sample_spacing_m"]))

    name = getattr(env.template_cfg, "name", "scene")
    spline_on = bool(env.env_cfg.astar.pipeline.cubic_spline)
    hill_on = bool(env.env_cfg.astar.pipeline.hill_climbing)
    title = (
        f"{name} — planned A* path + sampled headings\n"
        f"path_length={planned_path.total_length:.2f} m   "
        f"cubic_spline={'on' if spline_on else 'off'}   "
        f"hill_climbing={'on' if hill_on else 'off'}"
    )

    return render_path_map(
        m,
        style,
        astar_path_xy=positions,
        astar_headings=yaws,
        show_heading_arrows=bool(style["heading_arrows"]),
        start_xy=(sx, sy),
        goal_xy=(gx, gy),
        title=title,
        out_path=out_path,
    )
