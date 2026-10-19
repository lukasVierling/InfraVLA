"""Robot-local action-chunk visualisation.

One shared function — `plot_action_chunk` — draws an OmniVLA action chunk in the
robot-local normalized frame: the predicted waypoints (blue) and, when available,
the ground-truth / target chunk (red), in the paper's "forward up, left left"
convention. It draws onto a caller-provided matplotlib `ax` (both call sites
embed it in a multi-panel figure), the way `planning_viz.render_path_map` is the
single source for the world-frame map. Used by RunRecorder.save_robot_behavior
(eval: predicted only, plus the goal-pose star for pose-conditioned steps;
style = cfg.viz.action_chunk).

Pure numpy + matplotlib; no Isaac Sim deps. ALL cosmetics come from the `style`
mapping — no hardcoded colours/sizes/fonts here. Structural constants (marker
shapes, the (-y, x) plotting convention) stay in code.
"""
from __future__ import annotations

import numpy as np


def _local_to_plot(xy):
    """Map a robot-local (x=forward, y=left) waypoint array to plot axes.

    Paper convention (OmniVLA's visualize_train): plot (-y) horizontally and (x)
    vertically, so forward points UP and left points LEFT. A leading (0, 0) is
    prepended so the polyline starts at the robot.

    Args: xy — (N, >=2) array; only columns 0 (x) and 1 (y) are read.
    Returns: (horiz, vert) 1-D arrays of length N+1.
    """
    a = np.asarray(xy, dtype=float)
    x = a[:, 0]  # forward
    y = a[:, 1]  # left
    horiz = np.insert(-y, 0, 0.0)  # x-axis: left(+y) plotted to the left
    vert = np.insert(x, 0, 0.0)    # y-axis: forward(+x) plotted up
    return horiz, vert


def plot_action_chunk(
    ax,
    pred_xy,
    style,
    *,
    gt_xy=None,
    goal_pose_loc_norm=None,
    modality_label=None,
    title=None,
):
    """Draw a robot-local action chunk onto `ax`.

    Args:
        ax: matplotlib Axes to draw onto.
        pred_xy: (N, >=2) predicted waypoints, local frame (x=fwd, y=left) → pred colour.
        style: mapping of ALL cosmetics (cfg.viz.action_chunk). Required keys —
            equal_aspect, xlim, ylim, linewidth, markersize, origin_size,
            goal_star_size, annotate_offset, grid_alpha, title_fontsize,
            label_fontsize, tick_fontsize, legend_fontsize, annotate_fontsize,
            colors{pred,gt,goal,origin}.
        gt_xy: (N, >=2) ground-truth/target waypoints → gt colour. None to omit
            (e.g. closed-loop eval has no per-step GT).
        goal_pose_loc_norm: (>=2,) normalized goal in the same local frame → star.
            None to omit (e.g. image-only steps have no pose goal).
        modality_label: short string annotated bottom-right (e.g. "pose only").
        title: axis title. None to omit.

    Returns: ax (for chaining).
    """
    colors = style["colors"]
    lw = float(style["linewidth"])
    ms = float(style["markersize"])

    # GT first so the prediction draws ON TOP of it.
    if gt_xy is not None and len(gt_xy) > 0:
        h, v = _local_to_plot(gt_xy)
        ax.plot(h, v, "-o", color=colors["gt"], linewidth=lw,
                markersize=ms, label="target (GT)")

    h, v = _local_to_plot(pred_xy)
    ax.plot(h, v, "-o", color=colors["pred"], linewidth=lw,
            markersize=ms, label="predicted")

    # Robot at the current step sits at the origin.
    ax.scatter([0.0], [0.0], c=colors["origin"], s=float(style["origin_size"]),
               marker="s", zorder=5, label="robot (t)")

    # Goal-pose star, same (-y, x) convention, when a pose goal exists.
    if goal_pose_loc_norm is not None:
        gp = np.asarray(goal_pose_loc_norm, dtype=float)
        ax.plot(-gp[1], gp[0], marker="*", color=colors["goal"],
                markersize=float(style["goal_star_size"]), zorder=6, label="goal")

    if modality_label is not None:
        # Bottom-right annotation.
        ax.annotate(modality_label, xy=(1.0, 0.0), xycoords="axes fraction",
                    xytext=tuple(style["annotate_offset"]), textcoords="offset points",
                    ha="right", fontsize=float(style["annotate_fontsize"]))

    if bool(style["equal_aspect"]):
        ax.axis("equal")
    else:
        ax.set_xlim(*tuple(style["xlim"]))
        ax.set_ylim(*tuple(style["ylim"]))
    ax.set_xlabel("← left   lateral   right →", fontsize=float(style["label_fontsize"]))
    ax.set_ylabel("forward →", fontsize=float(style["label_fontsize"]))
    ax.tick_params(axis="both", labelsize=float(style["tick_fontsize"]))
    ax.grid(True, alpha=float(style["grid_alpha"]))
    ax.legend(loc="best", fontsize=float(style["legend_fontsize"]))
    if title is not None:
        ax.set_title(title, fontsize=float(style["title_fontsize"]))
    return ax
