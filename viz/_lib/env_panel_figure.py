
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib.patheffects as path_effects
from PIL import Image, ImageOps
from omegaconf import OmegaConf

HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
ISAAC_SIM_DIR = REPO_ROOT / "isaac_sim"
for _p in (str(REPO_ROOT), str(ISAAC_SIM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaac_sim.scenes._common import (                      # noqa: E402
    WAREHOUSE_2X3_ZONES, WAREHOUSE_2X3_NAV_X, WAREHOUSE_2X3_NAV_Y,
    warehouse_2x3_static_rects,
)
from isaac_sim.planning.a_star.offline_gate import path_points, rect_from_box  # noqa: E402
from isaac_sim.cam_to_map.camera_model import (                                # noqa: E402
    CameraModel, intrinsics_matrix, _USD_FROM_CV,
)
from isaac_sim.run_builders.build_2x3 import DataGenerator, _load_footprints   # noqa: E402

RENDERS = ISAAC_SIM_DIR / "outputs" / "env_panel"
SCENE_DIR = ISAAC_SIM_DIR / "configs" / "env" / "scene"
SWEEPS = REPO_ROOT / "runs" / "evaluation" / "sweeps"
DATAGEN_CFG = ISAAC_SIM_DIR / "configs" / "datagen_2x3.yaml"

CCTV_ORDER = ["cctv_NW", "cctv_NE", "cctv_SW", "cctv_SE"]

# ---- palette (colour-blind safe, readable on a projector) -------------------
C_RACK = "#c9ccd3"        # static racking
C_FREE = "#f7f8fa"        # navigable floor
C_SEAL = "#d62728"        # the forklift that seals an aisle
C_DISTR = "#8c8c8c"       # non-target props
C_TARGET = "#e6a700"      # the language target
C_PLAN = "#ff7f0e"        # A* reference plan
C_ROBOT = "#1f77b4"       # what the robot actually drove
C_START = "#2ca02c"
TWIN_COL = {"tA": "#1b9e77", "tB": "#d95f02", "tC": "#7570b3", "tD": "#e7298a"}

_TILE_H = 300     # px each camera frame is resized to before hstacking
_GAP = 14         # white px between tiles


# ---------------------------------------------------------------------------
# scene / rollout loading
# ---------------------------------------------------------------------------
def load_scene(name: str) -> dict:
    """Read a generated scene yaml ('# @package env' header + a JSON body)."""
    f = SCENE_DIR / f"{name}.yaml"
    body = "\n".join(l for l in f.read_text().splitlines() if not l.startswith("#"))
    return json.loads(body)


def load_trace(rollout_dir: pathlib.Path) -> np.ndarray:
    """The robot's driven path as (N, 2) world xy."""
    pts = []
    for line in (rollout_dir / "log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pts.append((d["x"], d["y"]))
    return np.asarray(pts)


_FOOTPRINTS = None


def footprints() -> dict:
    """Prop half-extents, keyed (kind, variant).

    The measured table has no entry for every kind the LANGUAGE scenes place
    (traffic_cone was never measured), so the same `footprint_overrides` block
    the language generator merges in is merged here — read from the yaml rather
    than copied, so the drawn rectangle can't drift from the placed prop.
    """
    global _FOOTPRINTS
    if _FOOTPRINTS is None:
        fps = _load_footprints()
        lang_cfg = OmegaConf.load(ISAAC_SIM_DIR / "configs" / "datagen_2x3_langv2test.yaml")
        for kind, variants in (lang_cfg.get("footprint_overrides") or {}).items():
            for variant, fp in variants.items():
                fps[(kind, int(variant))] = {"half_w": float(fp["half_w"]),
                                             "half_d": float(fp["half_d"]),
                                             "radius": float(fp["radius"])}
        _FOOTPRINTS = fps
    return _FOOTPRINTS


def obstacle_rects(scene: dict) -> list[tuple]:
    """World AABBs of the scene's placed props, in scene order."""
    fps = footprints()
    out = []
    for o in scene.get("obstacles", []):
        fp = fps[(o["kind"], o.get("variant", 0))]
        out.append(rect_from_box(o["x"], o["y"], fp["half_w"], fp["half_d"],
                                 o.get("yaw_deg", 0.0)))
    return out


_GEN = None      # DataGenerator is ~1 s to build (it rasterises a static grid
                 # map), so build it once on first use and reuse across panels.


def gen() -> DataGenerator:
    global _GEN
    if _GEN is None:
        _GEN = DataGenerator(OmegaConf.load(DATAGEN_CFG))
    return _GEN


def astar_plan(scene: dict) -> np.ndarray:
    """The A* reference route for this scene — the same planner the harness uses.

    Recomputed offline (no Isaac) from the static racks + the scene's own props,
    exactly as build_2x3_grid72_endseal did when it accepted the scene.
    """
    g = gen()
    pts = path_points(g.static_rects + obstacle_rects(scene),
                      tuple(scene["start_xy"]), tuple(scene["goal_xy"]),
                      g.gate_cfg, spacing_m=0.25)
    return np.asarray(pts) if pts else np.empty((0, 2))


# ---------------------------------------------------------------------------
# projecting world points into the CCTV frames
# ---------------------------------------------------------------------------
def cctv_camera_model(cam_name: str) -> CameraModel:
    """Build the pin-hole model of one bigcam CCTV from its camera yaml.

    The bigcam yamls carry a verbatim 4x4 USD transform (exported from the GUI)
    rather than an Euler triple, so `CameraModel.from_usd` — which takes Euler —
    does not apply. USD's matrix is ROW-major with the row-vector convention
    (world = local @ M), hence:
        R_world_from_usd = M[:3, :3].T      # transpose: row-vector -> column
        camera centre     = M[3, :3]
    then the same `diag(1, -1, -1)` flip to OpenCV axes the Euler path applies.
    """
    suffix = cam_name.split("_")[1].lower()             # cctv_NW -> "nw"
    cfg = OmegaConf.load(ISAAC_SIM_DIR / "configs" / "env" / "camera"
                         / f"cctv_2x3_{suffix}_bigcam.yaml")
    M = np.asarray([[float(v) for v in row]
                    for row in cfg.extrinsics.world_transform_matrix], dtype=np.float64)
    R_wc = (M[:3, :3].T) @ _USD_FROM_CV
    t = M[3, :3]
    w, h = int(cfg.resolution[0]), int(cfg.resolution[1])
    K = intrinsics_matrix(float(cfg.intrinsics.focal_length_mm),
                          float(cfg.intrinsics.horizontal_aperture_mm),
                          float(cfg.intrinsics.vertical_aperture_mm), w, h)
    return CameraModel(K=K, R_wc=R_wc, t=t)


def project_into_cctv(cam_name: str, xyz) -> tuple | None:
    """World point -> (u, v) in that CCTV's 720x720 frame, or None if unseen.

    None means either behind the camera (z_cam <= 0) or outside the image, which
    is exactly the "this camera cannot see it" case the panel wants to skip.
    """
    cam = cctv_camera_model(cam_name)
    uv, z = cam.project_points(np.asarray(xyz, dtype=np.float64)[None, :])
    if z[0] <= 0:
        return None
    u, v = float(uv[0, 0]), float(uv[0, 1])
    if not (0 <= u < 720 and 0 <= v < 720):
        return None
    return u, v


# ---------------------------------------------------------------------------
# the top-down map
# ---------------------------------------------------------------------------
def draw_map(ax, scene, traces, snap_poses, *, seal_names=(), object_labels=None,
             goal=True, plan=True):
    """Top-down warehouse map.

    traces:      list of (label, colour, (N,2) xy) — one per rollout to overlay
    snap_poses:  list of dicts with x/y/heading_rad — numbered camera snapshots
    seal_names:  aisle letters sealed in this scene (drawn as a red overlay)
    object_labels: optional {index_into_scene_obstacles: (text, colour)}
    """
    x0, x1 = WAREHOUSE_2X3_NAV_X
    y0, y1 = WAREHOUSE_2X3_NAV_Y

    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=C_FREE,
                           edgecolor="#5a5f6a", linewidth=1.4, zorder=0))

    # Sealed aisles: tint the whole corridor so the "this route is closed" reads
    # from the back of the room, not just the 1.2 m forklift footprint.
    for z in seal_names:
        zn = WAREHOUSE_2X3_ZONES[z]
        ax.add_patch(Rectangle((zn.x_range[0], zn.y_range[0]),
                               zn.x_range[1] - zn.x_range[0],
                               zn.y_range[1] - zn.y_range[0],
                               facecolor=C_SEAL, alpha=0.14, zorder=1))

    for (rx0, ry0, rx1, ry1) in warehouse_2x3_static_rects():
        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                               facecolor=C_RACK, edgecolor="none", zorder=2))

    # Zone letters — the A..G "digital-8" topology the tasks are defined over.
    # A white halo keeps them readable where a trace runs straight over them
    # (the robot drives down the middle of an aisle, i.e. over its label).
    halo = [path_effects.withStroke(linewidth=3.5, foreground="white")]
    for name, zn in WAREHOUSE_2X3_ZONES.items():
        cx, cy = zn.center
        sealed = name in seal_names
        ax.text(cx, cy, name, ha="center", va="center", zorder=8,
                fontsize=13, fontweight="bold", path_effects=halo,
                color=C_SEAL if sealed else "#9aa0aa",
                alpha=0.95 if sealed else 0.85)

    for i, (o, rect) in enumerate(zip(scene.get("obstacles", []),
                                      obstacle_rects(scene))):
        rx0, ry0, rx1, ry1 = rect
        role = o.get("role", "")
        col = C_SEAL if role == "blocker" else (C_TARGET if role == "target" else C_DISTR)
        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, facecolor=col,
                               edgecolor="black", linewidth=0.8, zorder=5))
        if object_labels and i in object_labels:
            txt, tcol = object_labels[i]
            # Label ABOVE the prop, not beside it: the aisles are ~1.1 m of clear
            # width on this map, so a side label lands on the neighbouring rack.
            ax.text(0.5 * (rx0 + rx1), ry1 + 0.6, txt, fontsize=9.5, zorder=9,
                    color=tcol, fontweight="bold", va="bottom", ha="center",
                    path_effects=halo)

    # The language panel overlays four twin traces to four different targets, so
    # a single scene's A* route would be one arbitrary line among them -> off.
    if plan:
        pts = astar_plan(scene)
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], "--", color=C_PLAN, linewidth=2.0,
                    zorder=6, label="A* reference route")

    for label, col, tr in traces:
        ax.plot(tr[:, 0], tr[:, 1], "-", color=col, linewidth=2.4, zorder=7,
                label=label)

    sx, sy = scene["start_xy"]
    # Snapshot (1) is always taken AT the spawn, so its white numbered disc would
    # sit exactly on top of a same-size start marker. Drawing the start larger and
    # underneath leaves a visible green ring around the "1".
    ax.plot([sx], [sy], "o", color=C_START, markersize=19, markeredgecolor="k",
            markeredgewidth=0.8, zorder=8, label="start")
    # Only the spatial task has ONE goal point worth marking; in the language
    # task each twin has its own goal (drawn as the labelled target props), so
    # the caller passes goal=False there.
    if goal:
        gx, gy = scene["goal_xy"]
        ax.plot([gx], [gy], "*", color="#d62728", markersize=19, markeredgecolor="k",
                markeredgewidth=0.8, zorder=9, label="goal")

    # Numbered snapshot markers + a heading arrow: ties each camera tile to a
    # place on the map, which is what makes the filmstrip readable on a slide.
    for i, p in enumerate(snap_poses, start=1):
        hx, hy = np.cos(p["heading_rad"]), np.sin(p["heading_rad"])
        ax.add_patch(FancyArrow(p["x"], p["y"], 1.9 * hx, 1.9 * hy, width=0.18,
                                head_width=0.9, head_length=0.9, length_includes_head=True,
                                color="black", zorder=10))
        ax.plot([p["x"]], [p["y"]], "o", color="white", markersize=15,
                markeredgecolor="black", markeredgewidth=1.4, zorder=11)
        ax.text(p["x"], p["y"], str(i), ha="center", va="center", fontsize=9.5,
                fontweight="bold", zorder=12)

    pad = 1.5
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("world x [m]", fontsize=9)
    ax.set_ylabel("world y [m]", fontsize=9)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# camera filmstrips
# ---------------------------------------------------------------------------
def strip(paths, labels, ax, border_cols=None, title=None, marks=None):
    """hstack images into ONE composite and imshow it.

    Building the strip in PIL rather than as N matplotlib subplots keeps the
    inter-tile gap exactly `_GAP` px regardless of the axes aspect — subplots
    with `aspect='equal'` drift apart unpredictably when the figure is resized.

    marks: optional {tile_index: [(u_src, v_src, text), ...]} in SOURCE-image
           pixels; rescaled to strip coords here so callers can hand over raw
           projection output without knowing the tile layout.
    """
    tiles, scales, offsets = [], [], []
    x = 0
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = _TILE_H / h
        im = im.resize((int(round(w * s)), _TILE_H))
        col = (border_cols or {}).get(i, (60, 60, 60))
        tiles.append(ImageOps.expand(im, border=4, fill=col))
        scales.append(s)
        offsets.append(x)
        x += tiles[-1].width + _GAP
    H = max(t.height for t in tiles)
    W = sum(t.width for t in tiles) + _GAP * (len(tiles) - 1)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    x, centers = 0, []
    for t in tiles:
        canvas.paste(t, (x, 0))
        centers.append(x + t.width / 2)
        x += t.width + _GAP

    ax.imshow(np.asarray(canvas), aspect="equal")
    ax.set_xlim(0, W)
    ax.set_ylim(H, -46)                     # headroom for the per-tile labels
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for cx, lab in zip(centers, labels):
        ax.text(cx, -8, lab, ha="center", va="bottom", fontsize=11)
    if title:
        ax.text(0, -30, title, ha="left", va="bottom", fontsize=12.5,
                fontweight="bold")

    for i, pts in (marks or {}).items():
        for (u_src, v_src, text) in pts:
            # +4 for the border expand() added around each tile
            u = offsets[i] + 4 + u_src * scales[i]
            v = 4 + v_src * scales[i]
            ax.plot([u], [v], "o", markersize=26, markerfacecolor="none",
                    markeredgecolor=C_SEAL, markeredgewidth=3.0, zorder=5)
            ax.annotate(text, xy=(u, v), xytext=(u, v + 78), ha="center",
                        fontsize=11, fontweight="bold", color=C_SEAL, zorder=6,
                        arrowprops=dict(arrowstyle="-", color=C_SEAL, lw=2.0),
                        path_effects=[path_effects.withStroke(linewidth=3.5,
                                                              foreground="white")])


# ---------------------------------------------------------------------------
# the two panels
# ---------------------------------------------------------------------------
def build_panel(key, *, title, subtitle, scene_name, traces, seal_names=(),
                object_labels=None, cctv_step=None, ego_captions=None, goal=True,
                plan=True, show_steps=None, highlight_world=None):
    meta = json.loads((RENDERS / key / "poses.json").read_text())
    # The renderer may capture more steps than the figure shows (rendering a
    # superset is cheap once Kit is booted; picking the legible frames is not).
    poses = meta["poses"]
    if show_steps is not None:
        by_step = {p["step"]: p for p in poses}
        poses = [by_step[s] for s in show_steps]
    steps = [p["step"] for p in poses]
    cctv_step = steps[0] if cctv_step is None else cctv_step
    scene = load_scene(scene_name)

    # Height ratios follow the two strips' true aspect (4 ego tiles are 4:3,
    # 4 CCTV tiles are 1:1) so neither axis letterboxes a band of white.
    fig = plt.figure(figsize=(18.0, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 6.3], height_ratios=[1.0, 1.34],
                          left=0.030, right=0.988, top=0.870, bottom=0.090,
                          wspace=0.06, hspace=0.20)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_ego = fig.add_subplot(gs[0, 1])
    ax_cctv = fig.add_subplot(gs[1, 1])

    draw_map(ax_map, scene, traces, poses, seal_names=seal_names,
             object_labels=object_labels, goal=goal, plan=plan)
    # Legend as a figure-level strip along the bottom: inside the map it either
    # covers the goal marker or the south aisles — the map is only ~1.5 in wide.
    handles, labels = ax_map.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=len(handles), frameon=False, fontsize=11.5)

    ego_paths = [RENDERS / key / f"step_{s:03d}_front_cam.png" for s in steps]
    caps = ego_captions or [""] * len(steps)
    ego_labels = [f"({i})  {c}" if c else f"({i})"
                  for i, c in enumerate(caps, start=1)]
    strip(ego_paths, ego_labels, ax_ego,
          title="Onboard camera  —  the ego view the policy is conditioned on")

    cctv_paths = [RENDERS / key / f"step_{cctv_step:03d}_{c}.png" for c in CCTV_ORDER]
    cctv_labels = [c.replace("cctv_", "CCTV ") for c in CCTV_ORDER]
    # Circle any called-out world point in whichever CCTVs actually see it —
    # projected through each camera's own intrinsics/extrinsics, not eyeballed.
    marks = {}
    for (wx, wy, wz, text) in (highlight_world or []):
        for i, cam in enumerate(CCTV_ORDER):
            uv = project_into_cctv(cam, (wx, wy, wz))
            if uv is not None:
                marks.setdefault(i, []).append((uv[0], uv[1], text))
    strip(cctv_paths, cctv_labels, ax_cctv, marks=marks,
          title=f"Fixed infrastructure cameras  —  all four, at snapshot "
                f"({steps.index(cctv_step) + 1})")

    fig.suptitle(title, fontsize=20, fontweight="bold", x=0.035, ha="left", y=0.975)
    fig.text(0.035, 0.923, subtitle, fontsize=11.5, ha="left", color="#444")

    out = RENDERS / f"panel_{key}.png"
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"[panel] wrote {out}")
    return out


def panel_spatial():
    meta = json.loads((RENDERS / "spatial" / "poses.json").read_text())
    rollout = SWEEPS / meta["rollout"]
    res = json.loads((rollout / "result.json").read_text())
    trace = load_trace(rollout)
    scene = load_scene(meta["scene"])
    blocker = next(o for o in scene["obstacles"] if o.get("role") == "blocker")
    return build_panel(
        "spatial",
        title="Spatial task — reach a goal pose when aisles are blocked",
        subtitle=(f"{meta['scene']}   ·   aisle B sealed by a forklift   ·   "
                  f"start in A (north strip) → goal in G (south strip)   ·   "
                  f"outcome: {res['outcome']} at {res['final_dist']:.2f} m "
                  f"after {res['steps']} steps"),
        scene_name=meta["scene"],
        traces=[("robot rollout", C_ROBOT, trace)],
        seal_names=("B",),
        ego_captions=["spawn, facing sealed aisle B", "detouring east into C",
                      "crossing back west via D", "descending the open aisle E"],
        # 150/170 were also rendered but sit close enough to the south wall that
        # the frame is mostly wall; 130 shows the corridor the robot is driving.
        show_steps=[0, 30, 100, 130],
        cctv_step=0,
        # Call out the blocker itself. Its world xy comes from the scene config;
        # z=1.0 aims at the forklift's mast rather than its contact patch, so the
        # circle lands on the body. Whichever CCTVs see it get the annotation.
        highlight_world=[(blocker["x"], blocker["y"], 1.0, "blocked aisle")],
    )


def panel_language():
    meta = json.loads((RENDERS / "language" / "poses.json").read_text())
    # All four twins share the scene; overlay every instruction's trace so the
    # panel shows the ONE thing that differs between them.
    base = SWEEPS / meta["rollout"]
    family_dir = base.parent
    stem = base.name.split("__lang__")[0][:-3]      # strip the "_tX" suffix
    traces, insts = [], {}
    for twin in ("tA", "tB", "tC", "tD"):
        hits = sorted(family_dir.glob(f"{stem}_{twin}__lang__*"))
        if not hits:
            continue
        d = hits[0]
        cfg = OmegaConf.load(d / "config.yaml")
        insts[twin] = str(cfg.task.lang_instruction)
        traces.append((f'{twin}:  "{insts[twin]}"', TWIN_COL[twin], load_trace(d)))

    scene = load_scene(meta["scene"])
    labels = {i: (o["kind"].replace("_", " "),
                  C_TARGET if o.get("role") == "target" else "#555")
              for i, o in enumerate(scene.get("obstacles", []))}
    return build_panel(
        "language",
        title="Language task — reach the named object",
        subtitle=("warehouse_2x3_langv2_0500   ·   four twin scenes, identical "
                  "spawn and identical object placement — only the instruction "
                  "differs   ·   all four reached their target"),
        scene_name=meta["scene"],
        traces=traces,
        object_labels=labels,
        goal=False,
        plan=False,
        ego_captions=['spawn (shared by all four)', 'committed toward the forklift',
                      "turning north into aisle C", "target in view"],
        cctv_step=meta["poses"][0]["step"],
    )


if __name__ == "__main__":
    panel_spatial()
    panel_language()
