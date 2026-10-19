from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from isaac_sim.planning.a_star.planner import A_star, PlanningError


# A world-space axis-aligned rectangle in metres: (xmin, ymin, xmax, ymax).
Rect = tuple[float, float, float, float]


def rect_from_box(cx, cy, half_w, half_d, yaw_deg=0.0) -> Rect:
    """World axis-aligned bounding rectangle of an oriented box.

    The box is centred at (cx, cy) with local half-extents `half_w` (along local
    x) and `half_d` (along local y), rotated `yaw_deg` about +Z. The returned AABB
    is what `UsdGeom.BBoxCache.ComputeAlignedRange()` yields for the real prim, so
    the rasterised footprint matches what the live `MapBuilder` would mark.

    Orientation is load-bearing for aisle blockers: a forklift
    (half_w=0.607, half_d=1.747, from measured_footprints.json) at yaw=0 is
    1.21 m wide x 3.49 m long; at yaw=90 its 3.49 m long axis swings into x and
    spans the aisle. e.g. rect_from_box(0.73, 16, 0.607, 1.747, 90)
    -> (-1.017, 15.393, 2.477, 16.607)  # 3.49 m across, 1.21 m deep
    """
    a = math.radians(yaw_deg)
    ca, sa = abs(math.cos(a)), abs(math.sin(a))
    hx = half_w * ca + half_d * sa   # AABB half-width  after rotation
    hy = half_w * sa + half_d * ca   # AABB half-depth  after rotation
    return (cx - hx, cy - hy, cx + hx, cy + hy)


@dataclass
class GridMap:
    """USD-free mirror of `a_star.map.Map`: the grid contract `A_star` reads.

    Fields and the two coordinate transforms are copied verbatim from
    `a_star.map.Map` so the planner behaves identically on either.
    """
    occupancy: np.ndarray         # (H, W) bool, True = occupied
    sdf: np.ndarray | None        # (H, W) float, metres to nearest obstacle
    origin: tuple[float, float]   # world (x, y) of cell (row=0, col=0)
    resolution: float             # metres per cell

    def get_map_coords(self, xy):
        """World (x, y) -> (row, col). NOTE row indexes y, col indexes x."""
        x, y = xy
        c = int(round((x - self.origin[0]) / self.resolution))
        r = int(round((y - self.origin[1]) / self.resolution))
        return r, c

    def get_world_coords(self, rc):
        """(row, col) -> world (x, y)."""
        r, c = rc
        return (self.origin[0] + c * self.resolution,
                self.origin[1] + r * self.resolution)

    def _rasterize(self, rect: Rect, max_half_extent_m: float) -> None:
        """Mark a world rectangle occupied.

        Mirrors `Map.add_prim`'s rasterisation AND the `_world_aabb` size cap: a
        rect whose half-extent exceeds `max_half_extent_m` is DROPPED — the same
        silent skip the live builder applies to building-scale geometry (so a
        long shelf row must be fed as its per-bay rects, not one 16 m rectangle).
        """
        xmin, ymin, xmax, ymax = rect
        hx, hy = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)
        if max(hx, hy) > max_half_extent_m:
            return  # too big -> excluded, same as the live MapBuilder
        res = self.resolution
        H, W = self.occupancy.shape
        c0 = max(0, int(math.floor((xmin - self.origin[0]) / res)))
        c1 = min(W, int(math.ceil((xmax - self.origin[0]) / res)))
        r0 = max(0, int(math.floor((ymin - self.origin[1]) / res)))
        r1 = min(H, int(math.ceil((ymax - self.origin[1]) / res)))
        self.occupancy[r0:r1, c0:c1] = True


def build_grid_map(rects, points, cfg) -> GridMap:
    """Rasterise `rects` into a GridMap sized to contain `rects` + `points`.

    Mirrors `MapBuilder._world_bounds` + `construct_map` (USD-free): bounds = union
    of every rect corner and every point (the start/goal, so the planner's
    endpoints are always inside), padded by `cfg.world_buffer_m`; then fill the
    ESDF.

    Args:
        rects: iterable of world AABB Rects (static fixtures + obstacles).
        points: iterable of world (x, y) that must lie inside the grid (start/goal).
        cfg: A* config — any object exposing the fields of configs/env/astar/*.yaml
             (grid_resolution_m, robot_inflation_m, world_buffer_m, ...).
    """
    res = cfg.grid_resolution_m
    buf = cfg.world_buffer_m
    max_half = cfg.max_obstacle_half_extent_m

    xs, ys = [], []
    for (xmin, ymin, xmax, ymax) in rects:
        xs += [xmin, xmax]
        ys += [ymin, ymax]
    for (x, y) in points:
        xs.append(x)
        ys.append(y)

    xmin, xmax = min(xs) - buf, max(xs) + buf
    ymin, ymax = min(ys) - buf, max(ys) + buf
    W = int(math.ceil((xmax - xmin) / res))
    H = int(math.ceil((ymax - ymin) / res))

    m = GridMap(np.zeros((H, W), dtype=bool), None, (xmin, ymin), res)
    for rect in rects:
        m._rasterize(rect, max_half)
    # Optional play-area restriction (mirrors a_star.map._mask_outside): mark cells
    # OUTSIDE navigable_bounds occupied, so the gate matches a live map built with
    # the same bounds. (Can't import map.py here — it pulls in omni.usd.)
    nb = getattr(cfg, "navigable_bounds", None)
    if nb is not None:
        xmn, ymn, xmx, ymx = (float(b) for b in nb)
        cols = xmin + np.arange(W) * res
        rows = ymin + np.arange(H) * res
        inside = np.outer((rows >= ymn) & (rows <= ymx), (cols >= xmn) & (cols <= xmx))
        m.occupancy |= ~inside
    # ESDF in metres: distance from each FREE cell to the nearest occupied cell.
    # Same formula as a_star.map.euclidean_sdf (invert so free cells are the
    # "measured" region).
    m.sdf = distance_transform_edt(~m.occupancy) * res
    return m


def has_path(rects, p_from, p_to, cfg) -> bool:
    """True iff A* finds a collision-free route from `p_from` to `p_to` over
    `rects`. A raised `PlanningError` (endpoint inside an inflated obstacle, or no
    route at all) -> False.

    The gate cares only about path EXISTENCE, so run the cfg with `pipeline`'s
    hill-climbing / cubic-spline OFF (they only smooth an already-found path and
    can't change whether one exists).
    """
    m = build_grid_map(rects, [p_from, p_to], cfg)
    try:
        A_star(cfg, m).get_a_star_path(p_from, p_to)
        return True
    except PlanningError:
        return False


def aisle_sealed(aisle_rects, blocker_rect, p_below, p_above, cfg) -> bool:
    """True iff `blocker_rect` seals the corridor: no path from `p_below` to
    `p_above` over (aisle walls + blocker). The inverse of `has_path` restricted
    to the local corridor — so an aisle counts as sealed even if a global detour
    via another aisle still exists.
    """
    return not has_path(list(aisle_rects) + [blocker_rect], p_below, p_above, cfg)


def path_length(rects, p_from, p_to, cfg) -> float | None:
    """Arc length (m) of the planned path, or None if unsolvable.

    For the path-elongation statistic
    (`path_length(obstacles) / path_length(empty)`), pass the SAME cfg for both
    calls so the ratio is consistent. The 2x3 generator uses its gate cfg
    (search-only) — robust and a fine relative measure; pass a full-pipeline cfg
    instead only if you need the absolute length to match the oracle's smoothed
    trajectory.
    """
    m = build_grid_map(rects, [p_from, p_to], cfg)
    try:
        return float(A_star(cfg, m).get_a_star_path(p_from, p_to).total_length)
    except PlanningError:
        return None


def path_points(rects, p_from, p_to, cfg, spacing_m=0.25):
    """Sampled (x, y) waypoints of the planned path over `rects`, or None if
    unsolvable. For route-aware blocking: identify WHICH aisles the oracle's route
    traverses, so a blocker can be placed where it actually forces a detour."""
    m = build_grid_map(rects, [p_from, p_to], cfg)
    try:
        return A_star(cfg, m).get_a_star_path(p_from, p_to).sample(spacing_m)
    except PlanningError:
        return None
