import math
from dataclasses import dataclass

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom
from scipy.ndimage import distance_transform_edt


_DYNAMIC_PRIM_ROOT = "/World/Obstacles/"

_STATIC_NAME_TOKENS = (
    "rackshelf", "rackframe",
    "pallet", "palette",
    "cardbox", "crate", "barel", "barrel",
    "forklift", "pushcart", "cart",
    "floordecal_recred",
)

_EXCLUDE_PREFIXES = (
    "/World/Goal",
    "/World/Obstacles",
    "/World/Carter", "/World/NovaCarter", "/World/Spot",
)

_SANE_CENTER_HALF_EXTENT_M = 100.0
_MAX_OBSTACLE_HALF_EXTENT_M = 3.0
_MAX_BOUNDS_HALF_EXTENT_M = 100.0

def euclidean_sdf(occupancy, resolution):
    """Returns the map with ESDF distances"""
    return distance_transform_edt(~occupancy) * resolution


def _navigable_bounds(cfg):
    """Optional navigable_bounds (xmin, ymin, xmax, ymax) from cfg (DictConfig or
    namespace), or None when the key is absent (-> no play-area restriction)."""
    try:
        nb = cfg.navigable_bounds
    except Exception:
        nb = getattr(cfg, "navigable_bounds", None)
    return list(nb) if nb is not None else None


def _mask_outside(m, bounds):
    """Mark every grid cell whose centre is OUTSIDE the world rectangle
    bounds=(xmin, ymin, xmax, ymax) as occupied — a planner play-area restriction
    that needs no physical wall prims (row indexes y, col indexes x)."""
    xmin, ymin, xmax, ymax = (float(b) for b in bounds)
    H, W = m.occupancy.shape
    cols = m.origin[0] + np.arange(W) * m.resolution
    rows = m.origin[1] + np.arange(H) * m.resolution
    inside = np.outer((rows >= ymin) & (rows <= ymax), (cols >= xmin) & (cols <= xmax))
    m.occupancy |= ~inside

@dataclass
class Map:
    """Map Class to create a map object from the environment that can be used for A* path generation."""
    occupancy: np.ndarray
    sdf: np.ndarray | None
    origin: tuple[float, float]
    resolution: float

    @classmethod
    def empty(cls, origin, resolution, shape):
        """Creates a new empty map"""
        return cls(np.zeros(shape, dtype=bool), None, origin, resolution)

    def add_prim(self, prim, bbox_cache, max_half_extent_m=_MAX_OBSTACLE_HALF_EXTENT_M):
        """
        Adds a prim to the map.

        This function gets the world bounds and adds the obejcts to the map by setting the respective fields to true.

        Arsg:
            prim: the specifc object obtained form stage
            bbox_cache: cache where the boundig box is cached
            max_half_extent_m: size cap forwarded to _world_aabb (yaml knob
                cfg.env.astar.max_obstacle_half_extent_m)
        """
        aabb = _world_aabb(prim, bbox_cache, max_half_extent_m)
        if aabb is None:
            return
        xmin, ymin, xmax, ymax = aabb
        res = self.resolution
        H, W = self.occupancy.shape
        c0 = max(0, int(math.floor((xmin - self.origin[0]) / res)))
        c1 = min(W, int(math.ceil((xmax - self.origin[0]) / res)))
        r0 = max(0, int(math.floor((ymin - self.origin[1]) / res)))
        r1 = min(H, int(math.ceil((ymax - self.origin[1]) / res)))
        self.occupancy[r0:r1, c0:c1] = True

    def get_map_coords(self, xy):
        """Returns xy coordinates expressed in rows and columns."""
        x, y = xy
        c = int(round((x - self.origin[0]) / self.resolution))
        r = int(round((y - self.origin[1]) / self.resolution))
        return r, c

    def get_world_coords(self, rc):
        """Returns world coordinates from rows and columns."""
        r, c = rc
        return (self.origin[0] + c * self.resolution,
                self.origin[1] + r * self.resolution)

class MapBuilder:
    """Map builder class to build a map from a given environment."""

    def __init__(self, a_star_cfg, env, sdf_fn=euclidean_sdf):
        """Initializes a map bilder."""
        self.env = env
        self.cfg = a_star_cfg
        self.sdf_fn = sdf_fn

    def construct_map(self):
        """
        Costructs a map.

        This function iterates over the objects in the stage, ones added by us but also fixed parts of the warehosue and adds each of those to the map.

        Return
            m: map that was build from the stage
        """
        stage = omni.usd.get_context().get_stage()
        bc = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_],
        )
        added_prims = _get_world_added_prims(stage)
        fix_prims = _get_world_fix_prims(stage)
        prims = added_prims + fix_prims

        # Yaml-sourced size cap (env/astar/*.yaml): prims with a larger
        # half-extent are treated as building structure and skipped.
        max_half = float(self.cfg.max_obstacle_half_extent_m)

        origin, shape = self._world_bounds(prims, bc, max_half)
        m = Map.empty(origin, self.cfg.grid_resolution_m, shape)
        for prim in prims:
            m.add_prim(prim, bc, max_half)
        # Optional play-area restriction: mark every cell OUTSIDE navigable_bounds
        # occupied, so A* won't route into non-operational space (e.g. the gap
        # behind the 2x3 shelves) — no fabricated wall prims needed. Off for any
        # scene whose astar cfg omits navigable_bounds.
        nb = _navigable_bounds(self.cfg)
        if nb is not None:
            _mask_outside(m, nb)
        m.sdf = self.sdf_fn(m.occupancy, m.resolution)
        return m

    def _world_bounds(self, prims, bc, max_half_extent_m=_MAX_OBSTACLE_HALF_EXTENT_M):
        """Returns the bounds of the world, caps if too big"""
        aabbs = [a for a in (_world_aabb(p, bc, max_half_extent_m) for p in prims)
                 if a is not None]
        if not aabbs:
            raise RuntimeError("MapBuilder: no obstacles found")
        xs = [v for ab in aabbs for v in (ab[0], ab[2])]
        ys = [v for ab in aabbs for v in (ab[1], ab[3])]
        sx, sy = self.env.template_cfg.start_xy
        gx, gy = self.env.template_cfg.goal_xy
        xs += [sx, gx]
        ys += [sy, gy]
        buf = self.cfg.world_buffer_m
        xmin, ymin = min(xs) - buf, min(ys) - buf
        xmax, ymax = max(xs) + buf, max(ys) + buf

        span = max(xmax - xmin, ymax - ymin)
        if span > 2 * _MAX_BOUNDS_HALF_EXTENT_M:
            print(f"[MapBuilder] WARN: world span {span:.1f} m exceeds cap, "
                  f"clamping to {2 * _MAX_BOUNDS_HALF_EXTENT_M} m")
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            xmin, xmax = cx - _MAX_BOUNDS_HALF_EXTENT_M, cx + _MAX_BOUNDS_HALF_EXTENT_M
            ymin, ymax = cy - _MAX_BOUNDS_HALF_EXTENT_M, cy + _MAX_BOUNDS_HALF_EXTENT_M

        res = self.cfg.grid_resolution_m
        W = int(math.ceil((xmax - xmin) / res))
        H = int(math.ceil((ymax - ymin) / res))
        return (xmin, ymin), (H, W)


def _get_world_added_prims(stage):
    """Returns the prims that have been added by us."""
    return [p for p in stage.Traverse()
            if str(p.GetPath()).startswith(_DYNAMIC_PRIM_ROOT)
            and p.IsA(UsdGeom.Imageable)]


def _get_world_fix_prims(stage):
    """Returns a list of prims that are fixed part of the map."""
    out = []
    for p in stage.Traverse():
        path = str(p.GetPath())
        if any(path.startswith(ex) for ex in _EXCLUDE_PREFIXES):
            continue
        if not any(tok in path.lower() for tok in _STATIC_NAME_TOKENS):
            continue
        if not p.IsA(UsdGeom.Imageable):
            continue
        out.append(p)
    return out


def _world_aabb(prim, bc, max_half_extent_m=_MAX_OBSTACLE_HALF_EXTENT_M):
    """returns the min and max for x and y axis of one object given the bounding box cache."""
    try:
        r = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = r.GetMin(), r.GetMax()
        xmin, ymin = float(mn[0]), float(mn[1])
        xmax, ymax = float(mx[0]), float(mx[1])
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
        return None
    if xmax < xmin or ymax < ymin:
        # USD's EMPTY-range sentinel is (+FLT_MAX, -FLT_MAX): geometry-less
        # Imageables (e.g. a "Looks" Scope under a referenced prop). Its
        # negative half-extent would slip past the size cap below and blow the
        # world bounds up to ±3.4e38.
        return None
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    hx, hy = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)
    if abs(cx) > _SANE_CENTER_HALF_EXTENT_M or abs(cy) > _SANE_CENTER_HALF_EXTENT_M:
        return None
    if max(hx, hy) > max_half_extent_m:
        return None
    return (xmin, ymin, xmax, ymax)
