"""Template: warehouse_2x3 — the reshaped 2x3 warehouse with 4 fixed CCTVs.

Unlike the other templates, this one does NOT build the warehouse procedurally.
The scene config sets `base_usda` (the hand-fixed scenes/warehouse_2x3_base.usda),
which IsaacSimEnv opens as the stage BEFORE this build() runs — so the 2x3
reshape, manual tile fixes, and the 4 symmetric CCTV cameras are loaded verbatim
from the file. This build() only layers the task on top: lighting + goal +
optional obstacles, exactly like the other templates.

Everything else (other templates, wall_course, all existing camera configs) is
untouched — this is additive.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

from .._common import (
    BaseTemplateConfig,
    LIGHTING_SCALE,
    PROP_CATALOG,
    place_goal_or_designate,
    place_prop,
    scale_baked_lights,
    validate_kinds,
    warehouse_2x3_zone_at,
)

# Things we KEEP even when they land in a navigable zone: the intentional aisle
# number signs, and structural pillars/beams (load-bearing, sit at the far edge).
_CLUTTER_KEEP = ("s_aislesign", "sm_pillar")
_CLUTTER_MAX_HALF_M = 1.0   # bigger than this = structure (racks/walls/pillars), not clutter
_CLUTTER_MAX_FLOOR_Z = 0.5  # base higher than this = wall/ceiling-mounted, not on the floor


def _strip_navigable_clutter(stage) -> int:
    """function to strip navigable clutter"""
    from pxr import Usd, UsdGeom

    wh = stage.GetPrimAtPath("/World/Warehouse")
    if not wh or not wh.IsValid():
        return 0
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                           includedPurposes=[UsdGeom.Tokens.default_])
    n = 0
    for child in wh.GetChildren():
        if not child.IsActive() or not child.IsA(UsdGeom.Imageable):
            continue
        if any(k in child.GetName().lower() for k in _CLUTTER_KEEP):
            continue
        rng = bc.ComputeWorldBound(child).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        hx, hy = 0.5 * (mx[0] - mn[0]), 0.5 * (mx[1] - mn[1])
        if max(hx, hy) > _CLUTTER_MAX_HALF_M:   # structure, not a prop
            continue
        if float(mn[2]) > _CLUTTER_MAX_FLOOR_Z:  # wall/ceiling-mounted, off the floor
            continue
        cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
        if warehouse_2x3_zone_at(cx, cy) is not None:
            child.SetActive(False)
            n += 1
    return n


@dataclass
class Config(BaseTemplateConfig):
    """Config class for a 2x3 warehouse inherits form the base config """

    template: str = "warehouse_2x3"
    base_usda: str | None = None

    # goal_clear_radius is emitted into every scene yaml (so from_dict must accept
    # it) but build() never reads it here — place_goal_or_designate ignores it.
    # (It IS a live knob in template1-5 + the goal success-check.)
    goal_xy: tuple[float, float] = (0.0, 0.0)
    goal_kinds: list[str] = field(default_factory=list)   # [] = designate point only
    goal_yaw_deg: float = 0.0
    goal_clear_radius: float = 1.0
    obstacles: list[dict] = field(default_factory=list)

    # Unused by 2x3 (placement is generator-decided, replayed from `obstacles`)
    # but KEPT: the shared validate_kinds() asserts on it for EVERY template
    # config, so removing it would raise AttributeError. See end-of-file notes.
    obstacle_kinds: list[str] = field(default_factory=list)

    lighting: str = "normal"

    def __post_init__(self) -> None:
        """validates the objects and initializes the config"""
        validate_kinds(self)
        super().__post_init__()   # asserts start_xy / start_heading_deg set

    def to_dict(self) -> dict:
        """returns a dictioanry version of the config"""
        d = asdict(self)
        d["lighting_factor"] = LIGHTING_SCALE[self.lighting]
        return d

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        """turns a dictionary into a config object"""
        raw = dict(raw)
        for k in ("start_xy", "goal_xy"):
            if raw.get(k) is not None:
                raw[k] = tuple(raw[k])
        raw.pop("lighting_factor", None)   # to_dict-only derived field
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"Config: unknown keys {sorted(unknown)}. "
                f"Valid: {sorted(cls.__dataclass_fields__)}."
            )
        return cls(**raw)


def build(world, assets_root: str, config: Config) -> None:
    """takes in world, assets and the config, places the config objects according to specification"""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    rng = random.Random(config.seed)
    #adjust lifght
    factor = LIGHTING_SCALE[config.lighting]
    #changes the baked in lights of the scene
    changes = scale_baked_lights(stage, factor)
    print(f"[warehouse_2x3] lighting={config.lighting} factor={factor:.2f} "
          f"(scaled {len(changes)} UsdLux prims)")
    #TODO investigate what is hapenning here
    n_clutter = _strip_navigable_clutter(stage)
    print(f"[warehouse_2x3] stripped {n_clutter} stray floor props "
          f"(margins sealed by the A* navigable_bounds mask)")

    #places the goal object if one exissts
    gx, gy = config.goal_xy
    place_goal_or_designate(stage, assets_root, config, rng, (gx, gy),
                            log_prefix="[warehouse_2x3]")
    #places each object and get them first from the catalogue then places prop on the stage
    for i, o in enumerate(config.obstacles):
        spec = PROP_CATALOG[o["kind"]][int(o.get("variant", 0))]
        # role=="target" (language task) -> /World/Goal: invisible to the A* map
        # (MapBuilder only scans /World/Obstacles/) and triggers the env's
        # measured-AABB standoff, so the oracle stops at the shell, never inside.
        prim_path = ("/World/Goal" if o.get("role") == "target"
                     else f"/World/Obstacles/Obs_{i:03d}")
        place_prop(stage, assets_root, prim_path,
                   spec, float(o["x"]), float(o["y"]), float(o.get("yaw_deg", 0.0)))
    #blockers get extra print to separate how many blockers and how many distractors have been placed
    if config.obstacles:
        n_block = sum(1 for o in config.obstacles if o.get("role") == "blocker")
        print(f"[warehouse_2x3] placed {len(config.obstacles)} obstacles "
              f"({n_block} blockers, {len(config.obstacles) - n_block} distractors)")
