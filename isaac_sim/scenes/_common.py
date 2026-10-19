"""
Shared infrastructure for warehouse scene templates.

Templates (templates/warehouse_2x3.py) import from here:
  - PROP_CATALOG: spawnable Isaac assets, grouped by category (single catalog
                  for obstacles and goal props)
  - BaseTemplateConfig: dataclass each template extends with its own knobs
  - load_base_warehouse(): adds NVIDIA's full_warehouse.usd as a stage reference
  - scale_baked_lights(): walks UsdLux prims and multiplies their intensity
  - sample_obstacles_along_line(): rejection sampler for "near-line + off-line"
                                    obstacle placement between two points
  - place_prop(): adds a prop USD reference at a given world pose

The build() and load_config() at the bottom dispatch a config to the right
template module based on its `template` field.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# Stock NVIDIA full warehouse — has shelves, walls, lighting, the works.
BASE_WAREHOUSE_USD_REL = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"


# ---------------------------------------------------------------------------
# Asset catalogs
# ---------------------------------------------------------------------------

@dataclass
class PropSpec:
    """One spawnable Isaac asset + footprint metadata for collision rejection."""
    usd_path: str            # Nucleus path; assets_root prepended at spawn time
    footprint_radius: float  # rough horizontal radius in m
    z_height: float = 0.0    # vertical offset where the prop's origin sits


# Obstacles / distractors — the things that go *between* robot and goal.
# footprint_radius values are measured (see utils/scene_prep/measured_footprints.json):
# the rotation-invariant disc radius from each USD's world-space AABB at yaw=0,
# r = sqrt(half_w**2 + half_d**2) — the smallest disc containing the prop at
# any yaw, so a sampler using it can't produce placements the env's grid-SDF
# will reject. Values are rounded up to 2 dp.
#
# Sign variants couldn't be measured (USD payload didn't resolve in the
# stand-alone bbox script); they keep a 0.2 m guess and should not be used as
# sampled obstacles.
PROP_CATALOG: dict[str, list[PropSpec]] = {
    "box": [
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd", 0.44),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd", 0.51),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd", 0.23),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_02.usd", 0.23),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_03.usd", 0.24),
    ],
    "pallet": [
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd", 0.79),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_02.usd", 0.73),
        PropSpec("/Isaac/Props/Pallet/pallet.usd", 0.73),
    ],
    "crate": [
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_A_01.usd", 0.40),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_01.usd", 0.35),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_C_01.usd", 0.46),
    ],
    "barrel": [
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd", 0.47),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd", 0.47),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_C_01.usd", 0.30),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_D_01.usd", 0.21),
    ],
    "cart": [
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd", 0.95),
    ],
    "forklift": [
        PropSpec("/Isaac/Props/Forklift/forklift.usd", 1.85),
        PropSpec("/Isaac/Samples/Rigging/Forklift/forklift_b_unrigged_cm.usd", 1.85),
    ],
    "bucket": [
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd", 0.48),
    ],
    "rack_pile": [
        # SM_RackPile.usd (variant 0) couldn't be measured (USD payload).
        # SM_RackPile_06.usd (variant 1) measured 1.72 m. Keep only the
        # measured variant for kinds we sample as obstacles.
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_06.usd", 1.72),
    ],
    "sign": [
        # Unmeasured (USD payload didn't resolve in the bbox script); 0.2 m
        # guess. Not suitable as a sampled obstacle.
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_SignA.usd", 0.2),
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_SignB.usd", 0.2),
    ],
    # --- goal-prop kinds ------------------------------------------------------
    # One catalog for goals and obstacles: the language task makes every object
    # both (target in one scene, distractor in another). `goal_kinds` (and
    # place_goal_or_designate) resolve from here.
    "traffic_cone": [
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd", 0.3),
    ],
    "fire_extinguisher": [
        # SM_FireExtinguisher_02's pivot is mid-body, not at the base, so
        # z_height=0 sinks the bottom half through the floor. ~0.25 m lifts
        # the base flush. Tune by eye if it still floats/clips.
        PropSpec("/Isaac/Environments/Simple_Warehouse/Props/SM_FireExtinguisher_02.usd", 0.2, z_height=0.25),
    ],
}


# ---------------------------------------------------------------------------
# Base config — every template's Config extends this
# ---------------------------------------------------------------------------

# Valid values for BaseTemplateConfig.start_heading_mode (validated in
# __post_init__, consumed via resolve_start_heading_rad below).
START_HEADING_MODES = ("astar", "fixed", "random")


def resolve_start_heading_rad(
    mode: str,
    start_heading_deg: float,
    seed: int,
    astar_heading_rad: float,
) -> float:
    """Pick the spawn yaw (radians) according to start_heading_mode.

    Pure function so it's unit-testable without booting Isaac Sim — the env's
    resolve_effective_start_and_goal_pose computes astar_heading_rad and
    delegates the mode choice here.

      "astar"  -> astar_heading_rad (face along the planned path;
                  leaks the planner's answer into the first observation)
      "fixed"  -> radians(start_heading_deg) from the scene yaml (incl. any
                  config-time jitter the template already applied to it)
      "random" -> seeded uniform in [-pi, pi); string-seeded stream so it is
                  decoupled from every other consumer of cfg.seed
    """
    if mode == "astar":
        return float(astar_heading_rad)
    if mode == "fixed":
        return math.radians(float(start_heading_deg))
    if mode == "random":
        rng = random.Random(f"{seed}-start-heading")
        return rng.uniform(-math.pi, math.pi)
    raise ValueError(
        f"unknown start_heading_mode {mode!r}. Valid: {START_HEADING_MODES}."
    )


@dataclass
class BaseTemplateConfig:
    name: str
    template: str           # which templates/<name>.py to dispatch to
    seed: int = 0

    # Robot spawn — server reads initial_pos / initial_ori from these.
    # Default `None` is a *sentinel* meaning "not yet set":
    #   - "Explicit-spawn" templates require the config to set these, and
    #     __post_init__ asserts non-None via super().
    #   - "Derived-spawn" templates compute them in __post_init__ and should
    #     refuse a config-set value so a stale one can't shadow the computed
    #     pose.
    start_xy: tuple[float, float] | None = None
    start_heading_deg: float | None = None

    # How the SPAWN yaw is chosen (consumed by IsaacSimEnv's
    # resolve_effective_start_and_goal_pose; see resolve_start_heading_rad):
    #   "astar"  — (default) face along the first K segments of the A* plan.
    #              Note: this leaks the planner's answer into the first
    #              observation — the robot starts pre-aligned with the
    #              correct route.
    #   "fixed"  — face start_heading_deg from the scene yaml (plus any
    #              config-time jitter the template applied to it).
    #   "random" — seeded uniform in [-pi, pi); RNG stream derived from
    #              cfg.seed but independent of other consumers.
    start_heading_mode: str = "astar"

    # Parking standoff — how far in front of the goal prop the robot should
    # stop:
    #   standoff_m = max(goal_radius_m + standoff_buffer_m, min_standoff_m)
    # where goal_radius_m is the prop's AABB half-extent. (IsaacSimEnv reads
    # the equivalent knobs from cfg.env.astar.)
    standoff_buffer_m: float = 0.3
    min_standoff_m: float = 0.5

    # A* obstacle inflation: minimum clearance the planned path keeps from any
    # obstacle's edge. Carter's chassis half-width is ~0.3 m; the default 0.4 m
    # adds ~10 cm of margin for diff-drive tracking error. Below ~0.32 m the
    # path can clip obstacles; much above ~1.0 m A* fails on cluttered envs.
    robot_inflation_m: float = 0.4

    # Unused config field, kept so existing scene configs still parse.
    record_realtime_scout: bool = False

    def __post_init__(self) -> None:
        """Base validation. Each template's __post_init__ must call
        super().__post_init__() at the END (after any derived-spawn
        computation) so this check sees the final value."""
        if self.start_xy is None or self.start_heading_deg is None:
            raise ValueError(
                f"Config {self.name!r}: start_xy / start_heading_deg unset "
                f"after template construction. Explicit-spawn templates must "
                f"declare them in JSON; derived-spawn templates must compute "
                f"them in __post_init__ before super()."
            )
        if self.start_heading_mode not in START_HEADING_MODES:
            raise ValueError(
                f"Config {self.name!r}: unknown start_heading_mode "
                f"{self.start_heading_mode!r}. Valid: {START_HEADING_MODES}."
            )

    @property
    def initial_pos(self) -> np.ndarray:
        return np.array([self.start_xy[0], self.start_xy[1], 0.0])

    @property
    def initial_ori(self) -> np.ndarray:
        h = math.radians(self.start_heading_deg)
        return np.array([math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)])

    @property
    def approach_heading_rad(self) -> float | None:
        """The direction the robot is *facing* when it arrives at the standoff.

        Default `None` means scouting falls back to atan2(goal − start) — the
        straight-line approach assumption, correct for any template whose
        trajectory really is a straight line (open-area runs).

        Override on aisle-mode templates where A* takes a non-direct route
        and the final approach is along the corridor axis, not the spawn-to-
        goal straight line. The standoff position is computed as
        `goal_xy − standoff_m · (cos(approach), sin(approach))`, so passing
        +π/2 puts the standoff *south* of the goal with the robot facing
        north — i.e. arriving "from below, looking up the aisle."
        """
        return None

    @property
    def approach_heading_candidates(self) -> list[float] | None:
        """Optional list of *alternative* approach headings to try.

        When the right approach direction depends on which side of an aisle
        is shorter to enter (e.g. aisle A → aisle B via the south or north
        exit), a template may return multiple candidates here; the shortest
        trial A* path decides.

        Default `None` → no selection step; use the single
        `approach_heading_rad` value. Returning a list of length 1 is
        equivalent to using `approach_heading_rad` directly.
        """
        return None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage helpers (call from inside template build() — kit must be running)
# ---------------------------------------------------------------------------

def load_base_warehouse(world, assets_root: str, prim_path: str = "/World/Warehouse") -> None:
    """Add the stock full_warehouse.usd as a reference under prim_path."""
    from isaacsim.core.utils.stage import add_reference_to_stage
    add_reference_to_stage(
        usd_path=assets_root + BASE_WAREHOUSE_USD_REL,
        prim_path=prim_path,
    )


def scale_baked_lights(stage, factor: float) -> list[tuple[str, float, float]]:
    """Walk all UsdLux prims and multiply their intensity by `factor`.

    Returns [(prim_path, original_intensity, new_intensity), ...] for logging.
    Use factor < 1 to dim, > 1 to brighten. Affects every light in the
    referenced warehouse USD as well.
    """
    from pxr import UsdLux

    light_types = (
        UsdLux.DistantLight, UsdLux.SphereLight, UsdLux.RectLight,
        UsdLux.DiskLight, UsdLux.DomeLight, UsdLux.CylinderLight,
    )
    changes: list[tuple[str, float, float]] = []
    for prim in stage.Traverse():
        for lt in light_types:
            if prim.IsA(lt):
                attr = lt(prim).GetIntensityAttr()
                if attr and attr.IsValid():
                    orig = attr.Get() or 0.0
                    new = float(orig) * float(factor)
                    attr.Set(new)
                    changes.append((str(prim.GetPath()), float(orig), new))
                break
    return changes


def hide_ceiling_lights(stage) -> int:
    """Hide the hanging ceiling-lamp FIXTURES (SM_LampCeilingA_*) — the visible
    lamp objects the oblique cameras see as long black bars. These are separable
    mesh prims, so making them invisible removes them from view without
    darkening the scene: illumination still comes from the DistantLight +
    RectLights (left untouched). Returns the number hidden.

    Gated by the HIDE_CEILING_LIGHTS env var in begin_build; off by default,
    so training/eval scenes are unaffected.
    """
    from pxr import UsdGeom

    n = 0
    for prim in stage.Traverse():
        if prim.GetName().startswith("SM_LampCeilingA"):
            UsdGeom.Imageable(prim).MakeInvisible()
            n += 1
    return n


def place_prop(stage, assets_root: str, prim_path: str,
               spec: PropSpec, x: float, y: float, yaw_deg: float = 0.0) -> None:
    """Reference a prop USD at the given path and set its world transform."""
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom, Gf

    add_reference_to_stage(usd_path=assets_root + spec.usd_path, prim_path=prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim) 
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(spec.z_height)))
    xform.AddRotateZOp().Set(float(yaw_deg))


# ---------------------------------------------------------------------------
# Warehouse layout — single source of truth for the stock full_warehouse.usd
# ---------------------------------------------------------------------------
# Derived from inspecting the NVIDIA stock warehouse asset. If you ever swap
# to a different base USD, re-derive these by listing the SM_RackShelf_*
# translates.

WAREHOUSE_SHELF_X_POSITIONS: tuple[float, ...] = (
    4.41, -0.55, -5.50, -10.46, -15.41, -20.37, -25.32,
)
WAREHOUSE_AISLE_X_CENTERS: tuple[float, ...] = tuple(
    (WAREHOUSE_SHELF_X_POSITIONS[i] + WAREHOUSE_SHELF_X_POSITIONS[i + 1]) / 2.0
    for i in range(len(WAREHOUSE_SHELF_X_POSITIONS) - 1)
)
# Aisles run along +Y between the shelf rows. South entry is just below the
# first shelf row (at y=10.9); north exit is just above the last row (y=22.9).
# A few decimetres of margin on each end so robots can enter/exit cleanly.
WAREHOUSE_AISLE_Y_SOUTH: float = 9.0
WAREHOUSE_AISLE_Y_NORTH: float = 25.0
WAREHOUSE_AISLE_LENGTH: float = WAREHOUSE_AISLE_Y_NORTH - WAREHOUSE_AISLE_Y_SOUTH
WAREHOUSE_NUM_AISLES: int = len(WAREHOUSE_AISLE_X_CENTERS)


def aisle_xy(aisle_index: int, depth_m: float) -> tuple[float, float]:
    """World xy for a point inside an aisle, measured along the corridor.

    `depth_m=0` is the south entry, `depth_m=WAREHOUSE_AISLE_LENGTH` is the
    north exit, `depth_m=WAREHOUSE_AISLE_LENGTH/2` is the centre.
    """
    if not (0 <= aisle_index < WAREHOUSE_NUM_AISLES):
        raise ValueError(
            f"aisle_index={aisle_index} out of range "
            f"[0, {WAREHOUSE_NUM_AISLES - 1}]."
        )
    return (WAREHOUSE_AISLE_X_CENTERS[aisle_index],
            WAREHOUSE_AISLE_Y_SOUTH + depth_m)


# ---------------------------------------------------------------------------
# Reshaped 2x3 "digital-8" warehouse — geometry for the procedural data
# generator (run_builders/build_2x3.py).
# ---------------------------------------------------------------------------
# These describe the baked layout in
# scenes/warehouse_layout/scenes/warehouse_2x3_base.usda (the reshape knobs the
# USDA was generated from). The USDA is frozen, so the values are stable.

_2X3_FRONT_DY = -24.0          # front-row shelves shifted south (whole shelves)
_2X3_GAP = 1.5                 # perimeter-wall <-> block gap on both E/W sides
_2X3_BLOCK_SHIFT_X = -1.2      # whole shelf block nudged west
_2X3_BACK_IDX = (0, 1, 2)      # stock shelf columns kept as the back row
_2X3_RACK_HALF_DEPTH_X = 0.55  # rack half-extent along x
_2X3_EAST_WALL_INNER_X = 5.3   # fixed east perimeter wall, inner face
_2X3_NORTH_WALL_Y = 30.3       # fixed north perimeter wall, inner face
_2X3_BACK_ROW_Y = (8.8, 25.0)  # back-row rack span in y (stock south/north edges)

# Shelf-column x's (both rows sit under the same x's). -> (3.21, -1.75, -6.70)
WAREHOUSE_2X3_COLUMN_XS: tuple[float, ...] = tuple(
    WAREHOUSE_SHELF_X_POSITIONS[i] + _2X3_BLOCK_SHIFT_X for i in _2X3_BACK_IDX
)

# The two N-S aisles, each centred between an adjacent column pair.
# -> east = 0.73, west = -4.225
WAREHOUSE_2X3_AISLE_X: dict[str, float] = {
    "east": (WAREHOUSE_2X3_COLUMN_XS[0] + WAREHOUSE_2X3_COLUMN_XS[1]) / 2.0,
    "west": (WAREHOUSE_2X3_COLUMN_XS[1] + WAREHOUSE_2X3_COLUMN_XS[2]) / 2.0,
}

# Clear (rack-face-to-rack-face) aisle width = column spacing - 2*rack half-depth.
# -> 4.955 - 1.10 = 3.855 m. Used for the blocker-seal arithmetic.
WAREHOUSE_2X3_AISLE_CLEAR_W: float = (
    abs(WAREHOUSE_2X3_COLUMN_XS[0] - WAREHOUSE_2X3_COLUMN_XS[1])
    - 2.0 * _2X3_RACK_HALF_DEPTH_X
)

# Interior free rectangle (inside the perimeter walls). x:(-8.75, 5.30), y:(-20.5, 30.3)
_2X3_WEST_WALL_X = (WAREHOUSE_2X3_COLUMN_XS[2] - _2X3_RACK_HALF_DEPTH_X) - _2X3_GAP
_2X3_SOUTH_WALL_Y = (
    (_2X3_BACK_ROW_Y[0] + _2X3_FRONT_DY)                  # front-row south edge (-15.2)
    - (_2X3_NORTH_WALL_Y - _2X3_BACK_ROW_Y[1])            # minus the mirrored north gap (5.3)
)
WAREHOUSE_2X3_INTERIOR_X: tuple[float, float] = (_2X3_WEST_WALL_X, _2X3_EAST_WALL_INNER_X)
WAREHOUSE_2X3_INTERIOR_Y: tuple[float, float] = (_2X3_SOUTH_WALL_Y, _2X3_NORTH_WALL_Y)

# Rack-row y spans. back = (8.8, 25.0); front = back + FRONT_DY = (-15.2, 1.0).
_2X3_FRONT_ROW_Y = (_2X3_BACK_ROW_Y[0] + _2X3_FRONT_DY, _2X3_BACK_ROW_Y[1] + _2X3_FRONT_DY)

# Navigable box: bounded E/W by the OUTER rack faces (so the ~1.5 m rack<->wall
# margins are NON-navigable) and N/S by the perimeter walls. The open zones A/D/G
# span this width. The A* navigable_bounds MASK (env/astar/warehouse_2x3.yaml,
# applied in build_grid_map / MapBuilder) marks everything OUTSIDE this box
# occupied — sealing the gap behind the shelves without fabricated wall prims
# (the stock perimeter walls are excluded from the A* map by name, so without this
# the margins are open phantom bypasses).
# -> x: (-7.25, 3.76), y: (-20.5, 30.3)
WAREHOUSE_2X3_NAV_X: tuple[float, float] = (
    WAREHOUSE_2X3_COLUMN_XS[2] - _2X3_RACK_HALF_DEPTH_X,   # west outer-rack west face
    WAREHOUSE_2X3_COLUMN_XS[0] + _2X3_RACK_HALF_DEPTH_X,   # east outer-rack east face
)
WAREHOUSE_2X3_NAV_Y: tuple[float, float] = WAREHOUSE_2X3_INTERIOR_Y


@dataclass(frozen=True)
class Zone2x3:
    """One free-space zone of the digital-8 layout: A/D/G are open strips,
    B/C/E/F the four transit aisles that can be sealed by an obstacle."""
    name: str                       # "A".."G"
    kind: str                       # "aisle" (B/C/E/F) or "open" (A/D/G)
    x_range: tuple[float, float]    # world x bounds (xmin, xmax)
    y_range: tuple[float, float]    # world y bounds (ymin, ymax)

    @property
    def center(self) -> tuple[float, float]:
        """World (x, y) of the zone centre."""
        return (0.5 * (self.x_range[0] + self.x_range[1]),
                0.5 * (self.y_range[0] + self.y_range[1]))


# The seven zones. A/D/G are full-width open strips; B/C/E/F are the aisle
# corridors (x spans only the clear aisle width, centred on the aisle).
_2X3_HALF_AISLE = WAREHOUSE_2X3_AISLE_CLEAR_W / 2.0
_2X3_W, _2X3_E = WAREHOUSE_2X3_AISLE_X["west"], WAREHOUSE_2X3_AISLE_X["east"]
WAREHOUSE_2X3_ZONES: dict[str, Zone2x3] = {
    "A": Zone2x3("A", "open",  WAREHOUSE_2X3_NAV_X,
                 (_2X3_BACK_ROW_Y[1], _2X3_NORTH_WALL_Y)),               # north strip (25.0, 30.3)
    "B": Zone2x3("B", "aisle", (_2X3_W - _2X3_HALF_AISLE, _2X3_W + _2X3_HALF_AISLE),
                 _2X3_BACK_ROW_Y),                                       # west-back aisle
    "C": Zone2x3("C", "aisle", (_2X3_E - _2X3_HALF_AISLE, _2X3_E + _2X3_HALF_AISLE),
                 _2X3_BACK_ROW_Y),                                       # east-back aisle
    "D": Zone2x3("D", "open",  WAREHOUSE_2X3_NAV_X,
                 (_2X3_FRONT_ROW_Y[1], _2X3_BACK_ROW_Y[0])),            # cross-aisle (1.0, 8.8)
    "E": Zone2x3("E", "aisle", (_2X3_W - _2X3_HALF_AISLE, _2X3_W + _2X3_HALF_AISLE),
                 _2X3_FRONT_ROW_Y),                                      # west-front aisle
    "F": Zone2x3("F", "aisle", (_2X3_E - _2X3_HALF_AISLE, _2X3_E + _2X3_HALF_AISLE),
                 _2X3_FRONT_ROW_Y),                                      # east-front aisle
    "G": Zone2x3("G", "open",  WAREHOUSE_2X3_NAV_X,
                 (_2X3_SOUTH_WALL_Y, _2X3_FRONT_ROW_Y[0])),             # south strip (-20.5, -15.2)
}

# Adjacency for the rule engine: which zones share a traversable border. B/C and
# E/F are NOT adjacent — they meet only through A/D/G (the digital-8 topology),
# which is what makes the `no_adjacent` rule force >=2-hop routes.
WAREHOUSE_2X3_ZONE_ADJACENCY: dict[str, frozenset[str]] = {
    "A": frozenset({"B", "C"}),
    "B": frozenset({"A", "D"}),
    "C": frozenset({"A", "D"}),
    "D": frozenset({"B", "C", "E", "F"}),
    "E": frozenset({"D", "G"}),
    "F": frozenset({"D", "G"}),
    "G": frozenset({"E", "F"}),
}


def warehouse_2x3_zone_at(x: float, y: float) -> str | None:
    """Return the name of the zone whose rectangle contains (x, y), else None.

    Rack columns sit at x in {3.21, -1.75, -6.70} — BETWEEN the aisle zones and in
    rack-row y-bands with no open zone — so rack CONTENTS never land in a zone.
    A prop that IS in a zone is therefore stray aisle/open-floor clutter.
    """
    for name, z in WAREHOUSE_2X3_ZONES.items():
        if z.x_range[0] <= x <= z.x_range[1] and z.y_range[0] <= y <= z.y_range[1]:
            return name
    return None


def warehouse_2x3_rack_rects(seg_len: float = 4.0) -> list[tuple[float, float, float, float]]:
    """Tiled occupancy rectangles for the 6 shelf racks (3 columns x 2 rows).

    Each rack is the strip x in [col +/- rack_half_depth], y over its row's span,
    tiled into <= seg_len pieces (a 16 m row would exceed the planner's cap). This
    is a derived approximation of the baked racks (check it against the live
    MapBuilder if the base USDA changes). This is the static occupancy
    the offline gate plans against (the gap behind the shelves is sealed by the A*
    navigable_bounds mask, not by rects).
    """
    hx = _2X3_RACK_HALF_DEPTH_X
    rects: list[tuple[float, float, float, float]] = []
    for col in WAREHOUSE_2X3_COLUMN_XS:
        for (y0, y1) in (_2X3_BACK_ROW_Y, _2X3_FRONT_ROW_Y):
            s = y0
            while s < y1 - 1e-9:
                e = min(s + seg_len, y1)
                rects.append((col - hx, s, col + hx, e))
                s = e
    return rects


def warehouse_2x3_static_rects() -> list[tuple[float, float, float, float]]:
    """The fixed-env occupancy the generator's gate plans against: just the racks.
    The rack<->wall margins (gap behind the shelves) are sealed by the A*
    navigable_bounds MASK (build_grid_map / MapBuilder), not by boundary rects.
    (Procedurally placed obstacles are appended on top.)"""
    return warehouse_2x3_rack_rects()


# ---------------------------------------------------------------------------
# Shared template infrastructure
# ---------------------------------------------------------------------------

LIGHTING_SCALE: dict[str, float] = {
    "dark":   0.33,
    "normal": 1.0,
    "bright": 3.0,
}


def validate_kinds(config) -> None:
    """Shared validation for fields every template config carries.

    Checks `lighting`, `obstacle_kinds`, `goal_kinds` against the catalogs.
    Call from each template's `__post_init__` after template-specific checks.
    """
    if config.lighting not in LIGHTING_SCALE:
        raise ValueError(
            f"Config {config.name!r}: unknown lighting {config.lighting!r}. "
            f"Choose from {sorted(LIGHTING_SCALE)}."
        )
    bad_obs = set(config.obstacle_kinds) - PROP_CATALOG.keys()
    if bad_obs:
        raise ValueError(
            f"Config {config.name!r}: unknown obstacle_kinds {sorted(bad_obs)}. "
            f"Valid: {sorted(PROP_CATALOG)}."
        )
    bad_goals = set(config.goal_kinds) - PROP_CATALOG.keys()
    if bad_goals:
        raise ValueError(
            f"Config {config.name!r}: unknown goal_kinds {sorted(bad_goals)}. "
            f"Valid: {sorted(PROP_CATALOG)}."
        )


def begin_build(world, assets_root: str, config, *,
                log_prefix: str) -> tuple[object, random.Random]:
    """Shared opening of every template's build(): load warehouse, scale
    lights, seed an rng, and return `(stage, rng)`.

    Templates call this first, then place their goal + obstacles using the
    returned stage and rng.
    """
    import omni.usd

    load_base_warehouse(world, assets_root)
    stage = omni.usd.get_context().get_stage()

    factor = LIGHTING_SCALE[config.lighting]
    changes = scale_baked_lights(stage, factor)
    print(f"{log_prefix} lighting={config.lighting} factor={factor:.2f} "
          f"(scaled {len(changes)} UsdLux prims)")

    import os
    if os.environ.get("HIDE_CEILING_LIGHTS"):
        n = hide_ceiling_lights(stage)
        print(f"{log_prefix} HIDE_CEILING_LIGHTS=1: hid {n} ceiling-light objects "
              f"(invisible to cameras; still illuminating)")

    rng = random.Random(config.seed)
    return stage, rng


def place_goal_or_designate(stage, assets_root: str, config, rng,
                            goal_xy: tuple[float, float], *,
                            log_prefix: str) -> None:
    """Place the goal prop at `goal_xy` if `config.goal_kinds` is non-empty,
    otherwise just print that the goal is a designated point.

    Empty `goal_kinds` is intentional: for image-mode tasks the goal scene
    often works better when it doesn't carry an "obstacle" visual prior
    from training.
    """
    gx, gy = goal_xy
    if config.goal_kinds:
        goal_specs = [s for k in config.goal_kinds for s in PROP_CATALOG[k]]
        goal_spec = rng.choice(goal_specs)
        place_prop(stage, assets_root, "/World/Goal", goal_spec,
                   gx, gy, config.goal_yaw_deg)
        print(f"{log_prefix} goal: {goal_spec.usd_path} at ({gx:.2f}, {gy:.2f})")
    else:
        print(f"{log_prefix} no goal prop placed (goal_kinds=[]); "
              f"goal point at ({gx:.2f}, {gy:.2f})")


# ---------------------------------------------------------------------------
# Obstacle sampling — "near-line + off-line" mix
# ---------------------------------------------------------------------------

@dataclass
class Placement:
    spec: PropSpec
    x: float
    y: float
    yaw_deg: float
    pool: str            # "near" or "off" — for logging / colouring later


def sample_obstacles_along_line(
    rng: random.Random,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    n_near: int,
    n_off: int,
    near_max_offset: float,
    corridor_half_width: float,
    obstacle_specs: list[PropSpec],
    min_clearance: float = 0.5,
    parallel_t_range: tuple[float, float] = (0.1, 0.9),
    near_min_offset: float = 0.4,
    reserved_zones: list[tuple[float, float, float]] = (),
    max_attempts: int = 50,
) -> list[Placement]:
    """Sample obstacle placements between start and goal.

    Two pools:
      - `n_near` obstacles placed close to the start->goal line (perpendicular
        offset in [near_min_offset, near_max_offset]). These create occlusions
        and force the robot to weave.
      - `n_off` obstacles placed wider (offset in [near_max_offset,
        corridor_half_width]). These are visual distractors that don't
        obstruct the direct path.

    Both pools are placed with the parallel coordinate (along the line) in
    `parallel_t_range`, so obstacles cluster between rather than around the
    endpoints.

    Reserved zones is a list of (x, y, radius) circles — typically the start
    and goal locations, plus their clear radii.
    """
    line_vec = np.array([goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]])
    line_len = float(np.linalg.norm(line_vec))
    if line_len < 1e-6:
        raise ValueError("start_xy == goal_xy; can't sample along zero-length line")
    line_dir = line_vec / line_len
    perp = np.array([-line_dir[1], line_dir[0]])   # left-perpendicular (unit)

    placed: list[Placement] = []
    reserved = list(reserved_zones)

    def too_close(x: float, y: float, r: float) -> bool:
        for rx, ry, rr in reserved:
            if math.hypot(x - rx, y - ry) < r + rr + min_clearance:
                return True
        for p in placed:
            if math.hypot(x - p.x, y - p.y) < r + p.spec.footprint_radius + min_clearance: 
                return True
        return False

    def try_pool(pool: str, count: int, off_range: tuple[float, float]) -> None:
        for i in range(count):
            spec = rng.choice(obstacle_specs)
            for _ in range(max_attempts):
                t = rng.uniform(*parallel_t_range)
                offset = rng.uniform(*off_range) * rng.choice((-1.0, 1.0)) #sample within the distance that we need to be in the close or dstiant range and flip the sign
                x = start_xy[0] + t * line_vec[0] + offset * perp[0]
                y = start_xy[1] + t * line_vec[1] + offset * perp[1]
                if not too_close(x, y, spec.footprint_radius):
                    placed.append(Placement(spec, x, y, rng.uniform(0.0, 360.0), pool))
                    break
            else:
                print(f"[obstacles] couldn't place {pool}-pool obstacle {i} "
                      f"({spec.usd_path}) after {max_attempts} attempts")

    try_pool("near", n_near, (near_min_offset, near_max_offset))
    try_pool("off",  n_off,  (near_max_offset, corridor_half_width))
    return placed


# ---------------------------------------------------------------------------
# Config dispatch
# ---------------------------------------------------------------------------

def load_config(
    path: Path | str,
    seed_override: int | None = None,
) -> BaseTemplateConfig:
    """Read a JSON config, look at its `template` field, hand off to the
    matching templates/<name>.py for dataclass construction.

    `seed_override` (if not None) replaces the JSON's `seed` field before
    construction — iterate seeds on a single config file without writing N
    copies of the same JSON. The override flows through __post_init__ so
    anything seeded from it gets recomputed.
    """
    from .templates import REGISTRY  # local import to avoid circular at module load

    with open(path) as f:
        raw = json.load(f)
    if seed_override is not None:
        raw["seed"] = int(seed_override)
    name = raw.get("template")
    if not name:
        raise ValueError(f"Env config {path} has no 'template' field.")
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown template {name!r} in {path}. Available: {sorted(REGISTRY)}."
        )
    return REGISTRY[name].Config.from_dict(raw)


def build(world, assets_root: str, config: BaseTemplateConfig) -> None:
    """Dispatch to the template module's build()."""
    from .templates import REGISTRY
    REGISTRY[config.template].build(world, assets_root, config)
