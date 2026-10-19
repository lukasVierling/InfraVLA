from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from types import SimpleNamespace

from omegaconf import OmegaConf

from isaac_sim.scenes._common import (
    WAREHOUSE_2X3_ZONES,
    WAREHOUSE_2X3_AISLE_CLEAR_W,
    WAREHOUSE_2X3_NAV_X,
    warehouse_2x3_static_rects,
)
from isaac_sim.planning.a_star.offline_gate import build_grid_map

_THIS = pathlib.Path(__file__).resolve().parent          # run_builders/
_NSC = _THIS.parent                                      # -> isaac_sim/
SCENE_DIR = _NSC / "configs" / "env" / "scene"
EXP_DIR = _NSC / "configs" / "experiment"
BASE_USDA = str(_NSC / "scenes" / "warehouse_layout" / "scenes" / "warehouse_2x3_base.usda")
FOOTPRINTS_JSON = _NSC / "utils" / "scene_prep" / "measured_footprints.json"

# Experiment-yaml constants (same stack as the current forklift campaign).
EXP_POLICY = "astar_oracle_nobrake"
EXP_CAMERA_SET = "with_4cctv_2x3_bigcam"
EXP_ASTAR = "warehouse_2x3"
EXP_RECORDER = "data_gen_nobrake"
RUNS_ROOT_TMPL = "${{oc.env:INFRAVLA_ROOT}}/runs/data_gen/warehouse_2x3_{tag}_{date}"


# --- tiny loaders (intentionally duplicated from build_2x3.py to keep the two
#     generators decoupled) -------------------------------------------------
def _load_footprints() -> dict:
    data = json.loads(FOOTPRINTS_JSON.read_text())
    out = {}
    for v in data["per_variant"]:
        ab = v["aabb_xy"]
        out[(v["kind"], v["variant_index"])] = {
            "half_w": float(ab["half_w"]),
            "half_d": float(ab["half_d"]),
            "radius": float(v["measured_footprint_radius"]),
        }
    return out


def _gate_cfg(cfg) -> SimpleNamespace:
    astar = OmegaConf.load(_NSC / "configs" / "env" / "astar" / f"{cfg.astar_config}.yaml")
    astar.robot_inflation_m = float(cfg.robot.inflation_m)
    return SimpleNamespace(
        grid_resolution_m=float(astar.grid_resolution_m),
        world_buffer_m=float(astar.world_buffer_m),
        max_obstacle_half_extent_m=float(astar.max_obstacle_half_extent_m),
        robot_inflation_m=float(astar.robot_inflation_m),
        alpha_clearance=float(astar.alpha_clearance),
        hill_climb_window_m=float(astar.hill_climb_window_m),
        pipeline=SimpleNamespace(sdf_weighted=bool(astar.pipeline.sdf_weighted),
                                 hill_climbing=False, cubic_spline=False),
        navigable_bounds=(list(astar.navigable_bounds)
                          if astar.get("navigable_bounds") is not None else None),
    )


class LangSceneGenerator:
    """Places one named-object-per-zone scenes for the D-spawn language task."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.footprints = _load_footprints()
        # Merge hand footprints for kinds with no MEASURED entry (e.g. traffic_cone).
        for kind, variants in (cfg.get("footprint_overrides") or {}).items():
            for variant, fp in variants.items():
                self.footprints[(kind, int(variant))] = {
                    "half_w": float(fp["half_w"]), "half_d": float(fp["half_d"]),
                    "radius": float(fp["radius"]),
                }
        self.gate_cfg = _gate_cfg(cfg)
        self.static_rects = warehouse_2x3_static_rects()
        # Rebuild the same static occupancy/SDF the forklift gate uses, for
        # placement-clearance queries only (no A* / oracle is run here).
        ymin = min(z.y_range[0] for z in WAREHOUSE_2X3_ZONES.values())
        ymax = max(z.y_range[1] for z in WAREHOUSE_2X3_ZONES.values())
        corners = [(WAREHOUSE_2X3_NAV_X[i], y) for i in (0, 1) for y in (ymin, ymax)]
        self.static_map = build_grid_map(self.static_rects, corners, self.gate_cfg)
        self.pool = [dict(p) for p in cfg.pool]
        self.object_zones = list(cfg.object_zones)
        self._validate()

    def _validate(self) -> None:
        valid = set(WAREHOUSE_2X3_ZONES)
        assert self.cfg.spawn_zone in valid, f"unknown spawn_zone {self.cfg.spawn_zone!r}"
        for z in self.object_zones:
            assert z in valid, f"unknown object zone {z!r}"
            assert z != self.cfg.spawn_zone, f"object zone {z!r} collides with spawn_zone"
        for p in self.pool:
            key = (p["kind"], int(p["variant"]))
            assert key in self.footprints, f"pool entry {key} has no measured footprint"
        assert len(self.pool) >= 1
        mode = str(self.cfg.get("family_mode", "goal_plus_distractors"))
        assert mode in ("goal_plus_distractors", "one_per_aisle"), \
            f"unknown family_mode {mode!r}"
        band = str(self.cfg.placement.get("aisle_band", "full"))
        assert band in ("full", "outer_half"), f"unknown aisle_band {band!r}"
        if mode == "one_per_aisle":
            # One kind per zone, every zone filled -> the pool must map 1:1
            # onto the object zones, and the outer-half band only makes sense
            # for aisle corridors (open zones have no "CCTV side").
            assert len(self.pool) == len(self.object_zones), (
                f"one_per_aisle needs len(pool)=={len(self.object_zones)} "
                f"(one kind per zone), got {len(self.pool)}")
            for z in self.object_zones:
                assert WAREHOUSE_2X3_ZONES[z].kind == "aisle", \
                    f"one_per_aisle object zone {z!r} is not an aisle"

    # --- geometry helpers ---------------------------------------------------
    def _cross_aisle_half(self, kind: str, variant: int, yaw_deg: float) -> float:
        """World-x half-extent of the object at the given yaw (what has to fit
        across a N-S aisle). yaw 0 -> half_w spans x; yaw 90 -> half_d spans x."""
        fp = self.footprints[(kind, int(variant))]
        return fp["half_w"] if int(round(yaw_deg)) % 180 == 0 else fp["half_d"]

    def _clearance_at(self, x: float, y: float) -> float:
        m = self.static_map
        r, c = m.get_map_coords((x, y))
        H, W = m.sdf.shape
        if not (0 <= r < H and 0 <= c < W):
            return 0.0
        return float(m.sdf[r, c])

    def _yaw_for(self, kind: str, zone_kind: str) -> float:
        """Forklift gets the along-aisle yaw in aisles (non-sealing); everything
        else is axis-aligned at 0."""
        if kind == "forklift" and zone_kind == "aisle":
            return float(self.cfg.forklift_aisle_yaw_deg)
        return 0.0

    def _place(self, zone: str, kind: str, variant: int, rng: random.Random):
        """Sample a valid (x, y, yaw) for `kind` in `zone`, or None."""
        z = WAREHOUSE_2X3_ZONES[zone]
        yaw = self._yaw_for(kind, z.kind)
        cmargin = float(self.cfg.placement.clearance_margin_m)
        cross_half = self._cross_aisle_half(kind, variant, yaw)
        need = cross_half + cmargin

        if z.kind == "aisle":
            # Sample x ACROSS the aisle too (2026-07-19, Lukas): objects may sit
            # anywhere between the rack faces — e.g. a box leaning against a
            # shelf. More realistic than pinning to the centreline, and transit
            # past the object gets easier (one side stays wide open; a hugging
            # object can never seal the aisle). The SDF check enforces the
            # rack gap: clearance >= cross_half + clearance_margin_m.
            m = float(self.cfg.placement.aisle_end_margin_m)
            x_lo, x_hi = z.x_range[0] + need, z.x_range[1] - need
            y_lo, y_hi = z.y_range[0] + m, z.y_range[1] - m
            # v2 'outer_half' band: keep only the aisle half FARTHER from the
            # D cross-aisle = closer to that aisle's bigcam (B/C cams on the
            # north wall, E/F cams on the south wall) -> north half of B/C,
            # south half of E/F. Far from D also means the object cannot leak
            # into the onboard view from the spawn (Lukas 2026-07-22).
            if str(self.cfg.placement.get("aisle_band", "full")) == "outer_half":
                y_mid = 0.5 * (z.y_range[0] + z.y_range[1])
                d_center_y = WAREHOUSE_2X3_ZONES[self.cfg.spawn_zone].center[1]
                if z.center[1] > d_center_y:
                    y_lo = max(y_lo, y_mid)    # north aisle: keep [mid, north end)
                else:
                    y_hi = min(y_hi, y_mid)    # south aisle: keep (south end, mid]
            if x_lo >= x_hi or y_lo >= y_hi:
                return None
            for _ in range(int(self.cfg.placement.max_tries)):
                x, y = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
                if self._clearance_at(x, y) < need:
                    continue
                return (round(x, 3), round(y, 3), yaw)
            return None

        # open zone (A/G): sample x,y in the navigable strip with margins.
        rad = self.footprints[(kind, int(variant))]["radius"]
        m = float(self.cfg.placement.open_margin_m)
        x_lo, x_hi = z.x_range[0] + m, z.x_range[1] - m
        y_lo, y_hi = z.y_range[0] + m, z.y_range[1] - m
        if x_lo >= x_hi or y_lo >= y_hi:
            return None
        for _ in range(int(self.cfg.placement.max_tries)):
            x, y = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
            if self._clearance_at(x, y) < rad + cmargin:
                continue
            return (round(x, 3), round(y, 3), 0.0)
        return None

    # --- family sampling (one placement, two counterfactual twins) ----------
    def sample_family(self, base_name: str, seed: int):
        """One placement -> TWO scenes (`_tA`, `_tB`) that share every object
        pose byte-for-byte and differ ONLY in which object is the target
        (role flags + goal_xy + instruction). The purest language A/B: pixels
        identical, label different. Returns [(scene, meta) x2] or None."""
        rng = random.Random(seed)

        goal_zone = rng.choice(self.object_zones)
        kinds = self.pool[:]
        rng.shuffle(kinds)                     # random kind order = unbiased draw
        target = kinds.pop()                   # twin-A target kind
        remaining = kinds                      # distractor pool (target excluded)

        objects, placed = [], {}               # role-neutral placements

        # twin-A target first, in the goal zone (always present)
        p = self._place(goal_zone, target["kind"], target["variant"], rng)
        if p is None:
            return None                        # unplaceable target -> resample
        x, y, yaw = p
        objects.append({"kind": target["kind"], "variant": int(target["variant"]),
                        "x": x, "y": y, "yaw_deg": yaw, "zone": goal_zone})
        placed[goal_zone] = target["kind"]

        # distractors: each other object-zone fires with distractor_prob; kinds
        # drawn WITHOUT replacement (no kind repeats -> a duplicate can never
        # signal "not the goal", and every object is uniquely referable).
        pdist = float(self.cfg.distractor_prob)
        for zone in self.object_zones:
            if zone == goal_zone:
                continue
            if rng.random() >= pdist:
                continue
            if not remaining:
                break
            spec = remaining.pop()             # already shuffled
            q = self._place(zone, spec["kind"], spec["variant"], rng)
            if q is None:
                continue                        # unplaceable distractor -> skip zone
            x, y, yaw = q
            objects.append({"kind": spec["kind"], "variant": int(spec["variant"]),
                            "x": x, "y": y, "yaw_deg": yaw, "zone": zone})
            placed[zone] = spec["kind"]

        # A twin needs a second referable object. P(zero distractors at p=0.5,
        # 5 zones) ~3% -> just resample the family.
        if len(objects) < 2:
            return None
        # twin-B target: uniform among the OTHER placed objects.
        alt = rng.choice(objects[1:])
        # Drawn ONCE per family — twins must share lighting to stay pixel-identical.
        lighting = str(rng.choice(list(self.cfg.lighting)))

        spawn = self._spawn_pose()
        out = []
        for suffix, tgt in (("tA", objects[0]), ("tB", alt)):
            name = f"{base_name}_{suffix}"
            # Per-twin role assignment over the SHARED placements. The target
            # rides at /World/Goal (template routes role=="target"), so the two
            # twins differ in A*-map content but not in any placed pose.
            obstacles = [{"kind": o["kind"], "variant": o["variant"],
                          "x": o["x"], "y": o["y"], "yaw_deg": o["yaw_deg"],
                          "role": ("target" if o is tgt else "distractor")}
                         for o in objects]
            # Language label: pattern drawn per twin, AFTER all placement draws,
            # so the pattern pool can change without moving any object.
            phrase = str(self.cfg.lang.phrases[tgt["kind"]])
            pattern = rng.choice(list(self.cfg.lang.patterns))
            scene = {
                "template": "warehouse_2x3", "name": name, "base_usda": BASE_USDA,
                "seed": seed,
                "start_xy": list(spawn["xy"]),
                "start_heading_deg": spawn["heading_deg"],
                "start_heading_mode": spawn["heading_mode"],
                "goal_xy": [tgt["x"], tgt["y"]], "goal_kinds": [],
                "goal_yaw_deg": 0.0,
                "goal_clear_radius": float(self.cfg.goal_clear_radius_m),
                "lighting": lighting,
                "obstacles": obstacles,
            }
            meta = {
                "name": name, "family": base_name, "twin": suffix,
                "pair_name": f"{base_name}_{'tB' if suffix == 'tA' else 'tA'}",
                "seed": seed,
                "goal_zone": tgt["zone"], "target_kind": tgt["kind"],
                "target_variant": int(tgt["variant"]),
                "target_xy": [tgt["x"], tgt["y"]],
                "target_phrase": phrase, "lang_pattern": pattern,
                "instruction": pattern.format(x=phrase),
                "zone_objects": placed,         # zone -> kind (shared by twins)
                "n_objects": len(objects),
                "spawn_xy": list(spawn["xy"]),
                "spawn_heading_deg": spawn["heading_deg"],
            }
            out.append((scene, meta))
        return out

    def sample_family_one_per_aisle(self, base_name: str, seed: int):
        """v2 family (2026-07-22): shuffle the pool over object_zones so EVERY
        aisle holds exactly one object (kind permutation + exact positions =
        the family's identity), then emit one twin scene per placed object as
        the target (tA..tD, ordered by object_zones). All twins share every
        placement + lighting byte-for-byte and differ ONLY in role flags,
        goal_xy and instruction. Returns [(scene, meta) x len(pool)] or None."""
        rng = random.Random(seed)

        kinds = self.pool[:]
        rng.shuffle(kinds)                     # permutation over the aisles

        objects = []                           # role-neutral placements
        for zone, spec in zip(self.object_zones, kinds):
            p = self._place(zone, spec["kind"], spec["variant"], rng)
            if p is None:
                return None                    # any unplaceable object -> resample family
            x, y, yaw = p
            objects.append({"kind": spec["kind"], "variant": int(spec["variant"]),
                            "x": x, "y": y, "yaw_deg": yaw, "zone": zone})
        placed = {o["zone"]: o["kind"] for o in objects}
        # Drawn ONCE per family — twins must share lighting to stay pixel-identical.
        lighting = str(rng.choice(list(self.cfg.lighting)))

        spawn = self._spawn_pose()
        suffixes = [f"t{chr(ord('A') + i)}" for i in range(len(objects))]  # tA..tD
        out = []
        for suffix, tgt in zip(suffixes, objects):
            name = f"{base_name}_{suffix}"
            # Per-twin role assignment over the SHARED placements (same
            # /World/Goal routing as v1: role=="target" never enters the A* map).
            obstacles = [{"kind": o["kind"], "variant": o["variant"],
                          "x": o["x"], "y": o["y"], "yaw_deg": o["yaw_deg"],
                          "role": ("target" if o is tgt else "distractor")}
                         for o in objects]
            # Language label: pattern drawn per twin, AFTER all placement draws,
            # so the pattern pool can change without moving any object.
            phrase = str(self.cfg.lang.phrases[tgt["kind"]])
            pattern = rng.choice(list(self.cfg.lang.patterns))
            scene = {
                "template": "warehouse_2x3", "name": name, "base_usda": BASE_USDA,
                "seed": seed,
                "start_xy": list(spawn["xy"]),
                "start_heading_deg": spawn["heading_deg"],
                "start_heading_mode": spawn["heading_mode"],
                "goal_xy": [tgt["x"], tgt["y"]], "goal_kinds": [],
                "goal_yaw_deg": 0.0,
                "goal_clear_radius": float(self.cfg.goal_clear_radius_m),
                "lighting": lighting,
                "obstacles": obstacles,
            }
            meta = {
                "name": name, "family": base_name, "twin": suffix,
                # 4 twins -> a single pair_name is ill-defined; list the others.
                "sibling_names": [f"{base_name}_{s}" for s in suffixes if s != suffix],
                "seed": seed,
                "goal_zone": tgt["zone"], "target_kind": tgt["kind"],
                "target_variant": int(tgt["variant"]),
                "target_xy": [tgt["x"], tgt["y"]],
                "target_phrase": phrase, "lang_pattern": pattern,
                "instruction": pattern.format(x=phrase),
                "zone_objects": placed,         # zone -> kind (shared by twins)
                "n_objects": len(objects),
                "spawn_xy": list(spawn["xy"]),
                "spawn_heading_deg": spawn["heading_deg"],
            }
            out.append((scene, meta))
        return out

    def _spawn_pose(self) -> dict:
        z = WAREHOUSE_2X3_ZONES[self.cfg.spawn_zone]
        mode = str(self.cfg.spawn.mode)
        if mode == "center":
            xy = (z.center[0], z.center[1])
        elif mode == "west_wall":
            # Back against D's NAVIGABLE west edge (NAV_X[0], not the perimeter
            # wall — the rack<->wall margin is masked non-navigable in the A*
            # map), pushed in by wall_buffer_m so the spawn cell is plannable.
            xy = (WAREHOUSE_2X3_NAV_X[0] + float(self.cfg.spawn.wall_buffer_m),
                  z.center[1])
        else:
            raise NotImplementedError(f"spawn.mode={mode!r} not implemented")
        return {"xy": (round(xy[0], 3), round(xy[1], 3)),
                "heading_deg": float(self.cfg.spawn.heading_deg),
                "heading_mode": str(self.cfg.spawn.heading_mode)}

    def generate(self):
        scenes, metas = [], []
        seed = int(self.cfg.base_seed)
        idx = int(self.cfg.name_start_index)
        prefix = str(self.cfg.get("name_prefix", "warehouse_2x3_lang"))
        mode = str(self.cfg.get("family_mode", "goal_plus_distractors"))
        attempts, made = 0, 0
        n_fam = int(self.cfg.n_families)
        max_attempts = n_fam * 4 + 16
        while made < n_fam and attempts < max_attempts:
            attempts += 1
            base = f"{prefix}_{idx + made:04d}"
            if mode == "one_per_aisle":
                res = self.sample_family_one_per_aisle(base, seed)
            else:
                res = self.sample_family(base, seed)
            seed += 1
            if res is None:
                continue
            for scene, meta in res:
                scenes.append(scene)
                metas.append(meta)
            made += 1
        return scenes, metas


def _write_scene(scene: dict) -> str:
    name = scene["name"]
    (SCENE_DIR / f"{name}.yaml").write_text(
        "# @package env\n# AUTO-GENERATED by build_2x3_lang.py. "
        "DO NOT hand-edit; re-run the generator.\n" + json.dumps(scene, indent=2) + "\n")
    return name


# Deliberately duplicated from build_2x3.py (the two generators stay decoupled).
# Same launcher contract: launchers/run_datagen_partition.sh.
_EXP_HEADER = """# @package _global_
# AUTO-GENERATED by build_2x3_lang.py — 2x3 LANGUAGE-goal data-gen, partition {k}/{n}.
# A* oracle over {m} warehouse_2x3_lang scenes (counterfactual twin families),
# recording ego+4x bigcam CCTV+pose+action. Instructions live in the manifest
# (datagen_2x3_<tag>_manifest.json), joined on scene name at training time.
# Launch: bash launchers/run_datagen_partition.sh data_gen_2x3_{label}{k}
defaults:
  - override /policy: {policy}
  - override /task: pose
  - override /env/scene: {placeholder}
  - override /env/robot: nova_carter
  - override /env/camera_set: {camera_set}
  - override /env/astar: {astar}
  - override /controller: data_gen_default
  - override /recorder: {recorder}
  - _self_
run:
  name: "datagen_warehouse_2x3_{label}{k}"
  notes: "2x3 language-goal data-gen, partition {k}/{n}."
task:
  max_steps: {max_steps}
# Data-gen REQUIRES the CCTV frames (they are the training input). Force the
# logging flags here so a working-tree flip of the shared camera yamls (e.g.
# an eval campaign saving disk) can never silently yield an ego-only dataset —
# exactly what happened 2026-07-20 (bigcam yamls flipped to logging: false,
# first 12-scene batch recorded no CCTV; see potential_bugs.md).
env:
  cameras:
    cctv_NE: {{logging: true}}
    cctv_NW: {{logging: true}}
    cctv_SE: {{logging: true}}
    cctv_SW: {{logging: true}}
recorder:
  runs_root: "{runs_root}/{label}{k}"
runs:
"""


def _emit_experiments(scene_names: list, parts: int, runs_root: str,
                      max_steps: int, label: str) -> None:
    """Emit one experiment yaml per partition (contiguous split, so a family's
    two twins land in the same partition and share a tmux/GPU session)."""
    per = math.ceil(len(scene_names) / parts)
    for k in range(parts):
        bucket = scene_names[k * per:(k + 1) * per]
        if not bucket:
            continue
        runs = "\n".join(
            f'  - {{ name: "{nm}", env_scene: {nm}, action: RELOAD }}'
            for nm in bucket)
        txt = _EXP_HEADER.format(
            k=k + 1, n=parts, m=len(scene_names), placeholder=scene_names[0],
            policy=EXP_POLICY, camera_set=EXP_CAMERA_SET, astar=EXP_ASTAR,
            recorder=EXP_RECORDER, runs_root=runs_root, max_steps=max_steps,
            label=label) + runs + "\n"
        (EXP_DIR / f"data_gen_2x3_{label}{k + 1}.yaml").write_text(txt)
        print(f"  data_gen_2x3_{label}{k + 1}.yaml: {len(bucket)} runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_NSC / "configs" / "datagen_2x3_lang.yaml"))
    ap.add_argument("--date", default="0000-00-00",
                    help="stamp for runs_root (no Date.now in scripts)")
    ap.add_argument("--tag", default="lang",
                    help="campaign tag -> runs_root warehouse_2x3_<tag>_<date>, "
                         "exp prefix <tag>_p")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    gen = LangSceneGenerator(cfg)
    scenes, metas = gen.generate()
    if not scenes:
        print("[build_2x3_lang] generated 0 scenes — check zones/pool/config")
        return 1

    names = [_write_scene(s) for s in scenes]
    per_part = int(cfg.partition_size)
    parts = max(1, math.ceil(len(names) / per_part))
    runs_root = RUNS_ROOT_TMPL.format(tag=args.tag, date=args.date)
    _emit_experiments(names, parts, runs_root, int(cfg.max_steps),
                      label=f"{args.tag}_p")
    # Manifest keyed by --tag so a new campaign can never clobber an older
    # manifest a training config still points at (default tag "lang" keeps the
    # v1 filename byte-identical).
    manifest = EXP_DIR / f"datagen_2x3_{args.tag}_manifest.json"
    manifest.write_text(json.dumps({"scenes": metas}, indent=2))

    n_fam = len({m["family"] for m in metas})
    print(f"[build_2x3_lang] wrote {len(names)} scenes ({n_fam} twin families) "
          f"in {parts} partition(s) -> {SCENE_DIR}")
    for m in metas:
        objs = ", ".join(f"{z}:{k}" for z, k in m["zone_objects"].items())
        print(f"  {m['name']}: '{m['instruction']}' @ {m['goal_zone']}  |  {objs}")
    print(f"[build_2x3_lang] manifest: {manifest}")
    print(f"[build_2x3_lang] runs_root: {runs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
