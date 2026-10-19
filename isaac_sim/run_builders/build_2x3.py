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
    WAREHOUSE_2X3_ZONE_ADJACENCY,
    WAREHOUSE_2X3_AISLE_CLEAR_W,
    WAREHOUSE_2X3_NAV_X,
    WAREHOUSE_2X3_NAV_Y,
    warehouse_2x3_static_rects,
)
from isaac_sim.planning.a_star.offline_gate import (
    rect_from_box, build_grid_map, has_path, path_length, aisle_sealed, path_points,
)

#setting up the direcotries to find the props etc that we need
_THIS = pathlib.Path(__file__).resolve().parent          # run_builders/
_NSC = _THIS.parent                                      # -> isaac_sim/
SCENE_DIR = _NSC / "configs" / "env" / "scene"
EXP_DIR = _NSC / "configs" / "experiment"
BASE_USDA = str(_NSC / "scenes" / "warehouse_layout" / "scenes" / "warehouse_2x3_base.usda")
FOOTPRINTS_JSON = _NSC / "utils" / "scene_prep" / "measured_footprints.json"

EXP_POLICY = "astar_oracle_nobrake"
EXP_CAMERA_SET = "with_4cctv_2x3_bigcam"
EXP_ASTAR = "warehouse_2x3"
EXP_RECORDER = "data_gen_nobrake"
CF_RUNS_ROOT_TMPL = "${{oc.env:INFRAVLA_ROOT}}/runs/data_gen/warehouse_2x3_{tag}_{date}"

#Rule set currently only holds one but could be extended to e.g. no aisle on the same line
RULES = {
    # No 1-hop pairs: goal must differ from start and not be a graph neighbour.
    "no_adjacent": lambda s, g, adj: g != s and g not in adj[s],
    # (future) "min_graph_distance", "must_cross_a_blocked_strip", ...
}


def _weighted_choice(items: list, rng: random.Random):
    """Samples froim a categorical distribution for objects"""
    weights = [float(it.get("weight", 1.0)) for it in items]
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for it, w in zip(items, weights):
        upto += w
        if r <= upto:
            return it
    return items[-1]


def _load_footprints() -> dict:
    """loads the footprint for each object with widht heigh and radius """
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


def _load_astar(cfg):
    """Loads the config for Astar and gets the robot infaltion"""
    astar = OmegaConf.load(_NSC / "configs" / "env" / "astar" / f"{cfg.astar_config}.yaml")
    astar.robot_inflation_m = float(cfg.robot.inflation_m)
    return astar


def _gate_cfg_from(astar) -> SimpleNamespace:
    """returns the config for Astar"""
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


class DataGenerator:
    """Data Generator class to generate data"""

    def __init__(self, cfg):
        """initializes the data generator by storing the configs, footprints, zones, as well as rules"""
        self.cfg = cfg
        self.gate_cfg = _gate_cfg_from(_load_astar(cfg))
        self.footprints = _load_footprints()
        self.static_rects = warehouse_2x3_static_rects()      # racks + boundary
        self.aisle_zones = [z for z, s in WAREHOUSE_2X3_ZONES.items() if s.kind == "aisle"]
        self.start_zones = list(cfg.zones.start)
        self.goal_zones = list(cfg.zones.goal)
        self.rules = list(cfg.zones.rules)
        self._validate()
        corners = [(WAREHOUSE_2X3_NAV_X[i], WAREHOUSE_2X3_NAV_Y[j])
                   for i in (0, 1) for j in (0, 1)]
        self.static_map = build_grid_map(self.static_rects, corners, self.gate_cfg)

    def _validate(self) -> None:
        """validates that all the rules and zones given by the config as wel as the objects are usable and correct."""
        valid = set(WAREHOUSE_2X3_ZONES)
        for z in self.start_zones + self.goal_zones:
            assert z in valid, f"unknown zone {z!r} (valid: {sorted(valid)})"
        for r in self.rules:
            assert r in RULES, f"unknown rule {r!r} (valid: {sorted(RULES)})"
        for spec in (self.cfg.object_pools.aisle.blockers or []):
            key = (spec["kind"], int(spec.get("variant", 0)))
            assert key in self.footprints, \
                f"object_pools.aisle.blockers entry {key} has no measured footprint"
        for p in self.cfg.counterfactual_v3.patterns:
            assert str(p) == "0" or all(z in self.aisle_zones for z in str(p)), \
                f"pattern {p!r} names a non-aisle zone"

    def _footprint(self, kind: str, variant: int) -> dict:
        """returns the footprint of a given prop identified via the kind and variant tuple"""
        return self.footprints[(kind, variant)]

    def _obstacle_rect(self, o: dict):
        """returns a rectangle for a given obstacle dictonary"""
        fp = self._footprint(o["kind"], o.get("variant", 0))
        return rect_from_box(o["x"], o["y"], fp["half_w"], fp["half_d"], o.get("yaw_deg", 0.0))

    def _clearance_at(self, x: float, y: float) -> float:
        """returns whether the map at given point is clear, checks bounds"""
        m = self.static_map
        r, c = m.get_map_coords((x, y))
        H, W = m.sdf.shape
        if not (0 <= r < H and 0 <= c < W):
            return 0.0
        return float(m.sdf[r, c])

    def _goal_ok(self, start: str, goal: str) -> bool:
        """Checks whether goal placement is according to rules"""
        return all(RULES[r](start, goal, WAREHOUSE_2X3_ZONE_ADJACENCY) for r in self.rules)

    def _sample_zone_pair(self, rng: random.Random):
        """samples start and goal zone and samples the goal whilst adhere ing to rules"""
        starts = self.start_zones[:]
        rng.shuffle(starts)
        for start in starts:
            legal = [g for g in self.goal_zones if g != start and self._goal_ok(start, g)]
            if legal:
                return start, rng.choice(legal)
        return None

    def _sample_point_in_zone(self, zone: str, rng: random.Random, clearance: float,
                              avoid: tuple = ()):
        """Sample a free point in a zone (for start or goal); None if no valid point is found."""
        #get the warehouse zones
        z = WAREHOUSE_2X3_ZONES[zone]
        #get the placmeen point margin m
        m = float(self.cfg.placement.point_margin_m)
        # we get the buffer for x coordinates which is different for aisles than for non asiles
        xin = (WAREHOUSE_2X3_AISLE_CLEAR_W / 2.0 - clearance) if z.kind == "aisle" else m
        x_lo, x_hi = z.x_range[0] + xin, z.x_range[1] - xin
        y_lo, y_hi = z.y_range[0] + m, z.y_range[1] - m
        #make sure the lowe and upper bounds are well defined
        if x_lo >= x_hi or y_lo >= y_hi:
            return None
        # now we sample n times
        for _ in range(int(self.cfg.placement.max_tries)):
            #get x and y coordinates uniformly sampled
            x, y = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
            #check whether coordinates are free
            if self._clearance_at(x, y) <= clearance:
                continue
            # Reject points too close to the `avoid` circles.
            if any(math.hypot(x - ax, y - ay) < clearance + ar for ax, ay, ar in avoid):
                continue
            return (round(x, 3), round(y, 3))
        return None

    def _make_blocker(self, zone: str, rng: random.Random):
        """returns a blocker object that has been samples"""
        #draw element from the list
        pool = list(self.cfg.object_pools.aisle.blockers)
        if not pool:
            return None
        spec = _weighted_choice(pool, rng)
        #get the kind and variant
        kind, variant = spec["kind"], int(spec.get("variant", 0))
        # ge the zone
        z = WAREHOUSE_2X3_ZONES[zone]
        ax = z.center[0]                                    # aisle centre x
        #get the margin value from config
        margin = float(self.cfg.placement.blocker_end_margin_m)
        # v3: y confined to the configured span of the aisle, fractions measured
        # from the SOUTH end (blocker_y_span_frac=[0.0, 0.5] = south half, ego-
        # hidden for the fixed A/D->G directions).
        f_lo, f_hi = (float(v) for v in self.cfg.placement.blocker_y_span_frac)
        y0, y1 = z.y_range
        span_lo = max(y0 + margin, y0 + f_lo * (y1 - y0))
        span_hi = min(y1 - margin, y0 + f_hi * (y1 - y0))
        if span_lo >= span_hi:
            return None
        y = round(rng.uniform(span_lo, span_hi), 3)
        yaw = 90.0 if spec.get("yaw_across", True) else 0.0  # across the N-S aisle
        #save the object
        o = {"kind": kind, "variant": variant, "x": round(ax, 3), "y": y,
             "yaw_deg": yaw, "role": "blocker"}
        #seal the aisle
        if not self._seals_aisle(zone, o):
            return None
        return o

    def _seals_aisle(self, zone: str, blocker: dict) -> bool:
        """True iff the blocker seals the aisle (no A* path past it inside the walled-in aisle)."""
        z = WAREHOUSE_2X3_ZONES[zone]
        x0, x1 = z.x_range
        y0, y1 = z.y_range
        t = 0.3                                      # rail thickness (any value > grid res seals)
        seg = self.gate_cfg.max_obstacle_half_extent_m   # < 2*cap -> a rail isn't dropped
        rails = []
        s = y0 - t
        # Wall the aisle in with rails so the only way from below to above the
        # blocker is past it: if A* then finds no path, the blocker seals the
        # aisle. Rails are tiled into pieces below the map's obstacle size cap,
        # which would otherwise drop them as building-scale geometry.
        while s < y1 + t - 1e-9:
            e = min(s + seg, y1 + t)
            rails.append((x0 - t, s, x0, e))         # left rail (tiled)
            rails.append((x1, s, x1 + t, e))         # right rail (tiled)
            s = e
        rails.append((x0, y0 - t, x1, y0))           # bottom cap
        rails.append((x0, y1, x1, y1 + t))           # top cap
        blk = self._obstacle_rect(blocker)
        r = self.gate_cfg.robot_inflation_m
        below = (z.center[0], max(y0 + 0.1, blk[1] - r - 0.1))
        above = (z.center[0], min(y1 - 0.1, blk[3] + r + 0.1))
        return aisle_sealed(rails, blk, below, above, self.gate_cfg)

    # --- route classification -------------------------------------------------

    def _aisles_on_points(self, pts: list) -> list:
        """returns the aisle zones a sampled path threads (most-traversed first)."""
        if not pts:
            return []
        min_pts = 3   # >= this many sampled path points inside an aisle => on-route
        counts = []
        for z in self.aisle_zones:
            zz = WAREHOUSE_2X3_ZONES[z]
            x0, x1 = zz.x_range
            y0, y1 = zz.y_range
            inside = sum(1 for (x, y) in pts if x0 < x < x1 and y0 < y < y1)
            if inside >= min_pts:
                counts.append((z, inside))
        #sort so that the most traversed aisle is first
        counts.sort(key=lambda t: -t[1])          # most-traversed aisle first (has no impact on code)
        return [z for z, _ in counts]

    def _route_aisles(self, start_xy, goal_xy) -> list:
        """routes the aisles for a given start and goal on optimal path without obstacles and returns which zones are being corssed."""
        pts = path_points(self.static_rects, start_xy, goal_xy, self.gate_cfg)
        #pts is array of waypoints sampled across a star path
        return self._aisles_on_points(pts)

    def _draw_start_heading(self, start_xy, goal_xy, rng) -> float:
        """Spawn yaw (deg). cfg.heading.start_heading selects the mode:
          - "fixed": constant cfg.heading.start_heading_deg, decoupled from the goal
            (e.g. 270 = A->G corridor axis, due south). No goal-direction cue in the
            heading -> the aisle choice must come from CCTV/ego, not orientation.
          - "bearing_to_goal": yaw points at the goal point (leak-free across a cf pair
            since both siblings share the goal, but angles slightly toward the goal x).
        start_heading_noise_deg adds uniform +/- jitter (0.0 = exact)."""
        mode = str(self.cfg.heading.start_heading)
        if mode == "fixed":
            base = float(self.cfg.heading.start_heading_deg)
        elif mode == "bearing_to_goal":
            base = math.degrees(math.atan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0]))
        else:
            raise ValueError(f"unknown start_heading mode {mode!r} (fixed | bearing_to_goal)")
        noise = float(self.cfg.heading.get("start_heading_noise_deg", 0.0))
        if noise > 0.0:
            base += rng.uniform(-noise, noise)
        return round(base % 360.0, 2)

    # --- v3 balanced counterfactual pair --------------------------------------

    def sample_family_v3(self, base_name: str, seed: int):
        """Builds one v3 family: (start, goal) + one shared forklift per aisle,
        A* routes for all 9 patterns, then the (uniform, forced-divergent)
        pattern pair. Returns [(scene, meta), (scene, meta)] or None (resample)."""
        rng = random.Random(seed)
        sg_clear = float(self.cfg.placement.start_goal_clearance_m)

        pair = self._sample_zone_pair(rng)
        if pair is None:
            return None
        start_zone, goal_zone = pair
        start_xy = self._sample_point_in_zone(start_zone, rng, sg_clear)
        goal_xy = self._sample_point_in_zone(goal_zone, rng, sg_clear)
        if start_xy is None or goal_xy is None:
            return None

        patterns = [str(p) for p in self.cfg.counterfactual_v3.patterns]

        # One gate-verified forklift PER AISLE, shared by the whole family: a
        # pattern's obstacles are a SUBSET selection, so siblings differ only in
        # which forklifts exist (never where one stands).
        routes = None
        for _ in range(int(self.cfg.placement.solvable_retries)):
            blockers = {}
            for z in self.aisle_zones:
                b = self._make_blocker(z, rng)
                if b is None:
                    break
                blockers[z] = b
            if len(blockers) != len(self.aisle_zones):
                continue
            cand_routes, cand_lens, feasible = {}, {}, True
            for p in patterns:
                rects = ([] if p == "0"
                         else [self._obstacle_rect(blockers[z]) for z in p])
                if not has_path(self.static_rects + rects, start_xy, goal_xy, self.gate_cfg):
                    feasible = False
                    break
                pts = path_points(self.static_rects + rects, start_xy, goal_xy, self.gate_cfg)
                cand_routes[p] = tuple(sorted(self._aisles_on_points(pts)))
                cand_lens[p] = path_length(self.static_rects + rects, start_xy, goal_xy,
                                           self.gate_cfg)
            if feasible:
                routes, lens = cand_routes, cand_lens
                break
        if routes is None:
            return None

        # Draw 1: uniform over all 9 (unbiased subsample, draw_index=1). Draw 2:
        # uniform over the patterns whose route differs from draw 1's ->
        # guaranteed divergence pair (incl. detour-vs-detour pairs).
        p1 = rng.choice(patterns)
        divergent = [p for p in patterns if routes[p] != routes[p1]]
        if not divergent:
            return None            # degenerate cell (no transit aisle) — resample base
        p2 = rng.choice(divergent)

        start_heading = self._draw_start_heading(start_xy, goal_xy, rng)
        lighting = rng.choice(list(self.cfg.lighting))
        len0 = lens["0"]

        out = []
        for draw_index, p in ((1, p1), (2, p2)):
            name = f"{base_name}_cf{p}"
            obstacles = [] if p == "0" else [blockers[z] for z in p]
            scene = {
                "template": "warehouse_2x3", "name": name, "base_usda": BASE_USDA,
                "seed": seed,
                "start_xy": list(start_xy), "start_heading_deg": start_heading,
                "start_heading_mode": str(self.cfg.heading.start_heading_mode),
                "goal_xy": list(goal_xy), "goal_kinds": [], "goal_yaw_deg": 0.0,
                "goal_clear_radius": sg_clear,
                "lighting": lighting,
                "obstacles": obstacles,
            }
            elong = (lens[p] / len0) if (len0 and lens[p]) else None
            meta = {
                "name": name, "base": base_name, "seed": seed,
                "draw_index": draw_index,          # 1 = the unbiased uniform draw
                "pattern": p,
                "start_zone": start_zone, "goal_zone": goal_zone,
                "start_xy": list(start_xy), "goal_xy": list(goal_xy),
                "transit_unblocked": list(routes["0"]),
                "route_aisles": list(routes[p]),
                "route_changed": routes[p] != routes["0"],
                "pair_pattern": (p2 if draw_index == 1 else p1),
                "path_len_m": round(lens[p], 2) if lens[p] else None,
                "elongation": round(elong, 3) if elong else None,
                "detour": bool(elong and elong > float(self.cfg.stats.detour_threshold)),
            }
            out.append((scene, meta))
        return out

    def generate_v3(self, n_bases: int, base_seed: int, start_index: int):
        """Generates n_bases balanced pair families (2 scenes each)."""
        scenes, metas = [], []
        skipped, n_done = 0, 0
        seed, attempts = base_seed, 0
        max_attempts = n_bases * 4 + 16
        while n_done < n_bases and attempts < max_attempts:
            attempts += 1
            res = self.sample_family_v3(
                f"warehouse_2x3_{start_index + n_done:04d}", seed)
            seed += 1
            if res is None:
                skipped += 1
                continue
            for scene, meta in res:
                scenes.append(scene)
                metas.append(meta)
            n_done += 1
        stats = _summarize_v3(metas, skipped, n_done)
        return scenes, metas, stats


def _summarize_v3(metas: list, skipped: int, n_bases: int) -> dict:
    """v3 stats + the sampling-bias audit: per-pattern draw counts and
    P(route_changed | pattern), split by draw index (draw 1 = unbiased
    reference), plus per-cell outcome balance."""
    if not metas:
        return {"n": 0, "n_bases": n_bases, "skipped": skipped}

    def _tab(rows):
        per = {}
        for m in rows:
            d = per.setdefault(m["pattern"], {"n": 0, "changed": 0})
            d["n"] += 1
            d["changed"] += int(m["route_changed"])
        return {p: {"n": d["n"], "p_changed": round(d["changed"] / d["n"], 3)}
                for p, d in sorted(per.items())}

    changed = [m for m in metas if m["route_changed"]]
    elongs = [m["elongation"] for m in changed if m["elongation"]]
    # per start-zone outcome balance: max P(outcome) is the blind-model floor
    cells = {}
    for m in metas:
        key = m["start_zone"]
        cells.setdefault(key, []).append(tuple(m["route_aisles"]))
    floor = {}
    for key, outs in cells.items():
        counts = {}
        for o in outs:
            counts[o] = counts.get(o, 0) + 1
        floor[key] = round(max(counts.values()) / len(outs), 3)
    return {
        "n": len(metas), "n_bases": n_bases, "skipped": skipped,
        "start_zone_mix": {z: sum(1 for m in metas if m["start_zone"] == z)
                           for z in sorted({m["start_zone"] for m in metas})},
        "route_changed_share": round(len(changed) / len(metas), 3),
        "divergence_pairs": n_bases,       # by construction: one per family
        "pattern_audit_draw1": _tab([m for m in metas if m["draw_index"] == 1]),
        "pattern_audit_draw2": _tab([m for m in metas if m["draw_index"] == 2]),
        "blind_floor_by_start_zone": floor,
        "changed_elongation": ({"min": round(min(elongs), 3), "max": round(max(elongs), 3),
                                "avg": round(sum(elongs) / len(elongs), 3)} if elongs else None),
        "detour_scenes": sum(1 for m in metas if m["detour"]),
    }


def _write_scene(scene: dict) -> str:
    """Generates a yaml for a given scene config"""
    name = scene["name"]
    (SCENE_DIR / f"{name}.yaml").write_text(
        "# @package env\n# AUTO-GENERATED by build_2x3.py. "
        "DO NOT hand-edit; re-run the generator.\n" + json.dumps(scene, indent=2) + "\n")
    return name


_EXP_HEADER = """# @package _global_
# AUTO-GENERATED by build_2x3.py — 2x3 warehouse data-gen, partition {k}/{n}.
# A* oracle over {m} warehouse_2x3 scenes, recording ego+4xCCTV+pose+action.
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
  notes: "2x3 procedural data-gen, partition {k}/{n}."
task:
  max_steps: {max_steps}
recorder:
  runs_root: "{runs_root}/{label}{k}"
runs:
"""


def _emit_experiments(scene_names: list, parts: int, runs_root: str, max_steps: int,
                      label: str = "part", contiguous: bool = False) -> None:
    """Write the partitioned data-gen experiment yamls for the given scenes."""
    buckets = {k: [] for k in range(parts)}
    if contiguous:
        per = math.ceil(len(scene_names) / parts)
        for j, nm in enumerate(scene_names):
            buckets[min(j // per, parts - 1)].append(nm)
    else:
        for j, nm in enumerate(scene_names):
            buckets[j % parts].append(nm)
    for k in range(parts):
        runs = "\n".join(
            f'  - {{ name: "{nm}", env_scene: {nm}, action: RELOAD }}'
            for nm in buckets[k])
        txt = _EXP_HEADER.format(
            k=k + 1, n=parts, m=len(scene_names), placeholder=scene_names[0],
            policy=EXP_POLICY, camera_set=EXP_CAMERA_SET, astar=EXP_ASTAR,
            recorder=EXP_RECORDER, runs_root=runs_root, max_steps=max_steps,
            label=label) + runs + "\n"
        (EXP_DIR / f"data_gen_2x3_{label}{k + 1}.yaml").write_text(txt)
        print(f"data_gen_2x3_{label}{k+1}.yaml: {len(buckets[k])} runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_NSC / "configs" / "datagen_2x3.yaml"))
    ap.add_argument("--date", default="0000-00-00", help="stamp for runs_root (no Date.now in scripts)")
    ap.add_argument("--tag", default="cfv3",
                    help="campaign tag -> runs_root warehouse_2x3_<tag>_<date>, "
                         "exp prefix <tag>_p, manifest datagen_2x3_<tag>_manifest.json.")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    gen = DataGenerator(cfg)
    cf = cfg.counterfactual_v3
    scenes, metas, stats = gen.generate_v3(int(cf.n_bases), int(cf.base_seed),
                                           int(cf.name_start_index))
    if not scenes:
        print("[build_2x3] generated 0 scenes — check zones/rules/config")
        return 1

    names = [_write_scene(s) for s in scenes]
    per_part = int(cfg.partition_size)
    parts = max(1, math.ceil(len(names) / per_part))
    runs_root = CF_RUNS_ROOT_TMPL.format(tag=args.tag, date=args.date)
    _emit_experiments(names, parts, runs_root, int(cfg.max_steps),
                      label=f"{args.tag}_p", contiguous=True)
    manifest = EXP_DIR / f"datagen_2x3_{args.tag}_manifest.json"
    manifest.write_text(json.dumps({"stats": stats, "scenes": metas}, indent=2))

    print(f"\n[build_2x3] v3: wrote {len(names)} scenes ({stats['n_bases']} balanced "
          f"pair families) in {parts} partitions")
    print(f"[build_2x3] stats: {json.dumps(stats, indent=2)}")
    print(f"[build_2x3] manifest: {manifest}")
    print(f"[build_2x3] runs_root: {runs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
