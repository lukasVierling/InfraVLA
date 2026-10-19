from __future__ import annotations
import argparse, json, math, pathlib
from omegaconf import OmegaConf

from isaac_sim.scenes._common import WAREHOUSE_2X3_ZONES, WAREHOUSE_2X3_NAV_X
from isaac_sim.planning.a_star.offline_gate import has_path, path_length, path_points
from isaac_sim.run_builders.build_2x3 import DataGenerator, BASE_USDA
from isaac_sim.run_builders.build_2x3_grid72 import grid_xs, POSITIONS

_NSC = pathlib.Path(__file__).resolve().parent.parent
SCENE_DIR = _NSC / "configs" / "env" / "scene"
EXP_DIR = _NSC / "configs" / "experiment"
GRID_CFG = _NSC / "configs" / "eval_grid72_2x3.yaml"
DATAGEN_CFG = _NSC / "configs" / "datagen_2x3.yaml"

# Each eval arm = (policy config, camera_set override). The camera_set matters
# ONLY for CCTV arms — cfv3 was trained on BIGCAM (16 mm) CCTVs, so the CCTV arm
# MUST render with `with_4cctv_2x3_bigcam` (same camera names, wider FOV) or the
# onboard-vs-CCTV FOV mismatch corrupts the CCTV stream. The ego-only baseline
# ignores the CCTVs entirely, so its camera_set is irrelevant (kept standard).
ARMS = {
    "baseline": {
        "policy": "omnivla_finetuned_baseline_2x3cfv3_fg50_r16",
        # front-cam ONLY: the ego baseline reads front_cam and nothing else, and
        # front_cam == carter_front in every camera set — so rendering the 4 CCTVs
        # is pure wasted compute (rollout cost is render-dominated). `default` is
        # the 1-camera front-only set; results are identical, rollouts ~5x fewer
        # camera renders/query.
        "camera_set": "default",
    },
    "avgpool": {
        "policy": "omnivla_finetuned_cctv4_2x3cfv3_avgpool64_fg50_r16",
        "camera_set": "with_4cctv_2x3_bigcam",
    },
}


def make_blocker_endseal(zone: str, gen: DataGenerator) -> dict:
    """Forklift at the SOUTH-THIRD MIDPOINT of `zone` — the deterministic centre
    of the training sampling span [y0+margin, y0+0.333*span]."""
    z = WAREHOUSE_2X3_ZONES[zone]
    y0, y1 = z.y_range
    margin = float(gen.cfg.placement.blocker_end_margin_m)
    f_lo, f_hi = (float(v) for v in gen.cfg.placement.blocker_y_span_frac)
    span_lo = max(y0 + margin, y0 + f_lo * (y1 - y0))
    span_hi = min(y1 - margin, y0 + f_hi * (y1 - y0))
    y = round(0.5 * (span_lo + span_hi), 3)          # deterministic midpoint
    return {"kind": "forklift", "variant": 0,
            "x": round(z.center[0], 3), "y": y, "yaw_deg": 90.0, "role": "blocker"}


def build_grid(grid_cfg, gen: DataGenerator) -> list:
    xs = grid_xs(float(grid_cfg.edge_margin_m))
    spawn_y = float(grid_cfg.spawn_y)
    goal_y = round(WAREHOUSE_2X3_ZONES["G"].center[1], 3)
    sg_clear = float(gen.cfg.placement.start_goal_clearance_m)
    out = []
    for s_pos in POSITIONS:
        for g_pos in POSITIONS:
            start_xy = (xs[s_pos], spawn_y); goal_xy = (xs[g_pos], goal_y)
            assert path_length(gen.static_rects, start_xy, goal_xy, gen.gate_cfg), \
                f"no empty path s{s_pos}->g{g_pos}"
            for cond in list(grid_cfg.conditions):
                zones = list(str(cond))
                blockers = [make_blocker_endseal(z, gen) for z in zones]
                for z, b in zip(zones, blockers):
                    assert gen._seals_aisle(z, b), f"end-seal forklift fails to seal {z}"
                rects = [gen._obstacle_rect(b) for b in blockers]
                assert has_path(gen.static_rects + rects, start_xy, goal_xy, gen.gate_cfg), \
                    f"s{s_pos}->g{g_pos} cf{cond}: unreachable"
                pts = path_points(gen.static_rects + rects, start_xy, goal_xy, gen.gate_cfg)
                route = sorted(gen._aisles_on_points(pts))
                assert not set(route) & set(zones), f"cf{cond}: route threads a seal"
                name = f"warehouse_2x3_gridES_s{s_pos}_g{g_pos}_cf{cond}"
                scene = {"template": "warehouse_2x3", "name": name, "base_usda": BASE_USDA,
                         "seed": int(grid_cfg.seed), "start_xy": list(start_xy),
                         "start_heading_deg": float(grid_cfg.start_heading_deg),
                         "start_heading_mode": "fixed", "goal_xy": list(goal_xy),
                         "goal_kinds": [], "goal_yaw_deg": 0.0, "goal_clear_radius": sg_clear,
                         "lighting": str(grid_cfg.lighting), "obstacles": blockers}
                out.append(name if False else (scene, name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--arm", choices=sorted(ARMS), default="baseline",
                    help="which eval arm to emit shards for (scenes are arm-agnostic)")
    args = ap.parse_args()
    arm = ARMS[args.arm]
    policy, camera_set = arm["policy"], arm["camera_set"]
    grid_cfg = OmegaConf.load(GRID_CFG)
    gen = DataGenerator(OmegaConf.load(DATAGEN_CFG))
    pairs = build_grid(grid_cfg, gen)
    names = []
    for scene, name in pairs:
        (SCENE_DIR / f"{name}.yaml").write_text(
            "# @package env\n# AUTO end-seal grid72 (forklift at south-third, "
            "training-matched). DO NOT hand-edit.\n" + json.dumps(scene, indent=2) + "\n")
        names.append(name)
    # per-arm experiment shards (scenes above are shared across arms)
    runs_root = f"${{oc.env:INFRAVLA_ROOT}}/runs/evaluation/sweeps/warehouse_2x3_gridES_{args.date}/{args.arm}"
    per = math.ceil(len(names) / args.shards)
    for k in range(args.shards):
        shard = names[k * per:(k + 1) * per]
        if not shard:
            continue
        hdr = f"""# @package _global_
# AUTO end-seal grid72 {args.arm} eval shard {k+1}/{args.shards}.
defaults:
  - override /policy: {policy}
  - override /task: pose
  - override /env/scene: {shard[0]}
  - override /env/robot: nova_carter
  - override /env/camera_set: {camera_set}
  - override /env/astar: warehouse_2x3
  - override /controller: omnivla_default
  - override /recorder: default
run: {{name: "eval_2x3_gridES_{args.arm}_p{k+1}", notes: "end-seal grid72 {args.arm} shard {k+1}."}}
control_hz: 3.0
record_hz: 3.0
fine_grained: {{enabled: true, control_hz: 30}}
task: {{max_steps: 300}}
recorder: {{runs_root: "{runs_root}", log_frames: false}}
runs:"""
        L = [hdr]
        for j, nm in enumerate(shard):
            act = f"policy: {policy}, action: RELOAD" if j == 0 else "action: KEEP_POLICY"
            L.append(f'  - {{ name: "{nm}__pose", env_scene: {nm}, task: pose, {act} }}')
        (EXP_DIR / f"eval_2x3_gridES_{args.arm}_p{k+1}.yaml").write_text("\n".join(L) + "\n")
    print(f"[gridES] arm={args.arm} policy={policy} camera_set={camera_set}")
    print(f"[gridES] wrote {len(names)} scenes + {args.shards} {args.arm} shards")
    print(f"[gridES] forklift south-third: back y~{make_blocker_endseal('B', gen)['y']}, "
          f"front y~{make_blocker_endseal('E', gen)['y']} (vs stock centre 16.9 / -7.1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
