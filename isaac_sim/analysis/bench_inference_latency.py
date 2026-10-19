"""Inference latency of the semantic-task models.

Times one complete policy query (`VLAPolicy._forward`: image preprocessing,
vision encoding of every view, LLM prefill and action prediction) without
starting Isaac Sim. Observations are random images at the real camera
resolutions (on-board 640x480, CCTV 720x720); latency depends on tensor shapes,
not pixel values. The first call is reported separately and excluded from the
mean. Each model runs in its own process with `--all`.

Usage:
    python isaac_sim/analysis/bench_inference_latency.py --policy omnivla_langv2_pre3_best
    python isaac_sim/analysis/bench_inference_latency.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "runs/analysis/inference_latency"

# (table label, policy config name), in the order of the paper's semantic table.
ARMS = [
    ("OmniVLA (ego-only)",     "omnivla_langv2_egoonly_baseline_best"),
    ("No cf. upsampling",      "omnivla_langv2_noupsample_best"),
    ("Full model (ours)",      "omnivla_langv2_pre3_best"),
    ("+ stage-2 recovery",     "omnivla_langv2_A0_recovft_best"),
    ("16 CCTV tokens",         "omnivla_langv2_avgpool16_best"),
    ("Shared CCTV encoder",    "omnivla_langv2_sharedenc_best"),
    ("CCTV views first",       "omnivla_langv2_cctvfirst_best"),
    ("Freeze on-board vision", "omnivla_langv2_frozenego_best"),
    ("+ freeze LLM",           "omnivla_langv2_frozen_best"),
]

# Real capture resolutions, from configs/env/camera/*.yaml. Preprocessing (resize
# to 224) is part of the measured query, so feeding the true source size matters.
EGO_HW = (480, 640)
CCTV_HW = (720, 720)


def build_policy(policy_name):
    """Compose the eval config for one policy and instantiate VLAPolicy."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from isaac_sim.policies.policy import VLAPolicy
    from isaac_sim.task import Task

    cfg_dir = str(REPO / "isaac_sim/configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="eval", overrides=[
            f"policy={policy_name}",
            "task=lang",
            "env/camera_set=with_4cctv_2x3_bigcam",
        ])

    with open_dict(cfg):
        cfg.policy.save_dir = str(OUT_DIR / "_scratch")
    (OUT_DIR / "_scratch").mkdir(parents=True, exist_ok=True)

    policy = VLAPolicy(cfg.policy, cfg.controller, control_hz=float(cfg.control_hz))

    task = Task(**OmegaConf.to_container(cfg.task, resolve=True))
    task.set_node_spacing(cfg.controller.max_linear_vel_mps)
    # A plain pose goal ~12 m ahead: representative mid-rollout geometry. The goal
    # only feeds a few scalars into the prompt/pose token, so it does not affect
    # timing -- it just has to be present and well-formed.
    policy.set_task(task, {"xy": (10.0, 8.0), "heading_rad": 0.0})
    return policy, cfg


def synth_observations(policy, cfg, rng):
    """One observation dict shaped exactly like the sim's, with random pixels."""
    cams = {}
    if policy.cameras is not None:
        for i, view in enumerate(policy.cameras):
            hw = EGO_HW if i == 0 else CCTV_HW
            cams[view.obs_cam] = rng.integers(0, 255, (*hw, 3), dtype=np.uint8)
    else:
        cams[cfg.policy.cur_image_cam_name] = rng.integers(
            0, 255, (*EGO_HW, 3), dtype=np.uint8)
    return {"images": cams, "x": 0.0, "y": 0.0, "heading_rad": 0.0}


def merge_adapters(policy):
    """Fold the LoRA adapters into the base weights and drop the PEFT wrapper.

    The evaluation loads adapters unmerged, which adds two low-rank matmuls per
    adapted layer; a deployment would merge them. Returns the max abs difference
    of the predicted chunk before and after the merge (should be ~0).
    """
    from peft import PeftModel
    if not isinstance(policy.vla, PeftModel):
        return None                      # no adapters (shouldn't happen here)
    obs = synth_observations(policy, policy._bench_cfg, np.random.default_rng(123))
    before, _ = policy._forward(obs)
    policy.vla = policy.vla.merge_and_unload()
    after, _ = policy._forward(obs)
    return float(np.abs(np.asarray(after) - np.asarray(before)).max())


def bench_one(policy_name, reps, merge=False):
    import torch

    t_load0 = time.perf_counter()
    policy, cfg = build_policy(policy_name)
    load_s = time.perf_counter() - t_load0

    policy._bench_cfg = cfg
    merge_maxdiff = merge_adapters(policy) if merge else None

    rng = np.random.default_rng(0)
    lat = []
    for _ in range(reps):
        obs = synth_observations(policy, cfg, rng)   # fresh pixels each rep
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy._forward(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()               # else we time queue submission
        lat.append(time.perf_counter() - t0)

    kept = np.array(lat[1:])                        # drop the warm-up call
    return dict(
        policy=policy_name, reps=reps, merged=bool(merge),
        merge_chunk_maxdiff=merge_maxdiff, load_s=round(load_s, 1),
        n_image_slots=int(policy.inference_config.num_images_in_input),
        first_call_s=round(lat[0], 4),
        mean_s=round(float(kept.mean()), 4),
        std_s=round(float(kept.std(ddof=1)), 4) if len(kept) > 1 else 0.0,
        median_s=round(float(np.median(kept)), 4),
        min_s=round(float(kept.min()), 4), max_s=round(float(kept.max()), 4),
        cv_pct=round(float(100 * kept.std(ddof=1) / kept.mean()), 2) if len(kept) > 1 else 0.0,
        all_s=[round(v, 4) for v in lat],
    )


def run_all(reps, merge=False):
    """Driver: one subprocess per arm, so each starts with a clean allocator."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for label, name in ARMS:
        print(f"[bench] {label} ({name}) ...", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--policy", name, "--reps", str(reps)]
            + (["--merge"] if merge else []),
            capture_output=True, text=True, cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": str(REPO)})
        line = next((l for l in reversed(proc.stdout.splitlines())
                     if l.startswith("__BENCH_JSON__")), None)
        if line is None:
            print(f"  FAILED (exit {proc.returncode})")
            print("  stderr tail:", "\n".join(proc.stderr.splitlines()[-12:]))
            continue
        r = json.loads(line[len("__BENCH_JSON__"):])
        r["label"] = label
        results.append(r)
        md = r.get("merge_chunk_maxdiff")
        print(f"  slots={r['n_image_slots']}  load={r['load_s']}s  "
              f"first={r['first_call_s']}s  mean={r['mean_s']}s "
              f"+/-{r['std_s']}  (CV {r['cv_pct']}%)"
              + (f"  merge_maxdiff={md:.2e}" if md is not None else ""))

    hdr = (f"\n{'model':24s} {'slots':>5s} {'load_s':>7s} {'1st_s':>7s} "
           f"{'mean_s':>7s} {'sd_s':>6s} {'CV%':>6s} {'Hz':>5s}")
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['label']:24s} {r['n_image_slots']:5d} {r['load_s']:7.1f} "
              f"{r['first_call_s']:7.3f} {r['mean_s']:7.3f} {r['std_s']:6.3f} "
              f"{r['cv_pct']:6.2f} {1.0 / r['mean_s']:5.2f}")

    out = OUT_DIR / ("inference_latency_merged.json" if merge
                     else "inference_latency.json")
    out.write_text(json.dumps({"reps": reps, "merged": bool(merge),
                               "note": "first call excluded from mean",
                               "arms": results}, indent=1))
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None, help="single policy config name")
    ap.add_argument("--all", action="store_true", help="every arm, one process each")
    ap.add_argument("--reps", type=int, default=11,
                    help="timed queries per model; first is dropped from the mean")
    ap.add_argument("--merge", action="store_true",
                    help="merge LoRA adapters into the base weights before timing "
                         "(the deployable configuration); verifies the merge is "
                         "numerically inert first")
    a = ap.parse_args()

    if a.all:
        run_all(a.reps, merge=a.merge)
    elif a.policy:
        print("__BENCH_JSON__" + json.dumps(bench_one(a.policy, a.reps, merge=a.merge)))
    else:
        ap.error("pass --policy NAME or --all")


if __name__ == "__main__":
    main()
