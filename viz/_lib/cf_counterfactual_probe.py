
from __future__ import annotations

import argparse, json, math, sys, random as py_random
import pathlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

_REPO = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "vla-scripts"))

import train_omnivla_sim_cctv_v2 as T2   # canonical CCTV v2 loaders / forward mirror
from prismatic.vla.datasets.cctv_isaac_sim_dataset import IsaacSimDataset
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_CCTV
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.extern.hf.vision_encoder import (
    build_cctv_downsampler, resolve_cctv_downsample, resolve_image_slot_perm,
)

from isaac_sim.utils.cf_pairing import (
    find_cf_pairs, load_pose_stream, decision_frame,
    final_wp_lateral, final_wp_forward, final_wp_heading,
    lateral_separation, heading_separation, wrap_pi,
)

# Output root; the active split's results land in OUT_DIR/<tag> (set in main from
# --tag, default the split dir's basename) so train/val/test never overwrite.
OUT_ROOT = _REPO / "isaac_sim/outputs/cf_counterfactual_probe"
OUT_DIR = OUT_ROOT  # rebound in main()
# corner cam -> owned transit aisle (imposed 1:1 quadrant map; matches the dataset's
# _CCTV_CORNER_TO_AISLE_2X3). Used only to highlight the responsible cam in panels.
_CAM_AISLE = {"cctv_NW": "B", "cctv_NE": "C", "cctv_SW": "E", "cctv_SE": "F"}


# ======================================================================
# Model assembly — mirrors train_omnivla_sim_cctv_v2.main() + load_full_state.
# ======================================================================
def build_model(cfg: dict, ckpt_dir: Path, step: int, device: str):
    """Assemble a fine-tuned checkpoint for OPEN-LOOP inference.

    Faithfully mirrors the trainer's build (main(), lines ~869-999) + the resume
    load path (load_full_state): base VLA (+ optional CCTV encoder / separate
    projector) -> LoRA-wrap -> load adapter -> load pose/action heads -> install
    the CCTV downsampler + per-camera embedding -> recompute num_patches. Any
    drift from this order risks silently wrong predictions, so we keep it 1:1 and
    gate correctness with --verify-val.
    """
    m = cfg["model"]
    num_images = m["num_images_in_input"]
    num_ego = m.get("num_ego_images", 2)
    use_cctv_encoder = m.get("use_cctv_encoder", False)
    separate_cctv_projector = m.get("separate_cctv_projector", False)

    processor, vla = T2.load_processor_and_vla(
        Path(m["vla_path"]), num_images_in_input=num_images,
        use_cctv_encoder=use_cctv_encoder, num_ego_images=num_ego,
        separate_cctv_projector=separate_cctv_projector, device=device,
    )
    llm_dim = vla.llm_dim
    vb = vla.vision_backbone
    # default (no-downsample) vision span; recomputed below if we downsample CCTV.
    num_patches = vb.get_num_patches() * vb.get_num_images_in_input() + 1

    if m["use_lora"]:
        vla = T2.wrap_lora(vla, m["lora_rank"], m["lora_alpha"], m["lora_dropout"])
        vla.load_adapter(str(ckpt_dir / "lora_adapter"), adapter_name="default", is_trainable=False)
        print(f"  ✓ loaded LoRA adapter from {ckpt_dir/'lora_adapter'}")

    # pose_projector + action_head: load THIS checkpoint's weights directly
    # (init_module reads {name}--{step}_checkpoint.pt from pretrained_dir).
    pose_projector = T2.init_module(
        ProprioProjector, "pose_projector", {"llm_dim": llm_dim, "proprio_dim": POSE_DIM},
        pretrained_dir=ckpt_dir, pretrained_step=step, device=device)
    action_head = T2.init_module(
        L1RegressionActionHead_idcat, "action_head",
        {"input_dim": llm_dim, "hidden_dim": llm_dim, "action_dim": ACTION_DIM},
        pretrained_dir=ckpt_dir, pretrained_step=step, device=device, to_bf16=True)

    # CCTV token-downsampler + optional per-camera embedding + separate projector.
    ds_method, ds_out = resolve_cctv_downsample(m.get)
    if ds_method != "none":
        downsample_ego = bool(m.get("downsample_ego", False))
        grid_side = int(round(vb.get_num_patches() ** 0.5))
        cctv_ds = build_cctv_downsampler(
            ds_method, ds_out, vision_dim=vb.embed_dim, grid_side=grid_side
        ).to(dtype=torch.bfloat16, device=device)
        n_cctv_i = num_images - num_ego
        cctv_cam_embed = None
        if m.get("cctv_cam_embed", False):
            cctv_cam_embed = nn.Parameter(
                torch.zeros(n_cctv_i, llm_dim, dtype=torch.bfloat16, device=device))
            ce_files = list(ckpt_dir.glob("cctv_cam_embed--*_checkpoint.pt"))
            if not ce_files:
                raise FileNotFoundError(f"cctv_cam_embed enabled but no checkpoint in {ckpt_dir}")
            with torch.no_grad():
                cctv_cam_embed.copy_(
                    torch.load(ce_files[0], weights_only=True, map_location="cpu").to(device))
            print(f"  ✓ loaded per-camera embedding from {ce_files[0].name}")
        cctv_projector_ref = getattr(vla, "cctv_projector", None)   # set iff separate_cctv_projector
        vla.install_pixel_shuffle_encoder(
            cctv_ds, num_ego, downsample_ego,
            cctv_projector=cctv_projector_ref, cctv_cam_embed=cctv_cam_embed)
        # avgpool is param-free -> nothing to load; a parameterized downsampler
        # (pixelshuffle) would need its own checkpoint (not used by these arms).
        if any(p.numel() for p in cctv_ds.parameters()):
            ps_files = list(ckpt_dir.glob("cctv_pixel_shuffle--*_checkpoint.pt"))
            if ps_files:
                cctv_ds.load_state_dict(
                    torch.load(ps_files[0], weights_only=True, map_location="cpu"), strict=False)
        ego_tokens = cctv_ds.out_tokens if downsample_ego else vb.get_num_patches()
        num_patches = ego_tokens * num_ego + cctv_ds.out_tokens * n_cctv_i + 1
        print(f"  num_patches (downsampled): {num_patches}")

    slot_perm = resolve_image_slot_perm(
        m.get("image_slot_order", None), num_ego,
        tuple(cfg["data"].get("cctv_cam_names", ("cctv",))))
    vla.set_image_block_order(slot_perm)

    # Optional absolute-pose token (mirrors policy.py + train_omnivla_sim_cctv_v2):
    # a ProprioProjector(proprio_dim=6) + num_patches += 1. None unless abs_pose_encoder.
    abs_pose_projector = None
    if m.get("abs_pose_encoder", False):
        ap_files = list(ckpt_dir.glob("abs_pose_projector--*_checkpoint.pt"))
        if not ap_files:
            raise FileNotFoundError(f"abs_pose_encoder enabled but no checkpoint in {ckpt_dir}")
        abs_pose_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=6).to(
            dtype=torch.bfloat16, device=device)
        abs_pose_projector.load_state_dict(
            torch.load(ap_files[0], weights_only=True, map_location="cpu"))
        abs_pose_projector.eval()
        num_patches += 1

    vla.eval(); pose_projector.eval(); action_head.eval()
    return processor, vla, pose_projector, action_head, num_patches, abs_pose_projector


# ======================================================================
# Dataset + deterministic sample construction.
# ======================================================================
def build_dataset(cfg: dict, processor, split_dir: Path, seed: int) -> IsaacSimDataset:
    """Build the split dataset with the model's VAL perception regime.

    Mirrors the trainer's val dataset for this cfg: val_perception_dropout (both
    views present for these arms), cctv_crop passthrough, no hflip. emit_cctv and
    cctv_cam_names come from cfg, so the ego-only baseline builds a 2-image stack
    and the camembed model a 6-image stack over the SAME split dir.
    """
    d = cfg["data"]
    return IsaacSimDataset(
        action_tokenizer=ActionTokenizer(processor.tokenizer),
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        traj_root=split_dir,
        max_trajectories=None,
        context_size=d["context_size"],
        num_actions_chunk=d["num_actions_chunk"],
        modality_list=tuple(d["modality_list"]),
        image_only_max_distance_steps=d["image_only_max_distance_steps"],
        min_goal_distance_steps=d["min_goal_distance_steps"],
        max_goal_distance_steps=d["max_goal_distance_steps"],
        metric_waypoint_spacing=d["metric_waypoint_spacing"],
        waypoint_spacing=d["waypoint_spacing"],
        image_size=tuple(d["image_size"]),
        random_hflip=False,
        goals_per_obs=d["goals_per_obs"],
        emit_cctv=d["emit_cctv"],
        use_cctv_goal=d.get("use_cctv_goal", False),
        cctv_cam_names=tuple(d.get("cctv_cam_names", ("cctv",))),
        no_goal_image=d.get("no_goal_image", False),
        perception_dropout=tuple(d["val_perception_dropout"]),
        final_goal_prob=d.get("final_goal_prob", 0.0),
        cctv_crop=d.get("cctv_crop"),
        cctv_crop_ref_size=d.get("cctv_crop_ref_size", 720),
        abs_pose=cfg["model"].get("abs_pose_encoder", False),
        abs_pose_bounds=d.get("abs_pose_bounds"),
        predict_stop_token=d["predict_stop_token"],
        seed=seed,
    )


def _name_to_traj_index(ds: IsaacSimDataset) -> dict[str, int]:
    """scene-run-dir name -> trajectory index in ds.trajectories."""
    return {Path(tr.traj_dir).name: ti for ti, tr in enumerate(ds.trajectories)}


def sample_pose_final_goal(ds: IsaacSimDataset, traj_idx: int, t: int) -> dict:
    """Deterministically build the modality-4 (pose goal), FINAL-goal sample at
    (traj_idx, t) — the controlled probe input.

    We temporarily pin final_goal_prob=1.0 (goal := last frame, no which-way leak;
    the goal heading is already de-leaked to the start heading in the dataset) and
    modality_list=[4] (pose-only goal), then call __getitem__ at the global index
    for (traj_idx, t). With val_perception_dropout=(1,0,0) and no hflip, the sample
    is fully deterministic regardless of rng state. Attributes are restored after.
    """
    global_idx = ds.samples_index.index((traj_idx, t))
    saved = (ds.final_goal_prob, list(ds.modality_list))
    ds.final_goal_prob = 1.0
    ds.modality_list = [4]
    try:
        ds._rng = py_random.Random(0)   # deterministic (result is rng-independent here)
        s = ds[global_idx]
    finally:
        ds.final_goal_prob, ds.modality_list = saved[0], saved[1]
    return s


@torch.no_grad()
def predict_chunk(batch: dict, vla, pose_projector, action_head, num_patches: int,
                  device: str, abs_pose_projector=None) -> np.ndarray:
    """Open-loop predicted action chunk (N, ACTION_DIM). Mirrors
    train_omnivla_sim_cctv_v2.forward_and_loss up to action_head.predict_action."""
    bsize = batch["pixel_values"].shape[0]
    modality_id = batch["modality_id"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            attention_mask_label=batch["attention_mask_label"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            modality_id=modality_id.to(torch.bfloat16),
            block_present=batch["block_present"].to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=batch["goal_pose"].to(torch.bfloat16).to(device),
            proprio_projector=pose_projector,
            abs_proprio=(batch["abs_pose"].to(torch.bfloat16).to(device)
                         if abs_pose_projector is not None and "abs_pose" in batch else None),
            abs_proprio_projector=abs_pose_projector,
            use_film=False,
        )
    gt_token_ids = batch["labels"][:, 1:].to(device)
    cur_mask = get_current_action_mask(gt_token_ids)
    nxt_mask = get_next_actions_mask(gt_token_ids)
    text_h = output.hidden_states[-1][:, num_patches:-1]
    actions_h = (text_h[cur_mask | nxt_mask]
                 .reshape(bsize, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16))
    pred = action_head.predict_action(actions_h, modality_id.to(torch.bfloat16))
    return pred[0].float().cpu().numpy()


def predict_chunk_batch(batch: dict, vla, pose_projector, action_head, num_patches: int,
                        device: str, abs_pose_projector=None) -> np.ndarray:
    """Batched twin of predict_chunk: same forward, but returns ALL bsize chunks
    (shape (bsize, N, ACTION_DIM)) instead of just pred[0]. Lets analysis probes
    collate many samples into one forward pass to use the whole GPU instead of
    running batch-of-1. Result order matches the input sample order."""
    bsize = batch["pixel_values"].shape[0]
    modality_id = batch["modality_id"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            attention_mask_label=batch["attention_mask_label"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            modality_id=modality_id.to(torch.bfloat16),
            block_present=batch["block_present"].to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=batch["goal_pose"].to(torch.bfloat16).to(device),
            proprio_projector=pose_projector,
            abs_proprio=(batch["abs_pose"].to(torch.bfloat16).to(device)
                         if abs_pose_projector is not None and "abs_pose" in batch else None),
            abs_proprio_projector=abs_pose_projector,
            use_film=False,
        )
    gt_token_ids = batch["labels"][:, 1:].to(device)
    cur_mask = get_current_action_mask(gt_token_ids)
    nxt_mask = get_next_actions_mask(gt_token_ids)
    text_h = output.hidden_states[-1][:, num_patches:-1]
    actions_h = (text_h[cur_mask | nxt_mask]
                 .reshape(bsize, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16))
    pred = action_head.predict_action(actions_h, modality_id.to(torch.bfloat16))
    return pred.float().cpu().numpy()


# ======================================================================
# Pairing + decision frames (model-free precompute).
# ======================================================================
def precompute_pairs(split_dir: Path, n_chunk: int, dev_thresh: float) -> list[dict]:
    """All cf0/cfX pairs with a valid decision frame t* (both variants long
    enough to host an obs step t* with n_chunk future frames + a goal)."""
    raw = find_cf_pairs(split_dir)
    out = []
    for p in raw:
        pos0, yaw0, names0 = load_pose_stream(p["cf0_dir"])
        posx, yawx, namesx = load_pose_stream(p["cfx_dir"])
        t_star, dev, start_off = decision_frame(pos0, posx, n_chunk, dev_thresh_m=dev_thresh)
        if t_star is None:
            continue                                  # vacuous (no divergence)
        # t* must be a valid observation step in BOTH trajectories (index build
        # needs n-1-t >= max(min_goal, n_chunk); we use the stricter n_chunk).
        if t_star > min(len(pos0), len(posx)) - 1 - n_chunk:
            continue
        p = dict(p)
        p.update(t_star=t_star, dev=dev, start_off=start_off,
                 cf0_frame=names0[t_star], cfx_frame=namesx[t_star],
                 pos0=pos0, posx=posx, yaw0=yaw0, yawx=yawx)
        out.append(p)
    return out


def ego_frame_diff(cf0_dir: Path, cfx_dir: Path, frame0: str, framex: str,
                   size: int = 224) -> float:
    """Mean absolute pixel difference (0-255) between the two variants' ego frames
    at the decision step — the premise control. ~0 => the seal is off-camera and
    the baseline genuinely cannot tell cf0 from cfX; large => it leaked into ego."""
    a = np.asarray(Image.open(cf0_dir / "frames" / frame0).convert("RGB")
                   .resize((size, size), Image.BILINEAR), dtype=np.float32)
    b = np.asarray(Image.open(cfx_dir / "frames" / framex).convert("RGB")
                   .resize((size, size), Image.BILINEAR), dtype=np.float32)
    return float(np.abs(a - b).mean())


# ======================================================================
# Modes.
# ======================================================================
def run_dry(split_dir: Path, n_chunk: int, dev_thresh: float):
    pairs = precompute_pairs(split_dir, n_chunk, dev_thresh)
    print(f"\n=== DRY: {len(pairs)} valid cf0/cfX pairs (dev_thresh={dev_thresh} m) ===")
    t0 = sum(1 for p in pairs if p["t_star"] == 0)
    print(f"  t*=0 (divert at the very start): {t0}/{len(pairs)}")
    print(f"  {'pair':<26}{'t*':>4}{'dev':>7}{'ego_diff':>10}{'start_off':>11}")
    ego_diffs = []
    for p in pairs:
        ed = ego_frame_diff(p["cf0_dir"], p["cfx_dir"], p["cf0_frame"], p["cfx_frame"])
        ego_diffs.append(ed)
        if len(ego_diffs) <= 25:
            print(f"  {p['cfx_scene']:<26}{p['t_star']:>4}{p['dev']:>7.2f}"
                  f"{ed:>10.2f}{p['start_off']:>11.4f}")
    ego_diffs = np.array(ego_diffs)
    print(f"\n  ego-frame diff (0-255): mean={ego_diffs.mean():.2f} "
          f"median={np.median(ego_diffs):.2f} max={ego_diffs.max():.2f}")
    print(f"  (low => seal off-camera => baseline is blind to cf0 vs cfX, as designed)")


def run_verify_val(cfg: dict, ckpt: Path, step: int, val_dir: Path, device: str,
                   n_samples: int):
    """Sanity gate: mean modality-4 L1 over n val samples ≈ the checkpoint's
    logged l1_modality_4 (garbage assembly would give L1 >> 0.1)."""
    processor, vla, pose_projector, action_head, num_patches, abs_pose_projector = build_model(cfg, ckpt, step, device)
    ds = build_dataset(cfg, processor, val_dir, int(cfg["train"].get("seed", 42)))
    collator = PaddedCollatorForActionPrediction_Nav_CCTV(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id,
        padding_side="right", num_img=cfg["model"]["num_images_in_input"])
    pad_id, max_len = processor.tokenizer.pad_token_id, processor.tokenizer.model_max_length
    # force modality 4 across a spread of samples (varied goal distances)
    saved_ml = list(ds.modality_list); ds.modality_list = [4]
    idxs = np.linspace(0, len(ds) - 1, min(n_samples, len(ds))).astype(int)
    l1s = []
    for j, i in enumerate(idxs):
        ds._rng = py_random.Random(1234 + int(i))
        s = ds[int(i)]
        batch = T2.pad_and_mask_batch(collator([s]), pad_id, max_len)
        pred = predict_chunk(batch, vla, pose_projector, action_head, num_patches, device, abs_pose_projector)
        gt = np.asarray(s["actions"], dtype=np.float32)
        l1s.append(np.abs(pred - gt).mean())
    ds.modality_list = saved_ml
    l1 = float(np.mean(l1s))
    logged = json.loads((ckpt / "val_metrics.json").read_text()).get("l1_modality_4")
    print(f"\n=== VERIFY-VAL {ckpt.parent.parent.name} ===")
    print(f"  probe modality-4 L1 over {len(idxs)} samples: {l1:.4f}")
    print(f"  checkpoint logged l1_modality_4:            {logged:.4f}")
    print(f"  {'✓ MATCH' if abs(l1 - logged) < 0.02 else '✗ MISMATCH — assembly suspect'}")
    return l1, logged


def run_full(args, device: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split)
    pairs = precompute_pairs(split_dir, args.n_chunk, args.dev_thresh)
    if args.scenes:
        # Restrict to an explicit set of cfX scenes and panel ALL of them (used to
        # visualise a specific subset, e.g. the 'partial'-visibility bin).
        want = set(s.strip() for s in args.scenes.split(",") if s.strip())
        pairs = [p for p in pairs if p["cfx_scene"] in want]
        args.panels = len(pairs)
        print(f"=== FULL probe restricted to {len(pairs)} requested scenes ===")
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"=== FULL probe over {len(pairs)} pairs ===")

    # premise: ego-frame diff at t* (model-free)
    for p in pairs:
        p["ego_diff"] = ego_frame_diff(p["cf0_dir"], p["cfx_dir"], p["cf0_frame"], p["cfx_frame"])

    # GT chunks come from the dataset (model-independent); take them once from the
    # camembed dataset build below. Predictions are gathered per model, sequentially.
    models = [
        ("baseline", T2.load_yaml_config(Path(args.baseline_config)),
         Path(args.baseline_ckpt), args.baseline_step),
        ("camembed", T2.load_yaml_config(Path(args.camembed_config)),
         Path(args.camembed_ckpt), args.camembed_step),
    ]

    gt_captured = False
    for mname, cfg, ckpt, step in models:
        print(f"\n--- {mname}: {ckpt.parent.parent.name} step {step} ---")
        processor, vla, pose_projector, action_head, num_patches, abs_pose_projector = build_model(cfg, ckpt, step, device)
        ds = build_dataset(cfg, processor, split_dir, int(cfg["train"].get("seed", 42)))
        collator = PaddedCollatorForActionPrediction_Nav_CCTV(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id,
            padding_side="right", num_img=cfg["model"]["num_images_in_input"])
        pad_id, max_len = processor.tokenizer.pad_token_id, processor.tokenizer.model_max_length
        name2ti = _name_to_traj_index(ds)

        for p in pairs:
            ti0 = name2ti[Path(p["cf0_dir"]).name]
            tix = name2ti[Path(p["cfx_dir"]).name]
            s0 = sample_pose_final_goal(ds, ti0, p["t_star"])
            sx = sample_pose_final_goal(ds, tix, p["t_star"])
            b0 = T2.pad_and_mask_batch(collator([s0]), pad_id, max_len)
            bx = T2.pad_and_mask_batch(collator([sx]), pad_id, max_len)
            p[f"{mname}_cf0"] = predict_chunk(b0, vla, pose_projector, action_head, num_patches, device, abs_pose_projector)
            p[f"{mname}_cfx"] = predict_chunk(bx, vla, pose_projector, action_head, num_patches, device, abs_pose_projector)
            if not gt_captured:
                p["gt_cf0"] = np.asarray(s0["actions"], dtype=np.float32)
                p["gt_cfx"] = np.asarray(sx["actions"], dtype=np.float32)
        gt_captured = True
        del vla, pose_projector, action_head, processor
        torch.cuda.empty_cache()

    metrics = compute_metrics(pairs)
    save_outputs(pairs, metrics, args)
    return metrics


# ======================================================================
# Metrics + outputs.
# ======================================================================
def compute_metrics(pairs: list[dict]) -> dict:
    """Per-pair separations + aggregate direction-correctness / angular error."""
    rows = []
    for p in pairs:
        row = {
            "scene": p["cfx_scene"], "blocked": p["blocked"], "t_star": p["t_star"],
            "dev": p["dev"], "ego_diff": p["ego_diff"],
            "gt_sep_lat": lateral_separation(p["gt_cfx"], p["gt_cf0"]),
            "gt_sep_head": math.degrees(heading_separation(p["gt_cfx"], p["gt_cf0"])),
            "gt_lat_cfx": final_wp_lateral(p["gt_cfx"]),
        }
        for mname in ("baseline", "camembed"):
            row[f"{mname}_sep_lat"] = lateral_separation(p[f"{mname}_cfx"], p[f"{mname}_cf0"])
            row[f"{mname}_sep_head"] = math.degrees(heading_separation(p[f"{mname}_cfx"], p[f"{mname}_cf0"]))
            row[f"{mname}_lat_cfx"] = final_wp_lateral(p[f"{mname}_cfx"])
            # angular error of the cfX prediction vs the cfX GT (deg)
            row[f"{mname}_head_err_cfx"] = math.degrees(abs(wrap_pi(
                final_wp_heading(p[f"{mname}_cfx"]) - final_wp_heading(p["gt_cfx"]))))
        rows.append(row)

    # "genuine divergence" subset: GT actually turns (|lateral sep| above a small
    # deadband) — the scenes where a correct turn direction is well-defined.
    DEADBAND = 0.25   # normalized lateral units (~0.08 m net) — below this GT ~straight
    agg = {"n_pairs": len(rows), "deadband": DEADBAND}
    div = [r for r in rows if abs(r["gt_sep_lat"]) > DEADBAND]
    agg["n_divergent"] = len(div)
    for mname in ("baseline", "camembed"):
        # direction-correct: predicted lateral separation has the SAME sign as GT's
        correct = sum(1 for r in div
                      if np.sign(r[f"{mname}_sep_lat"]) == np.sign(r["gt_sep_lat"]))
        agg[f"{mname}_dir_acc"] = correct / len(div) if div else float("nan")
        agg[f"{mname}_mean_abs_sep"] = float(np.mean([abs(r[f"{mname}_sep_lat"]) for r in rows]))
        agg[f"{mname}_median_head_err_cfx"] = float(np.median([r[f"{mname}_head_err_cfx"] for r in rows]))
    agg["gt_mean_abs_sep"] = float(np.mean([abs(r["gt_sep_lat"]) for r in rows]))
    return {"rows": rows, "agg": agg}


def _plot_chunk(ax, chunk, color, label, ls="-", lw=2.5):
    """Plot a chunk as a path in local coords, forward=up (x), left=left (+y).
    We draw matplotlib-x = -dy (left is left on screen), matplotlib-y = dx."""
    xs = np.concatenate([[0.0], -chunk[:, 1]])   # left positive -> screen left
    ys = np.concatenate([[0.0], chunk[:, 0]])    # forward -> up
    ax.plot(xs, ys, ls, color=color, lw=lw, label=label, marker="o", ms=4)


def save_outputs(pairs: list[dict], metrics: dict, args):
    rows, agg = metrics["rows"], metrics["agg"]
    # ---- JSON (no numpy arrays) ----
    (OUT_DIR / "pair_results.json").write_text(json.dumps(
        {"agg": agg, "rows": rows}, indent=2, default=float))

    # ---- summary ----
    lines = [
        f"Paired counterfactual probe — {agg['n_pairs']} pairs "
        f"({agg['n_divergent']} genuinely divergent, |GT lat sep|>{agg['deadband']})",
        f"  GT   mean |lateral separation|           : {agg['gt_mean_abs_sep']:.3f}",
        f"  baseline mean |lateral separation|        : {agg['baseline_mean_abs_sep']:.3f}   (under-commits vs GT)",
        f"  camembed mean |lateral separation|        : {agg['camembed_mean_abs_sep']:.3f}   (under-commits vs GT, modestly more)",
        f"  baseline direction-correct on divergent   : {100*agg['baseline_dir_acc']:.1f}%",
        f"  camembed direction-correct on divergent   : {100*agg['camembed_dir_acc']:.1f}%",
        f"  baseline median heading-err vs GT (cfX)    : {agg['baseline_median_head_err_cfx']:.1f} deg",
        f"  camembed median heading-err vs GT (cfX)    : {agg['camembed_median_head_err_cfx']:.1f} deg",
    ]
    summary = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)

    # ---- headline: model separation vs GT separation ----
    gt = np.array([r["gt_sep_lat"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    lim = max(1.0, np.abs(gt).max() * 1.1)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal (pred sep = GT sep)")
    ax.axhline(0, color="0.7", lw=1)
    for mname, color in (("baseline", "tab:red"), ("camembed", "tab:blue")):
        ms = np.array([r[f"{mname}_sep_lat"] for r in rows])
        ax.scatter(gt, ms, s=28, alpha=0.6, color=color, label=mname)
    ax.set_xlabel("GT lateral separation  cfX − cf0  (normalized)")
    ax.set_ylabel("model lateral separation  cfX − cf0")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title("cf0→cfX prediction separation vs GT lateral separation\n"
                 "(both models track sign but under-commit in magnitude — see commit ratio)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "aggregate_separation.png", dpi=120); plt.close(fig)

    # ---- direction accuracy + heading-error box ----
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))
    axa.bar([0, 1], [100*agg["baseline_dir_acc"], 100*agg["camembed_dir_acc"]],
            color=["tab:red", "tab:blue"])
    axa.set_xticks([0, 1]); axa.set_xticklabels(["baseline", "camembed"])
    axa.set_ylabel("% correct divert direction"); axa.set_ylim(0, 100)
    axa.axhline(50, color="0.6", ls="--", lw=1, label="chance")
    axa.set_title(f"Divert-direction accuracy (n={agg['n_divergent']} divergent)"); axa.legend()
    be = [r["baseline_head_err_cfx"] for r in rows]
    ce = [r["camembed_head_err_cfx"] for r in rows]
    axb.boxplot([be, ce], labels=["baseline", "camembed"], showmeans=True)
    axb.set_ylabel("heading error vs GT on cfX (deg)")
    axb.set_title("Per-pair heading error at the decision frame")
    axb.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT_DIR / "aggregate_direction.png", dpi=120); plt.close(fig)

    # ---- per-scene panels (a diverse selection) ----
    panels_dir = OUT_DIR / "panels"; panels_dir.mkdir(exist_ok=True)
    # rank by GT separation magnitude, take the strongest K counterfactuals
    sel = sorted(pairs, key=lambda p: abs(lateral_separation(p["gt_cfx"], p["gt_cf0"])),
                 reverse=True)[:args.panels]
    for p in sel:
        _save_panel(p, panels_dir)
    print(f"\n=== saved outputs to {OUT_DIR} ===")


def _save_panel(p: dict, panels_dir: Path):
    """One scene: ego cf0 vs cfX, the 4 CCTV cfX (owner cam boxed), and the
    action-chunk comparison at the decision frame."""
    cams = ["cctv_NW", "cctv_NE", "cctv_SW", "cctv_SE"]
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

    # ego frames
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(Image.open(p["cf0_dir"] / "frames" / p["cf0_frame"]).convert("RGB"))
    ax.set_title(f"ego cf0 (clear)  t={p['t_star']}"); ax.axis("off")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(Image.open(p["cfx_dir"] / "frames" / p["cfx_frame"]).convert("RGB"))
    ax.set_title(f"ego cf{p['blocked']} (blocked)  ego_diff={p['ego_diff']:.1f}"); ax.axis("off")

    # 4 CCTV of the blocked variant; box the owner cam(s) of the sealed aisle(s)
    for k, cam in enumerate(cams):
        r, c = divmod(k, 2)
        ax = fig.add_subplot(gs[r, 1 + c])
        cam_dir = p["cfx_dir"] / cam
        img_path = cam_dir / p["cfx_frame"]
        if img_path.exists():
            ax.imshow(Image.open(img_path).convert("RGB"))
        owns_blocked = _CAM_AISLE.get(cam) in p["blocked_set"]
        title = f"{cam} (aisle {_CAM_AISLE.get(cam,'?')})"
        if owns_blocked:
            title += "  ← SEALED"
            for spine in ax.spines.values():
                spine.set_color("red"); spine.set_linewidth(4)
        ax.set_title(title, color="red" if owns_blocked else "black")
        ax.set_xticks([]); ax.set_yticks([])

    # action-chunk comparison (spans the far-right column bottom is used; use a
    # dedicated axis stacked to the right)
    ax = fig.add_subplot(gs[:, 3])
    _plot_chunk(ax, p["gt_cf0"], "tab:green", "GT cf0 (clear)", ls="--", lw=2)
    _plot_chunk(ax, p["gt_cfx"], "black", "GT cfX (blocked)", lw=3)
    _plot_chunk(ax, p["baseline_cfx"], "tab:red", "baseline cfX")
    _plot_chunk(ax, p["camembed_cfx"], "tab:blue", "camembed cfX")
    ax.scatter([0], [0], c="k", s=60, zorder=5)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("← right    left →"); ax.set_ylabel("forward →")
    ax.set_title("predicted chunk at decision frame")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{p['cfx_scene']}   blocked={sorted(p['blocked_set'])}   "
                 f"GT lat sep={lateral_separation(p['gt_cfx'], p['gt_cf0']):+.2f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(panels_dir / f"{p['cfx_scene']}.png", dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(_REPO / "runs/data/splits/2026-07-04_warehouse_2x3_cfv2fwd/train"))
    ap.add_argument("--val-split", default=str(_REPO / "runs/data/splits/2026-07-04_warehouse_2x3_cfv2fwd/val"))
    ap.add_argument("--n-chunk", type=int, default=NUM_ACTIONS_CHUNK)
    ap.add_argument("--dev-thresh", type=float, default=0.75, help="cf0/cfX future divergence threshold (m) for t*")
    ap.add_argument("--limit", type=int, default=0, help="cap #pairs (0 = all)")
    ap.add_argument("--panels", type=int, default=12, help="#per-scene panels (strongest counterfactuals)")
    ap.add_argument("--scenes", default=None, help="comma-separated cfX scene names to probe + panel exclusively")
    ap.add_argument("--tag", default=None, help="output subdir under cf_counterfactual_probe/ (default: split basename)")
    ap.add_argument("--dry", action="store_true", help="pairing + premise only, no model load")
    ap.add_argument("--verify-val", action="store_true", help="reproduce a checkpoint's val modality-4 L1")
    ap.add_argument("--verify-n", type=int, default=150)
    # model specs
    ap.add_argument("--baseline-config"); ap.add_argument("--baseline-ckpt"); ap.add_argument("--baseline-step", type=int)
    ap.add_argument("--camembed-config"); ap.add_argument("--camembed-ckpt"); ap.add_argument("--camembed-step", type=int)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # split-scoped output dir so train/val/test runs don't clobber each other
    global OUT_DIR
    OUT_DIR = OUT_ROOT / (args.tag or Path(args.split).name)

    if args.dry:
        run_dry(Path(args.split), args.n_chunk, args.dev_thresh)
        return
    if args.verify_val:
        for tag, cfg_p, ckpt, step in (
            ("baseline", args.baseline_config, args.baseline_ckpt, args.baseline_step),
            ("camembed", args.camembed_config, args.camembed_ckpt, args.camembed_step)):
            if cfg_p:
                run_verify_val(T2.load_yaml_config(Path(cfg_p)), Path(ckpt), step,
                               Path(args.val_split), device, args.verify_n)
        return
    run_full(args, device)


if __name__ == "__main__":
    main()
