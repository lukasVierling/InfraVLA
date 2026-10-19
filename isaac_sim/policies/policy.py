import math
import os
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

# File lives at isaac_sim/policies/policy.py; two parent hops -> repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, f"{_REPO_ROOT}/inference", f"{_REPO_ROOT}/isaac_sim"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_omnivla as ro
from run_omnivla import Inference, InferenceConfig, define_model
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.datasets.cctv_isaac_sim_dataset import (
    _CCTV_CORNER_TO_AISLE_2X3, _blockade_shade, _parse_blocked_aisles,
)
from isaac_sim.planning.a_star.goal_graph import (
    GoalGraph, SubgoalSelector,
)
from omegaconf import OmegaConf


def load_model(policy_cfg, inference_config):
    """ loads the model based on whether we have LoRA adapters in the checkpoint or not"""
    finetune = OmegaConf.select(policy_cfg, "finetune")
    # define_model only sees inference_config, but the CCTV-downsample knobs live in the
    # finetune block (with the other architecture knobs). Resolve them and bridge a single
    # "use v2 path" flag across so define_model loads the v2 model class (needed for BOTH
    # pixelshuffle and avgpool). Default none => v1 path unchanged.
    if finetune is not None:
        from prismatic.extern.hf.vision_encoder import resolve_cctv_downsample
        ds_method, _ = resolve_cctv_downsample(lambda k, d=None: OmegaConf.select(finetune, k, default=d))
        # The v2 model class is required for the CCTV downsamplers AND for image_slot_order
        # (the token reorder + set_image_block_order live ONLY in v2). A full-token ego-last
        # run has ds_method="none" but STILL needs v2 — so OR the two. v2 full-token forward
        # is bit-identical to v1 (verified seam), so this never changes non-reordered behaviour.
        has_slot_order = OmegaConf.select(finetune, "image_slot_order", default=None) is not None
        inference_config.use_cctv_downsample = (ds_method != "none") or has_slot_order
    model = define_model(inference_config)
    if finetune is not None:
        model = _apply_finetune_overlay(model, finetune)
    else:
        model = (*model, None)   # no abs-pose projector on the non-finetune path
    return model


def _assert_adapter_matches_cctv_flag(adapter_dir, use_cctv_encoder):
    """Fail loudly when the adapter and the use_cctv_encoder flag disagree.

    PEFT does NOT raise on either mismatch (verified 2026-06-09): it silently
    skips adapter weights with no matching module (dual-stream adapter, flag
    off) and silently leaves un-targeted modules adapter-less (old adapter,
    flag on). Both produce a model that runs but is not the one trained.
    """
    import json
    with open(pathlib.Path(adapter_dir) / "adapter_config.json") as f:
        targets = json.load(f).get("target_modules", [])
    if isinstance(targets, str):  # PEFT may store a single regex string
        adapter_has_cctv = "cctv" in targets
    else:
        adapter_has_cctv = any("cctv" in t for t in targets)
    if adapter_has_cctv != use_cctv_encoder:
        raise ValueError(
            f"finetune.use_cctv_encoder={use_cctv_encoder} but adapter at "
            f"{adapter_dir} {'targets' if adapter_has_cctv else 'does not target'} "
            f"cctv modules — flag must match how the checkpoint was trained."
        )


def _apply_finetune_overlay(model, finetune):
    """Appleis finetune overlay on the model."""
    vla, action_head, pose_projector, device_id, num_patches, action_tokenizer, processor = model

    from peft import PeftModel
    ckpt_dir = pathlib.Path(finetune.checkpoint_dir)
    step = int(finetune.step)
    adapter_dir = ckpt_dir / "lora_adapter"
    action_head_path = ckpt_dir / f"action_head--{step}_checkpoint.pt"
    pose_projector_path = ckpt_dir / f"pose_projector--{step}_checkpoint.pt"

    print(f"[load_model] fine-tune overlay from {ckpt_dir} (step {step})")
    use_cctv_encoder = bool(OmegaConf.select(finetune, "use_cctv_encoder", default=False))
    _assert_adapter_matches_cctv_flag(adapter_dir, use_cctv_encoder)
    if use_cctv_encoder:
        # must exist before adapter load: adapter_config.json targets
        # vision_backbone.cctv_featurizer.* module names
        num_ego_images = int(OmegaConf.select(finetune, "num_ego_images", default=2))
        print(f"[load_model]   adding parallel CCTV encoder (num_ego_images={num_ego_images})")
        vla.vision_backbone.add_cctv_encoder(num_ego_images=num_ego_images)
        # Distinct CCTV projector: must exist BEFORE the adapter load so the adapter's
        # cctv_projector.* LoRA keys resolve (mirrors the cctv encoder above).
        if bool(OmegaConf.select(finetune, "separate_cctv_projector", default=False)):
            import copy as _copy
            vla.cctv_projector = _copy.deepcopy(vla.projector)
            print("[load_model]   added separate CCTV projector (pre-adapter)")
    print(f"[load_model]   applying LoRA adapter: {adapter_dir}")
    vla = PeftModel.from_pretrained(vla, str(adapter_dir))
    vla = vla.to(device_id)

    print(f"[load_model]   loading fine-tuned action_head: {action_head_path}")
    action_head.load_state_dict(
        torch.load(action_head_path, weights_only=True, map_location="cpu"),
        strict=False,
    )
    print(f"[load_model]   loading fine-tuned pose_projector: {pose_projector_path}")
    pose_projector.load_state_dict(
        torch.load(pose_projector_path, weights_only=True, map_location="cpu"),
        strict=False,
    )

    # --- v2 CCTV token-downsample (pixelshuffle=learned, avgpool=param-free) ---
    # Install the standalone downsampler AFTER the PEFT wrap (NOT LoRA-targeted; for
    # pixelshuffle it is saved/loaded as a peer of the heads), mirroring
    # train_omnivla_sim_cctv_v2.py. Then recompute the vision span so the action
    # hidden-state slice (run_omnivla.py uses NUM_PATCHES) is correct for the reduced
    # CCTV token counts.
    from prismatic.extern.hf.vision_encoder import build_cctv_downsampler, resolve_cctv_downsample
    ds_method, ds_out = resolve_cctv_downsample(lambda k, d=None: OmegaConf.select(finetune, k, default=d))
    downsample_ego = bool(OmegaConf.select(finetune, "downsample_ego", default=False))
    if ds_method != "none":
        n_ego = int(OmegaConf.select(finetune, "num_ego_images", default=2))
        base = vla.base_model.model                     # unwrap PeftModel -> v2 OpenVLA model (has the seam)
        grid_side = int(round(base.vision_backbone.get_num_patches() ** 0.5))
        ds = build_cctv_downsampler(
            ds_method, ds_out,
            vision_dim=base.vision_backbone.embed_dim,  # fused dim (2176), NOT llm_dim — runs pre-projector
            grid_side=grid_side,
        ).to(dtype=torch.bfloat16, device=device_id)
        # v2 CCTV extras: distinct projector (created pre-adapter, LoRA now loaded) +
        # per-camera identity embedding (peer .pt). Both None => shared-projector path.
        cctv_projector_ref = getattr(base, "cctv_projector", None)
        cctv_cam_embed = None
        if bool(OmegaConf.select(finetune, "cctv_cam_embed", default=False)):
            ce_path = ckpt_dir / f"cctv_cam_embed--{step}_checkpoint.pt"
            if not ce_path.exists():
                raise FileNotFoundError(f"cctv_cam_embed enabled but no checkpoint at {ce_path}")
            cctv_cam_embed = torch.nn.Parameter(
                torch.load(ce_path, weights_only=True, map_location="cpu").to(device_id))
        base.install_pixel_shuffle_encoder(
            ds, n_ego, downsample_ego,
            cctv_projector=cctv_projector_ref, cctv_cam_embed=cctv_cam_embed)   # encoder is downsampler-agnostic
        # Parameterized downsamplers (pixelshuffle) load trained weights — strict, so a
        # stages/out_tokens mismatch fails loudly. Param-free (avgpool) reconstruct from
        # config — nothing to load.
        if any(p.numel() for p in ds.parameters()):
            ds_path = ckpt_dir / f"cctv_pixel_shuffle--{step}_checkpoint.pt"
            if not ds_path.exists():
                raise FileNotFoundError(
                    f"cctv downsample '{ds_method}' needs weights but no checkpoint at {ds_path}")
            ds.load_state_dict(torch.load(ds_path, weights_only=True, map_location="cpu"))
        n_cctv = base.vision_backbone.get_num_images_in_input() - n_ego
        ego_tok = ds.out_tokens if downsample_ego else base.vision_backbone.get_num_patches()
        num_patches = ego_tok * n_ego + ds.out_tokens * n_cctv + 1
        print(f"[load_model]   cctv downsample '{ds_method}' (downsample_ego={downsample_ego}): "
              f"256->{ds.out_tokens} tok/view; NUM_PATCHES recomputed = {num_patches}")

    # --- Optional image-slot order (e.g. ego-last): permute the LLM token blocks so
    # the DEPLOYED token order matches training EXACTLY (a mismatch silently scrambles
    # the model's view). SAME resolver as train_omnivla_sim_cctv_v2.py — single source
    # of truth. Absent => identity => legacy ego-first order, every existing config
    # unaffected. predict_action() reorders via _process_vision_features once this is set.
    slot_order = OmegaConf.select(finetune, "image_slot_order", default=None)
    if slot_order is not None:
        from prismatic.extern.hf.vision_encoder import resolve_image_slot_perm
        slot_order = list(slot_order)
        cctv_names = list(OmegaConf.select(finetune, "cctv_cam_names", default=("cctv",)))
        n_ego_so = int(OmegaConf.select(finetune, "num_ego_images", default=2))
        slot_perm = resolve_image_slot_perm(slot_order, n_ego_so, cctv_names)
        base = vla.base_model.model              # unwrap PeftModel -> v2 model with the seam
        base.set_image_block_order(slot_perm)
        print(f"[load_model]   image_slot_order {slot_order} -> block perm {slot_perm}")

    # --- Optional absolute-pose token (eval side of train abs_pose_encoder) ---
    # A SECOND ProprioProjector (proprio_dim=6) mapping the world-frame vector
    # [x, y, cos(yaw), sin(yaw), goal_x, goal_y] to ONE extra token appended after
    # the relative-pose token. Loaded here; the per-step vector is assembled in
    # VLAPolicy._forward. Adds +1 to NUM_PATCHES so the action hidden-state slice
    # stays aligned (mirrors train_omnivla_sim_cctv_v2 num_patches += 1). None => off.
    abs_pose_projector = None
    if bool(OmegaConf.select(finetune, "abs_pose_encoder", default=False)):
        from prismatic.models.projectors import ProprioProjector
        ap_path = ckpt_dir / f"abs_pose_projector--{step}_checkpoint.pt"
        if not ap_path.exists():
            raise FileNotFoundError(f"abs_pose_encoder enabled but no checkpoint at {ap_path}")
        llm_dim = vla.base_model.model.llm_dim
        abs_pose_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=6).to(
            dtype=torch.bfloat16, device=device_id)
        abs_pose_projector.load_state_dict(
            torch.load(ap_path, weights_only=True, map_location="cpu"))
        abs_pose_projector.eval()
        num_patches += 1
        print(f"[load_model]   abs-pose encoder from {ap_path.name}; NUM_PATCHES -> {num_patches}")

    return (vla, action_head, pose_projector, device_id, num_patches,
            action_tokenizer, processor, abs_pose_projector)


class VLAPolicy:
    """VLA Policy class that uses a VLA to predict the next action of the robot."""

    def __init__(self, cfg, controller, control_hz):
        """initialize a VLA Policy object."""
        self.cfg = cfg
        self.controller = controller
        self.dt_s = 1.0 / float(control_hz)
        self.waypoint_spacing = float(self.controller.max_linear_vel_mps) / float(control_hz)

        # Multi-view (CCTV) path is opt-in via a `cameras` list in the policy
        # cfg. Absent -> legacy single-camera path (cur_image_cam_name), so all
        # existing eval configs keep working unchanged.
        self.cameras = OmegaConf.select(self.cfg, "cameras", default=None)
        self.use_cctv_goal = bool(OmegaConf.select(self.cfg, "use_cctv_goal", default=False))
        # 4-CCTV pose-only path: drop ALL goal-image slots (goal = pose token).
        # Slots become [ego_obs, cctv_obs...] + pose, matching how the 4-CCTV
        # model was trained (no_goal_image=true). Default false = legacy behaviour.
        self.no_goal_image = bool(OmegaConf.select(self.cfg, "no_goal_image", default=False))
        # Ablation escape hatch: when true, allow cameras[0] (the onboard ego view)
        # to be dropped (present: false) so a single-stream ablation eval can mask
        # the onboard input. Default false keeps the onboard-always guard active for
        # EVERY real run — only deliberate diagnostic configs opt out.
        self.allow_drop_onboard = bool(OmegaConf.select(self.cfg, "allow_drop_onboard", default=False))
        # Blockade-signal ablation (eval side of the training cctv_blockade_signal):
        # replace each CCTV frame with a solid black (owned aisle sealed) / white
        # (clear) image at inference — the bwsignal model saw ONLY solid frames in
        # training, so real pixels would be OOD. Sealed set primed per scene via
        # prime_blockade_signal(). No-op unless flag set; ego view never touched.
        self.blockade_signal = bool(OmegaConf.select(self.cfg, "blockade_signal", default=False))
        self._blocked_aisles = frozenset()
        # Per-camera responsibility crop (eval side of the training data.cctv_crop):
        # crop each CCTV frame to its camera's fixed box (ref-size coords, scaled to
        # the live frame) BEFORE the transformer resize — mirrors the dataset's
        # crop-then-resize exactly. A crop-trained model fed uncropped frames would
        # be train/test mismatched. None => no-op (every existing config unchanged).
        _crop = OmegaConf.select(self.cfg, "finetune.cctv_crop", default=None)
        self._cctv_crop = (
            {c: tuple(int(v) for v in box) for c, box in
             OmegaConf.to_container(_crop, resolve=True).items()}
            if _crop is not None else None)
        self._cctv_crop_ref = int(OmegaConf.select(self.cfg, "finetune.cctv_crop_ref_size", default=720))
        # Absolute-pose token (eval side of train abs_pose_encoder): world-frame xy
        # normaliser bounds; MUST match the training data.abs_pose_bounds. Consumed in
        # _forward only when the abspose projector is loaded. None => off.
        _apb = OmegaConf.select(self.cfg, "finetune.abs_pose_bounds", default=None)
        self._abs_pose_bounds = (OmegaConf.to_container(_apb, resolve=True)
                                 if _apb is not None else None)

        self.inference_config = InferenceConfig(**OmegaConf.to_container(self.cfg.inference, resolve=True))
        if self.cameras is not None:
            # The onboard (ego) observation MUST always be fed: cameras[0] is the
            # ego view and must be present. (Order is also load-bearing: ego obs
            # at slot 0, ego goal at slot 1 — matches OmniVLA pretraining; cctv
            # appended after.) Guard so a config can't silently drop the onboard.
            if not self.allow_drop_onboard and not bool(self.cameras[0].present):
                raise ValueError(
                    "cameras[0] (onboard ego view) must have present: true — the "
                    "onboard observation must always be fed to the model. (Set "
                    "allow_drop_onboard: true to override for a single-stream ablation.)")
            # Slot layout = [ego_obs, ego_goal, cctv_obs, (cctv_goal)]: every view
            # contributes an obs slot; the ego view (index 0) always contributes a
            # goal slot, the rest only when use_cctv_goal. Deriving num_images here
            # keeps it a single source of truth (config can't drift from reality).
            n_img = self._num_image_slots()
            self.inference_config.num_images_in_input = n_img
            print(f"[VLAPolicy] CCTV multi-view: {len(self.cameras)} views -> "
                  f"num_images_in_input={n_img}, use_cctv_goal={self.use_cctv_goal}")

        (self.vla, self.action_head, self.pose_projector, self.device_id,
         self.NUM_PATCHES, self.action_tokenizer, self.processor,
         self.abs_pose_projector) = load_model(self.cfg, self.inference_config)
        # An abspose checkpoint needs its normaliser bounds to build the world-frame
        # vector — fail loudly rather than silently mis-normalise.
        if self.abs_pose_projector is not None and self._abs_pose_bounds is None:
            raise ValueError(
                "finetune.abs_pose_encoder is set but finetune.abs_pose_bounds is "
                "missing — add the same bounds the model trained with.")
        self.inference = None
        self.task = None
        self.goal = None
        self.goal_image = None
        self.subgoal_selector = None
        self.rgb_pil = None

    def _num_image_slots(self) -> int:
        """Image-slot count for the CCTV layout: one obs slot per view, plus a
        goal slot for the ego view (always) and the others (iff use_cctv_goal).
        no_goal_image drops ALL goal slots -> exactly one obs slot per view."""
        if self.no_goal_image:
            return len(self.cameras)
        n = 0
        for i, _view in enumerate(self.cameras):
            n += 1                                   # obs slot
            if i == 0 or self.use_cctv_goal:         # goal slot
                n += 1
        return n

    def set_task(self, task, effective_goal):
        """
        Set the task for VLA Policy. 

        This function creates an inference objects, and get the goal graph if one is necessary.
        Also sets flags for modalities.        
        """
        self.reset_task()
        self.task = task
        self.inference = Inference(
            save_dir=self.cfg.save_dir,
            lan_inst_prompt=self.task.lang_instruction,
            goal_image_PIL=None,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )
        self.goal = effective_goal
        self.subgoal_selector = None
        if self.task.goal_image_path:
            self.goal_image = Image.open(self.task.goal_image_path).convert("RGB")
        if self.task.goal_graph_path:
            graph_path = pathlib.Path(self.task.goal_graph_path)
            graph = GoalGraph.load(graph_path)
            image_dir = GoalGraph.image_dir_for(graph_path)
            # CCTV path: register each non-ego view's goal-frame sibling dir so
            # the selector can serve that view's goal image at the same node.
            # e.g. <graph_dir>/cctv/NNN.png alongside the ego <graph_dir>/frames/.
            extra_image_dirs = self._cctv_goal_dirs(graph_path)
            self.subgoal_selector = SubgoalSelector(
                graph, image_dir, extra_image_dirs=extra_image_dirs
            )
        if self.task.goal_image_path and self.task.goal_graph_path:
            print("[Policy] can't have subgoal image and subgoal image path set at the same time")

        flags = task.to_modality_flags()
        ro.satellite  = flags["satellite"]
        ro.lan_prompt = flags["lan_prompt"]
        ro.pose_goal  = flags["pose_goal"]
        ro.image_goal = flags["image_goal"]

    def reset_task(self):
        """Resets the task of the VLA Policy."""
        self.inference = None

    def prime_blockade_signal(self, scene_name):
        """Refresh the sealed-aisle set for the current scene (called per run, so
        KEEP_POLICY reuses across scenes correctly). No-op unless blockade_signal.
        Reuses the TRAINING parser for `_cf` scenes; adds a `probe_<AISLE>` branch
        for the hand-designed probe scenes (`probe_none` -> nothing sealed)."""
        if not self.blockade_signal:
            return
        if "_cf" in scene_name:
            self._blocked_aisles = _parse_blocked_aisles(scene_name)
        elif "probe_" in scene_name:
            tok = scene_name.split("probe_", 1)[1].split("_")[0]   # "B"/"C"/"E"/"F"/"none"
            self._blocked_aisles = frozenset() if tok == "none" else frozenset({tok})
        else:
            raise ValueError(
                f"blockade_signal: scene {scene_name!r} is neither a _cf nor a "
                "probe_* scene — cannot determine the sealed-aisle set.")

    def _cctv_goal_dirs(self, graph_path):
        """Map each non-ego view's `goal_frames` name to its sibling frame dir
        under the goal-graph dir, when use_cctv_goal. Returns None in the legacy
        / no-cctv-goal case so SubgoalSelector behaves exactly as before."""
        if self.cameras is None or not self.use_cctv_goal:
            return None
        dirs = {}
        for i, view in enumerate(self.cameras):
            if i == 0:
                continue  # ego view's goal is the main graph frames/ dir
            goal_frames = OmegaConf.select(view, "goal_frames", default=None)
            if goal_frames:
                dirs[goal_frames] = GoalGraph.image_dir_for(graph_path) / goal_frames
        return dirs or None

    def _forward(self, observations):
        """Run one VLA forward pass on `observations`.

        Shared by __call__ (legacy single-waypoint twist) and predict_world_path
        (high-rate follower). Returns (chunk, trace):
          chunk: np.ndarray (num_chunk, 4) — predicted waypoints in NORMALIZED
                 units (cols 0,1 = fwd/left in waypoint_spacing units; cols 2,3 =
                 heading cos/sin). Scale cols 0,1 by self.waypoint_spacing for m.
          trace: viz side-channel (cur_img, goal_img, waypoints, modality, ...).
        """
        cameras = observations["images"]
        if self.blockade_signal:
            # Overwrite CCTV frames in-place with the solid black/white bit the
            # model trained on; ego cam (read below) stays the real onboard frame.
            for _cam, _aisle in _CCTV_CORNER_TO_AISLE_2X3.items():
                if _cam in cameras:
                    cameras[_cam] = np.full_like(
                        cameras[_cam], _blockade_shade(_aisle, self._blocked_aisles))
        if self._cctv_crop is not None:
            # Crop each CCTV frame to its camera's responsibility box (box coords are
            # in _cctv_crop_ref-square space, scaled to the live frame — mirrors the
            # dataset's _cctv_crop_box + crop; the transformer's resize follows).
            for _cam, _box in self._cctv_crop.items():
                if _cam in cameras:
                    _h, _w = cameras[_cam].shape[:2]
                    _sx, _sy = _w / self._cctv_crop_ref, _h / self._cctv_crop_ref
                    _x0, _y0, _x1, _y1 = _box
                    cameras[_cam] = cameras[_cam][
                        round(_y0 * _sy):round(_y1 * _sy),
                        round(_x0 * _sx):round(_x1 * _sx)]
        cur_x = observations["x"]
        cur_y = observations["y"]
        cur_yaw = observations["heading_rad"]
        # Ego "current image": the configured single cam (legacy) or the first
        # view in the CCTV camera list. This is the image goal_pose is anchored to.
        ego_cam_name = self.cameras[0].obs_cam if self.cameras is not None else self.cfg.cur_image_cam_name
        cur_img = Image.fromarray(cameras[ego_cam_name])
        gx, gy = self.goal["xy"]
        dx_w = gx - cur_x
        dy_w = gy - cur_y
        cos_h, sin_h = math.cos(cur_yaw), math.sin(cur_yaw)
        fwd = dx_w * cos_h + dy_w * sin_h
        left = -dx_w * sin_h + dy_w * cos_h
        radius = math.hypot(fwd, left)
        if radius > self.controller.thres_dist_m:
            fwd *= self.controller.thres_dist_m / radius
            left *= self.controller.thres_dist_m / radius
        dh = self.goal["heading_rad"] - cur_yaw
        goal_pose_loc_norm = np.array([
            fwd  / self.waypoint_spacing,
            left / self.waypoint_spacing,
            math.cos(dh),
            math.sin(dh),
        ])

        # Absolute-pose token: world-frame [x, y, cos(yaw), sin(yaw), goal_x, goal_y]
        # with xy normalised to [-1,1] by abs_pose_bounds — mirrors the dataset
        # (cctv_isaac_sim_dataset.py:593-602). Uses the RAW world goal (unclamped,
        # like training), not the clamped relative goal_pose above. Built only when
        # the abspose model is loaded (else None -> forward passes abs_proprio=None).
        abs_pose_vec = None
        if self.abs_pose_projector is not None:
            _xb, _yb = self._abs_pose_bounds["x"], self._abs_pose_bounds["y"]
            _nx = lambda v: 2.0 * (v - _xb[0]) / (_xb[1] - _xb[0]) - 1.0
            _ny = lambda v: 2.0 * (v - _yb[0]) / (_yb[1] - _yb[0]) - 1.0
            abs_pose_vec = np.array([
                _nx(cur_x), _ny(cur_y), math.cos(cur_yaw), math.sin(cur_yaw),
                _nx(gx), _ny(gy),
            ], dtype=np.float32)

        if self.subgoal_selector is not None:
            self.subgoal_selector.update((cur_x, cur_y))
            goal_img = self.subgoal_selector.current_image()
        else:
            goal_img = self.goal_image if self.goal_image is not None else cur_img

        lan_inst = self.task.lang_instruction if self.task.to_modality_flags()["lan_prompt"] else "xxxx"
        # CCTV multi-view extras (only populated on the multi-view path).
        cctv_img = None
        cctv_goal_img = None
        if self.cameras is None:
            # --- Legacy single-view path (unchanged) ---
            batch = self.inference.data_transformer_omnivla(
                current_image_PIL=cur_img,
                lan_inst=lan_inst,
                goal_image_PIL=goal_img,
                goal_pose_loc_norm=goal_pose_loc_norm,
                prompt_builder=PurePromptBuilder,
                action_tokenizer=self.action_tokenizer,
                processor=self.processor,
            )
        else:
            # --- CCTV multi-view path ---
            # Build the model's ordered image stack + block_present mask. Slot
            # order is LOAD-BEARING: [ego_obs, ego_goal, cctv_obs, (cctv_goal)],
            # block_present = [...slots..., pose]. Goal presence couples to obs
            # presence + the image-goal modality (mirrors training).
            flags = self.task.to_modality_flags()
            modality_img = bool(flags["image_goal"])
            modality_pose = bool(flags["pose_goal"])

            ordered_images = []
            block_flags = []
            if self.no_goal_image:
                # Pose-only goal: one obs slot per view, NO goal slots ->
                # [ego_obs, cctv_obs...] + pose. Mirrors the 4-CCTV training set
                # (cctv_isaac_sim_dataset.py no_goal_image branch). Falls through
                # to the shared trailing-pose + forward-pass code below.
                for view in self.cameras:
                    ordered_images.append(Image.fromarray(cameras[view.obs_cam]))
                    block_flags.append(bool(view.present))
            else:
                for i, view in enumerate(self.cameras):
                    present = bool(view.present)
                    obs_img = Image.fromarray(cameras[view.obs_cam])
                    ordered_images.append(obs_img)        # obs slot
                    block_flags.append(present)
                    if i == 0 or self.use_cctv_goal:       # goal slot exists for this view
                        goal_present = present and modality_img
                        if i == 0:
                            # Ego goal = the goal image already selected above.
                            view_goal = goal_img
                        elif goal_present:
                            # Non-ego goal = this view's frame at the SAME node the
                            # ego graph selected. Only fetched when the slot is
                            # actually visible (else it'd be masked anyway), so a
                            # pose-only task won't demand a CCTV goal graph.
                            if self.subgoal_selector is None:
                                raise RuntimeError(
                                    f"CCTV goal needed for view '{view.obs_cam}' (image-goal "
                                    f"modality) but no goal graph is set (task.goal_graph_path "
                                    f"is empty)."
                                )
                            view_goal = self.subgoal_selector.current_image_for(view.goal_frames)
                            cctv_img, cctv_goal_img = obs_img, view_goal
                        else:
                            # Masked goal slot: placeholder, zero-filled in the transformer.
                            view_goal = obs_img
                            cctv_img = obs_img
                        ordered_images.append(view_goal)
                        block_flags.append(goal_present)
                    elif i > 0:
                        cctv_img = obs_img
            block_flags.append(modality_pose)          # trailing pose block
            block_present = np.array(block_flags, dtype=np.float32)

            batch = self.inference.data_transformer_omnivla_cctv(
                ordered_images=ordered_images,
                block_present=block_present,
                lan_inst=lan_inst,
                goal_pose_loc_norm=goal_pose_loc_norm,
                prompt_builder=PurePromptBuilder,
                action_tokenizer=self.action_tokenizer,
                processor=self.processor,
            )
        # Attach the abs-pose token to the batch (both the legacy and cctv paths
        # built `batch` above); run_forward_pass forwards abs_proprio iff the
        # projector is set AND "abs_pose" is present.
        if abs_pose_vec is not None:
            batch["abs_pose"] = torch.from_numpy(abs_pose_vec).unsqueeze(0)   # (1, 6)
        actions, _modality_id = self.inference.run_forward_pass(
            vla=self.vla.eval(),
            action_head=self.action_head.eval(),
            noisy_action_projector=None,
            pose_projector=self.pose_projector.eval(),
            batch=batch,
            action_tokenizer=self.action_tokenizer,
            device_id=self.device_id,
            use_l1_regression=self.inference_config.use_l1_regression,
            use_diffusion=self.inference_config.use_diffusion,
            use_film=self.inference_config.use_film,
            num_patches=self.NUM_PATCHES,
            abs_pose_projector=self.abs_pose_projector,
        )

        waypoints = actions.float().cpu().numpy()
        trace = {
            "cur_img": cur_img,
            "goal_img": goal_img,
            "goal_pose_loc_norm": goal_pose_loc_norm,
            "waypoints": waypoints[0],
            "modality_id": _modality_id.cpu().numpy(),
            # CCTV multi-view extras (None on the legacy path).
            "cctv_img": cctv_img,
            "cctv_goal_img": cctv_goal_img,
        }
        return waypoints[0], trace

    def predict_target_waypoint(self, observations):
        """VLA forward -> the WORLD-frame position of the lookahead waypoint
        (chunk index controller.waypoint_select, ~1.667 m ahead) for the
        fine-grained follower. Returns (target_xy, trace).

        The follower steers toward this single fixed point at high rate
        (recomputed from the live pose) until the next ~3 Hz query refreshes it.
        We pick the SAME chunk index the legacy controller used (waypoint_select)
        and transform it to world so the high-rate loop can re-aim as the robot
        moves. Why a fixed point and not pure-pursuit over the whole chunk: at
        3 Hz the robot travels ~one waypoint-spacing (0.333 m) per query, so
        re-fetching this waypoint each query advances the carrot ~one node — the
        same net progress pure-pursuit's sliding would make.
        chunk[k]=[fwd, left, ..] (norm) -> ×waypoint_spacing (m) -> rotate by the
        current (anchor) yaw + translate to world.
        """
        chunk, trace = self._forward(observations)
        wp = chunk[self.controller.waypoint_select]
        fwd = float(wp[0]) * self.waypoint_spacing   # m, robot frame (forward)
        left = float(wp[1]) * self.waypoint_spacing  # m, robot frame (left)
        ax, ay, ayaw = observations["x"], observations["y"], observations["heading_rad"]
        cos_a, sin_a = math.cos(ayaw), math.sin(ayaw)
        # robot->world: R(+yaw) applied to (fwd, left), then translate.
        tx = ax + fwd * cos_a - left * sin_a
        ty = ay + fwd * sin_a + left * cos_a
        return (tx, ty), trace

    def __call__(self, observations, step):
        """Legacy single-waypoint controller: VLA forward -> pick waypoint
        `waypoint_select` from the chunk -> one (v, w) twist held until the next
        query. Returns (linear_vel_limited, angular_vel_limited, trace). The
        high-rate follower path uses predict_world_path instead (see
        waypoint_follower.py / eval_policy fine_grained loop).
        """
        chunk, trace = self._forward(observations)
        chosen = chunk[self.controller.waypoint_select].copy()
        chosen[:2] *= self.waypoint_spacing
        dx, dy, hx, hy = chosen

        if np.abs(dx) < self.controller.eps and np.abs(dy) < self.controller.eps:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.arctan2(hy, hx) / self.dt_s
        elif np.abs(dx) < self.controller.eps:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * self.dt_s)
        else:
            linear_vel_value = dx / self.dt_s
            angular_vel_value = np.arctan(dy / dx) / self.dt_s

        print("Raw outputs: ", linear_vel_value, angular_vel_value)
        print("skipping the non ratio preserving clip")

        maxv, maxw = self.controller.max_linear_vel_mps, self.controller.max_angular_vel_radps
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)

        print("linear angular", linear_vel_value_limit, angular_vel_value_limit)

        return linear_vel_value_limit, angular_vel_value_limit, trace
