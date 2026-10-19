"""Real-Spot trajectory dataset for OmniVLA fine-tuning.

Reads the bags collected on 2026-08-19 (exported by export_trainset.py, removed
2026-09-08 — recover from tag pre-cleanup-2026-09-04 if that session is revisited) and
emits samples field-for-field identical to `IsaacSimDataset`, so the existing
trainer, `PaddedCollatorForActionPrediction_Nav_MMN`, and checkpoints all work
UNMODIFIED. Lives under real_world/train/ so nothing in the sim pipeline is
touched.

Differences from IsaacSimDataset, and why:

  * Data source is one `dataset.npz` per run (images list, precomputed 3 Hz
    action chunks, world poses, timestamps, the language prompt) instead of a
    GoalGraph directory. The chunks were built by time-interpolating 50 Hz
    odometry onto each camera frame and validated against the driven path
    (chunks overlay the recorded trajectory exactly); recomputing them here
    from per-frame poses would re-introduce the frame-rate (15.6 Hz) as the
    action timestep, which is wrong — deployment replans at 3 Hz.
  * Language modalities (7 = language only, 8 = language + pose) are
    available: every take was driven toward a named object. The prompt is
    formatted exactly as lelan_dataset and inference-side run_omnivla do:
        "What action should the robot take to {instruction}?"
  * Goal sampling is done in TIME (3 Hz ticks ahead), then snapped to the
    nearest recorded frame, because our frames are 15.6 Hz, not tick-aligned.
  * Horizontal flip also flips nothing language-side: both instructions
    ("blue bin", "traffic cone") are side-agnostic nouns.

Constructor signature mirrors IsaacSimDataset exactly so the trainer's
`common_dataset_kwargs` pass through untouched (waypoint_spacing and
emit_cctv are accepted for compatibility; there is no CCTV stream yet, so
cctv_pixel_values is a zero tensor exactly like the sim dataset emits when a
trajectory has no cctv/ dir).
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK

TICK_HZ = 3.0  # OmniVLA replans at 3 Hz; the exported chunks use this timestep.


class _RunCache:
    """One recorded take, loaded once at __init__."""

    __slots__ = ("run_dir", "pano_dir", "images", "chunks", "pose", "stamps",
                 "prompt", "n_frames")

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.pano_dir = self.run_dir / "pano"
        z = np.load(self.run_dir / "dataset.npz", allow_pickle=True)
        self.images = [str(s) for s in z["images"]]          # basenames
        self.chunks = z["chunks"].astype(np.float32)         # (n, N, 4)
        self.pose = z["pose"].astype(np.float64)             # (n, 3) world x, y, yaw
        self.stamps = z["stamps"].astype(np.float64)         # (n,)
        self.prompt = str(z["prompt"])                       # "move toward the blue bin"
        self.n_frames = len(self.images)


class SpotRealDataset(Dataset):
    DATASET_NAME = "spot_real"

    def __init__(
        self,
        # External dependencies (provided by trainer) — identical to IsaacSimDataset.
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        # Data source.
        traj_root: Path,
        max_trajectories: int | None = None,
        # Behaviour (mirror YAML field names).
        context_size: int = 5,
        num_actions_chunk: int = NUM_ACTIONS_CHUNK,
        modality_list: tuple[int, ...] = (4, 5, 6, 7, 8),
        modality_weights=None,                   # per-modality sampling mass
        final_goal_prob: float = 0.0,            # P(goal = last frame of the take)
        image_only_max_distance_steps: int = 20,
        min_goal_distance_steps: int = 1,
        max_goal_distance_steps: int = 20,
        metric_waypoint_spacing: float = 0.09,   # derived from the recorded speed
        waypoint_spacing: int = 1,               # accepted for signature parity; unused
        image_size: tuple[int, int] = (224, 224),
        random_hflip: bool = True,
        goals_per_obs: int = 1,
        emit_cctv: bool = True,                  # parity; no CCTV stream recorded yet
        predict_stop_token: bool = True,
        seed: int = 42,
    ) -> None:
        if num_actions_chunk != NUM_ACTIONS_CHUNK:
            raise ValueError(
                f"num_actions_chunk={num_actions_chunk} must equal "
                f"prismatic.vla.constants.NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}"
            )
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        self.context_size = context_size
        self.num_actions_chunk = num_actions_chunk
        self.modality_list = list(modality_list)
        # Weighted modality sampling, matching IsaacSimDataset. The sim lang arms
        # put 0.8 of the mass on language; a flat choice over 5 ids gives language
        # only ~40%, so this knob is what makes the two runs comparable.
        self.modality_weights = list(modality_weights) if modality_weights else None
        if self.modality_weights and len(self.modality_weights) != len(self.modality_list):
            raise ValueError(f"modality_weights has {len(self.modality_weights)} entries "
                             f"but modality_list has {len(self.modality_list)}")
        self.final_goal_prob = float(final_goal_prob)
        self.image_only_max_distance_steps = image_only_max_distance_steps
        self.min_goal_distance_steps = min_goal_distance_steps
        self.max_goal_distance_steps = max_goal_distance_steps
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.image_size = tuple(image_size)
        self.random_hflip = random_hflip
        self.goals_per_obs = goals_per_obs
        self.predict_stop_token = predict_stop_token
        self._rng = random.Random(seed)

        run_dirs = sorted(
            d for d in Path(traj_root).iterdir()
            if d.is_dir() and (d / "dataset.npz").is_file()
        )
        if max_trajectories is not None:
            run_dirs = run_dirs[:max_trajectories]
        if not run_dirs:
            raise RuntimeError(f"no exported runs (dirs with dataset.npz) under {traj_root}")
        self.runs: list[_RunCache] = [_RunCache(d) for d in run_dirs]

        # Sanity: chunk spacing must match what the export was built with.
        for r in self.runs:
            z = np.load(r.run_dir / "dataset.npz", allow_pickle=True)
            sp = float(z["spacing_m"])
            if abs(sp - metric_waypoint_spacing) > 1e-6:
                raise ValueError(
                    f"{r.run_dir.name}: exported with spacing {sp} but config says "
                    f"metric_waypoint_spacing={metric_waypoint_spacing}. These MUST "
                    f"match deployment (policy_server --spacing / operator_loop "
                    f"spacing_m) or the robot will mis-scale every step."
                )

        self.samples_index: list[tuple[int, int]] = []
        for ri, run in enumerate(self.runs):
            # Every exported frame already has a full valid chunk (the exporter
            # applied the horizon + dead-lead-in filters), but the GOAL needs at
            # least min_goal_distance_steps of future too.
            min_goal_dt = self.min_goal_distance_steps / TICK_HZ
            t_last = run.stamps[-1]
            for t in range(run.n_frames):
                if run.stamps[t] + min_goal_dt > t_last:
                    break
                for _ in range(self.goals_per_obs):
                    self.samples_index.append((ri, t))

        n_samples, n_runs = len(self.samples_index), len(self.runs)
        print(f"[SpotRealDataset] {n_runs} runs → {n_samples} samples "
              f"(avg {n_samples / max(n_runs, 1):.1f} per run)  "
              f"spacing={metric_waypoint_spacing} m  modalities={self.modality_list}")

    def __len__(self) -> int:
        return len(self.samples_index)

    @property
    def trajectories(self):
        """Adapter for cf_windows.cf_window_sample_mask, which expects the sim
        dataset layout: .trajectories entries carrying .traj_dir / .positions /
        .yaws. Our pose array is [x, y, yaw] per valid frame, index-aligned
        with samples_index[..][1] - exactly the indexing the miner uses."""
        class _TrajView:
            __slots__ = ("traj_dir", "positions", "yaws", "n_frames")

            def __init__(self, run):
                self.traj_dir = run.run_dir
                self.positions = run.pose[:, :2]
                self.yaws = run.pose[:, 2]
                self.n_frames = run.n_frames
        return [_TrajView(r) for r in self.runs]

    # ------------------------------------------------------------------
    def _goal_frame(self, run: _RunCache, t: int) -> tuple[int, int]:
        """Sample a goal frame k ticks ahead; snap to the nearest recorded frame.

        Returns (goal_frame_index, distance_in_ticks). The tick distance is
        recomputed from the actual snapped timestamp so the image-only guard
        sees the true horizon even at the end of a run.
        """
        if self.final_goal_prob and self._rng.random() < self.final_goal_prob:
            # the takes LAST frame: the robot standing at the target. Sim uses 0.5,
            # which is what teaches goal-reaching rather than short-horizon drift.
            j = min(max(t + 1, run.n_frames - 1), run.n_frames - 1)
            if j > t:
                return j, max(1, int(round((run.stamps[j] - run.stamps[t]) * TICK_HZ)))
        k = self._rng.randint(self.min_goal_distance_steps, self.max_goal_distance_steps)
        t_goal = run.stamps[t] + k / TICK_HZ
        j = int(np.searchsorted(run.stamps, t_goal))
        j = min(max(j, t + 1), run.n_frames - 1)
        distance_steps = max(1, int(round((run.stamps[j] - run.stamps[t]) * TICK_HZ)))
        return j, distance_steps

    # attached post-init by onboard_swap.build_onboard_swap_map (opt-in)
    onboard_swap_map: dict | None = None
    onboard_swap_p: float = 0.0

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples_index[idx]
        run = self.runs[run_idx]

        goal_t, distance_steps = self._goal_frame(run, t)

        # ---- counterfactual ONBOARD swap (mirrors cctv_isaac_sim_dataset) --
        # An eligible spawn frame may read its onboard bundle (ego obs image,
        # ego goal image, goal_pose) from a cf SIBLING at the same index while
        # the action labels, instruction and CCTV frames stay the sample's own.
        # Within the gated frames the twins are physically interchangeable
        # (graft endpoint error capped at build time), so this makes the ego
        # start-view non-predictive of WHICH object is targeted - the
        # instruction has to explain the label.
        src = run
        if self.onboard_swap_map and self.onboard_swap_p > 0.0:
            sibs = self.onboard_swap_map.get((run_idx, t))
            if sibs and self._rng.random() < self.onboard_swap_p:
                pick = self._rng.choice([run_idx] + list(sibs))  # uniform incl own
                src = self.runs[pick]
        s_t = min(t, src.n_frames - 1)
        s_goal = min(goal_t, src.n_frames - 1)

        # Modality: same long-distance rule as IsaacSimDataset — image-only (6)
        # is unreliable when the goal frame is far, so drop it there. Language
        # (7/8) and pose (4/5) work at any distance.
        if distance_steps <= self.image_only_max_distance_steps:
            pool, wts = self.modality_list, self.modality_weights
        else:
            keep = [i for i, m in enumerate(self.modality_list) if m != 6]
            if not keep:
                keep = list(range(len(self.modality_list)))
            pool = [self.modality_list[i] for i in keep]
            # dropping 6 also drops its mass; random.choices renormalises over the
            # rest, so the language share survives at long goal distances.
            wts = [self.modality_weights[i] for i in keep] if self.modality_weights else None
        modality_id = (self._rng.choices(pool, weights=wts, k=1)[0]
                       if wts else self._rng.choice(pool))

        # ---- actions: the precomputed 3 Hz chunk, already body-frame and
        # spacing-normalised: [dx/spacing, dy/spacing, cos dyaw, sin dyaw] -----
        actions = run.chunks[t].copy()                                   # (N, 4)

        # ---- goal pose (pose modalities' conditioning) ---------------------
        # `src` (not `run`): under an onboard swap the pose vector must be
        # consistent with the swapped goal image - sim swaps the whole bundle.
        x0, y0, yaw0 = src.pose[s_t]
        gx, gy, _ = src.pose[s_goal]
        dx, dy = gx - x0, gy - y0
        c, s = math.cos(yaw0), math.sin(yaw0)
        goal_fwd = (dx * c + dy * s) / self.metric_waypoint_spacing
        goal_left = (-dx * s + dy * c) / self.metric_waypoint_spacing
        # De-leak (mirrors isaac_sim_dataset 2026-06-17): the goal heading is the
        # RUN-START heading, not the recorded heading at the goal step, so the
        # pose channel cannot hand the route to the policy.
        goal_yaw_local = src.pose[0][2] - yaw0
        goal_pose = np.array(
            [goal_fwd, goal_left, math.cos(goal_yaw_local), math.sin(goal_yaw_local)],
            dtype=np.float32,
        )

        # ---- images ---------------------------------------------------------
        cur_pil = Image.open(src.pano_dir / src.images[s_t]).convert("RGB")
        goal_pil = Image.open(src.pano_dir / src.images[s_goal]).convert("RGB")
        cur_pil_resized = cur_pil.resize(self.image_size, Image.BILINEAR)
        goal_pil_resized = goal_pil.resize(self.image_size, Image.BILINEAR)

        if self.random_hflip and self._rng.random() > 0.5:
            cur_pil_resized = cur_pil_resized.transpose(Image.FLIP_LEFT_RIGHT)
            goal_pil_resized = goal_pil_resized.transpose(Image.FLIP_LEFT_RIGHT)
            actions[:, 1] = -actions[:, 1]
            actions[:, 3] = -actions[:, 3]
            goal_pose[1] = -goal_pose[1]
            goal_pose[3] = -goal_pose[3]
            # instructions are side-agnostic nouns; nothing to flip in language

        pixel_values = self.image_transform(cur_pil_resized)
        pixel_values_goal = self.image_transform(goal_pil_resized)

        # ---- tokenised prompt + action string (identical to IsaacSimDataset,
        # with the lelan/inference language branch for modalities 7/8) --------
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        current_action_string = self.action_tokenizer(actions_torch[0])
        future_actions_string = "".join(self.action_tokenizer(actions_torch[1:]))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if modality_id in (7, 8):
            human_turn = f"What action should the robot take to {run.prompt}?"
        else:
            human_turn = "No language instruction"

        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", human_turn)
        prompt_builder.add_turn("gpt", action_chunk_string)
        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True,
        ).input_ids
        labels = list(input_ids)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # ---- MBRA/collator dummies — byte-identical to IsaacSimDataset ------
        cur_image_dummy = np.zeros((3 * self.context_size, 64, 85), dtype=np.float32)
        goal_image_8_dummy = np.zeros((3, 64, 85), dtype=np.float32)
        temp_dist_dummy = np.array(0.0, dtype=np.float32)
        action_select_mask = np.array(1.0, dtype=np.float32)
        obj_pose_norm = np.array([0.0, 0.0], dtype=np.float32)

        return dict(
            pixel_values=pixel_values,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=self.DATASET_NAME,
            modality_id=modality_id,
            actions=actions,
            action_select_mask=action_select_mask,
            goal_pose=goal_pose,
            obj_pose_norm=obj_pose_norm,
            cur_image=cur_image_dummy,
            goal_image_8=goal_image_8_dummy,
            temp_dist=temp_dist_dummy,
            img_PIL=cur_pil_resized,
            gimg_PIL=goal_pil_resized,
            cctv_pixel_values=torch.zeros_like(pixel_values),
        )
