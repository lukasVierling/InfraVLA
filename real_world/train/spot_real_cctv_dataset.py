"""SpotRealDataset + real CCTV frames, for train_omnivla_sim_cctv_v2.py.

Extends the real-Spot dataset (see spot_real_dataset.py) with the CCTV sample
contract of `prismatic/vla/datasets/cctv_isaac_sim_dataset.py`, so the v2 CCTV
trainer, its collator (`PaddedCollatorForActionPrediction_Nav_MMN` with the
cctv branch) and `_build_multimodal_attention_cctv` all work UNMODIFIED. Lives
under real_world/train/; nothing in the sim pipeline is touched.

Sample contract mirrored from the sim dataset (order is load-bearing):
  1 cam  -> legacy single-cctv path: `cctv_pixel_values` tensor;
            collator stacks [ego_obs, ego_goal, cctv]           (n_img = 3)
            block_present = [ego, ego_goal, cctv, pose]
  >=2 cams -> multi path: `cctv_pixel_values_list`;
            collator stacks [ego_obs, ego_goal, cctv_0..N-1]    (n_img = 2+N)
            block_present = [ego, ego_goal, cctv_0.., pose]

Data source: export_spot_cctv_2026_08_19 runs, whose dataset.npz carries
`cctv_cams` and per-cam `cctv_<cam>` basename arrays parallel to `images`
("" = no aligned frame -> zeros + block_present 0, same as a sim trajectory
with no cctv/ dir). A run missing a configured cam entirely behaves the same.

Deliberate differences from the sim dataset, and why:
  * random_hflip is REFUSED (constructor error) rather than silently accepted:
    camera3/4 are fixed external views - mirroring the ego image and action
    labels without mirroring the world breaks the ego<->cctv correspondence
    that this dataset exists to teach.
  * perception_dropout draws (both / ego-only / cctv-only) per sample like the
    sim dataset; a masked block is a ZERO image + block_present 0.
  * Unsupported sim knobs (use_cctv_goal, no_goal_image, blockade signal,
    crops, lang manifests, final_goal_prob...) are accepted but must be at
    their inert defaults - anything else raises, so a config typo cannot
    silently train a different experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spot_real_dataset import SpotRealDataset, _RunCache


class _CCTVRunCache(_RunCache):
    __slots__ = ("cctv_names",)

    def __init__(self, run_dir: Path):
        super().__init__(run_dir)
        z = np.load(Path(run_dir) / "dataset.npz", allow_pickle=True)
        cams = [str(c) for c in z["cctv_cams"]] if "cctv_cams" in z.files else []
        self.cctv_names: dict[str, list[str]] = {
            c: [str(s) for s in z[f"cctv_{c}"]] for c in cams
        }


class SpotRealCCTVDataset(SpotRealDataset):
    DATASET_NAME = "spot_real_cctv"

    def __init__(
        self,
        *args,
        cctv_cam_names: tuple[str, ...] = ("camera4",),
        perception_dropout: tuple[float, float, float] = (1.0, 0.0, 0.0),
        cctv_cam_dropout: float = 0.0,
        ego_blackout: bool = False,
        # sim-parity knobs this dataset does NOT implement - must stay inert
        use_cctv_goal: bool = False,
        no_goal_image: bool = False,
        cctv_blockade_signal: bool = False,
        cctv_crop=None,
        cctv_crop_ref_size: int = 720,
        modality_weights=None,
        lang_manifest=None,
        lang_pattern=None,
        first_step_only: bool = False,
        keep_steps_max=None,
        final_goal_prob: float = 0.0,
        abs_pose: bool = False,
        abs_pose_bounds=None,
        deterministic_sampling: bool = False,
        **kwargs,
    ) -> None:
        inert = dict(use_cctv_goal=use_cctv_goal, no_goal_image=no_goal_image,
                     cctv_blockade_signal=cctv_blockade_signal, cctv_crop=cctv_crop,
                     lang_manifest=lang_manifest,
                     lang_pattern=lang_pattern, first_step_only=first_step_only,
                     keep_steps_max=keep_steps_max,
                     abs_pose=abs_pose, abs_pose_bounds=abs_pose_bounds)
        bad = {k: v for k, v in inert.items() if v not in (False, None, 0.0)}
        if bad:
            raise ValueError(f"SpotRealCCTVDataset does not implement: {bad}")
        if kwargs.get("random_hflip"):
            raise ValueError(
                "random_hflip=True with fixed external CCTV cameras would mirror "
                "the ego view and action labels but not the world seen by "
                "camera3/4, corrupting exactly the correspondence this dataset "
                "teaches. Set data.random_hflip: false."
            )
        if abs(sum(perception_dropout) - 1.0) > 1e-6:
            raise ValueError(f"perception_dropout must sum to 1, got {perception_dropout}")
        # now implemented in SpotRealDataset -> forward instead of refusing
        kwargs.setdefault("modality_weights", modality_weights)
        kwargs.setdefault("final_goal_prob", final_goal_prob)
        super().__init__(*args, **kwargs)
        # fixed_val_sampling (mirrors the sim dataset): re-seed the RNG from the
        # sample identity on every __getitem__, so the val set is byte-identical
        # each pass and best-val selection compares weights, not sampling luck.
        self.deterministic_sampling = bool(deterministic_sampling)
        self._base_seed = int(kwargs.get("seed", 42))
        self.cctv_cam_names = tuple(cctv_cam_names)
        self.perception_dropout = tuple(perception_dropout)
        self.cctv_cam_dropout = float(cctv_cam_dropout)
        self.ego_blackout = bool(ego_blackout)
        # upgrade the run caches with the cctv name tables
        self.runs = [_CCTVRunCache(r.run_dir) for r in self.runs]
        cover = {c: sum(1 for r in self.runs if c in r.cctv_names)
                 for c in self.cctv_cam_names}
        print(f"[SpotRealCCTVDataset] cams={list(self.cctv_cam_names)} "
              f"run coverage={cover} perception_dropout={self.perception_dropout}")

    # ------------------------------------------------------------------
    def _cctv_pils(self, run: _CCTVRunCache, t: int) -> list[Image.Image | None]:
        """Aligned CCTV frame per configured cam; None = unavailable."""
        pils = []
        for cam in self.cctv_cam_names:
            names = run.cctv_names.get(cam)
            name = names[t] if names else ""
            if not name:
                pils.append(None)
                continue
            p = run.run_dir / "cctv" / cam / name
            pil = Image.open(p).convert("RGB").resize(self.image_size, Image.BILINEAR)
            pils.append(pil)
        return pils

    def __getitem__(self, idx: int) -> dict:
        run_idx, t = self.samples_index[idx]
        if self.deterministic_sampling:
            # every draw below AND inside super().__getitem__ (goal k, modality,
            # perception condition) becomes a pure function of the sample identity
            self._rng.seed(f"{self._base_seed}:{run_idx}:{t}")
        sample = super().__getitem__(idx)
        run = self.runs[run_idx]

        # --- perception-dropout draw (sim semantics: ego always usable in
        # "both"/"ego-only"; cctv usable in "both"/"cctv-only") --------------
        r = self._rng.random()
        p_both, p_ego, _ = self.perception_dropout
        if r < p_both:
            ego_present, cctv_present = True, True
        elif r < p_both + p_ego:
            ego_present, cctv_present = True, False
        else:
            ego_present, cctv_present = False, True

        zeros = torch.zeros_like(sample["pixel_values"])
        if self.ego_blackout or not ego_present:
            sample["pixel_values"] = zeros.clone()

        # --- the CCTV stack --------------------------------------------------
        pils = self._cctv_pils(run, t)
        stack, flags = [], []
        for pil in pils:
            avail = pil is not None and cctv_present
            if avail and self.cctv_cam_dropout > 0.0 and self._rng.random() < self.cctv_cam_dropout:
                avail = False
            stack.append(self.image_transform(pil) if avail else zeros.clone())
            flags.append(1.0 if avail else 0.0)

        # --- block_present, in the collator's exact concat order -------------
        modality_id = int(sample["modality_id"])
        # sim rule is `in (4, 5)` because the sim never used modality 8;
        # 8 = language + POSE, so its goal-pose block must be present too.
        modality_pose = modality_id in (4, 5, 8)
        modality_img = modality_id in (5, 6)
        ego_goal_present = ego_present and modality_img
        block = [float(ego_present), float(ego_goal_present)] + flags + [float(modality_pose)]
        sample["block_present"] = np.array(block, dtype=np.float32)
        sample["dataset_name"] = self.DATASET_NAME

        if len(self.cctv_cam_names) == 1:
            sample["cctv_pixel_values"] = stack[0]        # legacy 3-image path
        else:
            sample.pop("cctv_pixel_values", None)
            sample["cctv_pixel_values_list"] = stack      # multi-cctv path
        return sample
