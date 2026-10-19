"""Isaac Sim trajectory dataset for OmniVLA fine-tuning.

Mirrors `gnm_dataset.py` structurally so the existing
`PaddedCollatorForActionPrediction_Nav_MMN` consumes our batches without
modification. Key differences from paper's GNM dataset:

  - Single-source (no negative mining across sub-datasets).
  - Raw expert actions only — no MBRA pathway (sim has ground-truth
    velocities; the controller's commanded path *is* the demonstration).
  - Modality list locked to [4, 5, 6] = {pose-only, pose+image, image-only}.
    Append 7 and 8 when language returns.
  - Emits `cctv_pixel_values` in every batch — phase 6 hook. Trainer
    ignores this field in iter 1 but it's there so we don't re-collect.

Output dict per __getitem__ matches paper's gnm_dataset.py field-by-field
where the collator reads it. MBRA-only fields (`cur_image`, `goal_image_8`,
`temp_dist`) are emitted as small zero tensors of the right shape — the
collator stacks them, our trainer doesn't read them.

See: paper/root.tex §3.2 (training method), §4.1 (eval distances),
gnm_dataset.py for the reference shape.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Type

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK

# Make isaac_sim package importable for data_utils (we're under prismatic/
# but the helpers live under isaac_sim/).
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from isaac_sim.planning.image_graph.graph import GoalGraph  # noqa: E402
from isaac_sim.train.data_utils import list_successful_trajectories  # noqa: E402


# ---------------------------------------------------------------------------
# Local geometry helpers — same convention as frodobots_dataset.py:to_local_coords.
# Implemented inline so the dataset has no vint_train import dependency.
# ---------------------------------------------------------------------------

def _yaw_rotmat_2d(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _to_local_xy(positions: np.ndarray, origin: np.ndarray, origin_yaw: float) -> np.ndarray:
    """World-frame XY → local frame anchored at (origin, origin_yaw).

    positions: (..., 2)  world-frame
    Returns:   (..., 2)  local-frame, same shape

    Convention matches frodobots_dataset.py:84-101 — `(p - origin) @ R(yaw)`
    where R(yaw) is the standard 2x2 rotation. This is the *inverse*
    rotation (world → local) because (R^T)^T = R^T applied to a row-vector
    gives R^T @ p = R(-yaw) @ p.
    """
    rotmat = _yaw_rotmat_2d(origin_yaw)
    return (positions - origin) @ rotmat


def _chunk_abs_theta_deg(positions: np.ndarray, yaws: np.ndarray,
                         t: int, num_actions_chunk: int) -> float:
    """|chord angle| (deg) of the GT action chunk starting at frame t.

    Mirrors the __getitem__ chunk construction exactly — window t..t+N with
    terminal-pose padding, local frame at t — but is a pure function of the
    cached positions/yaws (no images), so it can be evaluated for every sample
    at DataLoader-build time. theta = atan2(y_end, x_end) of the chunk's chord
    in the robot frame: ~0 for straight cruise, ~45 for a 90-deg corner inside
    the window. Scale-invariant (the normalisation divides out) and hflip only
    flips its sign, so |theta| is augmentation-independent.

    e.g. 8 straight steps -> 0.0; a quarter-arc corner -> ~45.0.
    """
    N = num_actions_chunk
    end_idx = min(t + N + 1, len(positions))
    pos = positions[t:end_idx]
    if pos.shape[0] < N + 1:                       # pad with terminal pose,
        pad_n = N + 1 - pos.shape[0]               # matches __getitem__ /
        pos = np.concatenate(                      # gnm_dataset.py:242-244
            [pos, np.tile(pos[-1:], (pad_n, 1))], axis=0)
    local = _to_local_xy(pos, pos[0], yaws[t])     # (N+1, 2), robot frame at t
    chord = local[-1]                              # local[0] is the origin
    return abs(math.degrees(math.atan2(chord[1], chord[0])))


def _actions_to_sincos(actions_3d: np.ndarray) -> np.ndarray:
    """(N, 3) [dx, dy, dyaw] → (N, 4) [dx, dy, cos(dyaw), sin(dyaw)].

    cos/sin encoding side-steps the 0/2π discontinuity at the wrap-around
    boundary — gradients stay well-defined regardless of the underlying
    angle. Paper does this via `calculate_sin_cos` in vint_train.
    """
    yaw = actions_3d[..., 2:3]
    return np.concatenate(
        [actions_3d[..., :2], np.cos(yaw), np.sin(yaw)],
        axis=-1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Counterfactual CCTV blockade-signal helpers (cctv_blockade_signal mode).
#
# Ablation: replace each real CCTV frame with a solid black/white image that
# encodes a single bit — "is the transit aisle this camera owns sealed?".
# Black = sealed, white = clear. This strips the CCTV stream down to a clean
# per-aisle blockade signal, isolating whether the model needs to READ the seal
# out of pixels or just needs the bit. Only for the warehouse_2x3 counterfactual
# dataset with the with_4cctv_2x3 corner cams.
# ---------------------------------------------------------------------------

# The four transit aisles of the 2x3 "digital-8" warehouse — the only zones that
# ever get sealed (A/D/G are open strips). See isaac_sim/scenes/_common.py
# (WAREHOUSE_2X3_ZONES).
_TRANSIT_AISLES_2X3 = frozenset("BCEF")

# Corner CCTV cam -> the transit aisle it "owns" for the blockade signal. The 4
# corner cams and 4 transit aisles are a clean 1:1 quadrant match:
#   NW <-> B (west-back)    NE <-> C (east-back)
#   SW <-> E (west-front)   SE <-> F (east-front)
# IMPORTANT: this assignment is IMPOSED by us. The data carries no per-camera
# aisle label — each overhead corner cam physically sees several aisles. This
# dict is the operational definition of "the aisle a CCTV is responsible for".
# Cam poses live in warehouse_2x3_base.usda; aisle geometry in _common.py.
_CCTV_CORNER_TO_AISLE_2X3 = {
    "cctv_NW": "B",
    "cctv_NE": "C",
    "cctv_SW": "E",
    "cctv_SE": "F",
}


def _parse_blocked_aisles(scene_name: str) -> frozenset:
    """Sealed transit-aisle set for a counterfactual 2x3 scene, from its name.

    build_2x3.py emits counterfactual scenes as `<base>_cf<suffix>`, where the
    suffix is the sorted subset of sealed aisles (e.g. "CF") or "0" for the
    unsealed twin.
      "warehouse_2x3_0604_cfCF" -> frozenset({"C", "F"})
      "warehouse_2x3_0604_cf0"  -> frozenset()             (nothing sealed)
    """
    marker = "_cf"
    idx = scene_name.rfind(marker)
    if idx == -1:
        raise ValueError(
            f"cctv_blockade_signal: scene {scene_name!r} has no '_cf' suffix — "
            "this mode expects the warehouse_2x3 counterfactual dataset.")
    suffix = scene_name[idx + len(marker):]
    if suffix == "0":
        return frozenset()
    aisles = frozenset(suffix)
    unknown = aisles - _TRANSIT_AISLES_2X3
    if unknown:
        raise ValueError(
            f"cctv_blockade_signal: scene {scene_name!r} suffix {suffix!r} "
            f"contains non-transit-aisle letters {sorted(unknown)} "
            "(expected a subset of B/C/E/F).")
    return aisles


def _blocked_aisles_from_traj(traj_dir: Path) -> frozenset:
    """Read the sealed-aisle set for a recorded trajectory from its config.yaml.

    The composed run config written per episode (recorder.py) carries the scene
    under env.name (e.g. "warehouse_2x3_0604_cfCF"); we parse the _cf suffix.
    """
    cfg_path = traj_dir / "config.yaml"
    with open(cfg_path) as f:
        run_cfg = yaml.safe_load(f)
    try:
        scene_name = run_cfg["env"]["name"]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"cctv_blockade_signal: {cfg_path} has no env.name to read the "
            "sealed-aisle suffix from.") from e
    return _parse_blocked_aisles(scene_name)


def _blockade_shade(aisle: str, blocked_aisles: frozenset) -> int:
    """Grayscale shade (0-255) for a CCTV that owns `aisle`: 0 (black) if that
    aisle is sealed, else 255 (white)."""
    return 0 if aisle in blocked_aisles else 255


# ---------------------------------------------------------------------------
# Per-trajectory cache. Loaded once at __init__; __getitem__ reads from here.
# ---------------------------------------------------------------------------

class _TrajectoryCache:
    """Lazy-image, eager-pose cache for one trajectory dir.

    Pose data (xy, yaw, cmd_v, cmd_w) is small; load it all at __init__.
    Images stay on disk; PIL.Image.open() lazily reads them per __getitem__.
    OS page cache covers re-reads — no need for an explicit image LRU.
    """

    __slots__ = ("traj_dir", "graph", "frames_dir", "cctv_dirs", "has_cctv",
                 "positions", "yaws", "n_frames", "blocked_aisles")

    def __init__(self, traj_dir: Path, cctv_cam_names: tuple[str, ...] = ("cctv",),
                 blockade_signal: bool = False):
        self.traj_dir = Path(traj_dir)
        self.graph = GoalGraph.load(self.traj_dir / "frames.json")
        self.frames_dir = self.traj_dir / "frames"
        # One or more CCTV view dirs: the legacy single "cctv" dir for 1-cam runs,
        # or the 4 wall_course-facing array cams (cctv_wall_w/e, cctv_corner_sw/se)
        # for the 4-CCTV runs. Order is load-bearing — it sets the image-slot order.
        self.cctv_dirs = [self.traj_dir / c for c in cctv_cam_names]
        self.has_cctv = all(d.is_dir() and any(d.iterdir()) for d in self.cctv_dirs)
        # Vectorise pose arrays once — used by every __getitem__ on this traj.
        self.positions = np.array(
            [[n.xy[0], n.xy[1]] for n in self.graph.nodes], dtype=np.float64,
        )
        self.yaws = np.array([n.yaw_rad for n in self.graph.nodes], dtype=np.float64)
        self.n_frames = len(self.graph.nodes)
        # Counterfactual blockade-signal ablation: sealed transit-aisle set for
        # this scene, parsed from env.name in config.yaml. Only read when the
        # mode is on (otherwise leave None — the load path never touches it).
        if blockade_signal:
            self.blocked_aisles = _blocked_aisles_from_traj(self.traj_dir)
        else:
            self.blocked_aisles = None


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class IsaacSimDataset(Dataset):
    """Pose+image fine-tuning dataset over collected Isaac Sim trajectories.

    One sample = one (trajectory, observation-step `t`, goal-step `t+k`)
    triple. `k` is resampled per __getitem__ call (paper convention,
    gnm_dataset.py:177). Modality is also sampled per call from
    `modality_list`, with image-only dropped beyond
    `image_only_max_distance_steps` (paper convention, gnm_dataset.py:391).

    The index built by __init__ enumerates only the `t` (observation step);
    `k` and modality are random-per-call, so a single (traj, t) feeds many
    different supervisory signals across epochs.
    """

    DATASET_NAME = "isaac_sim"

    def __init__(
        self,
        # External dependencies (provided by trainer).
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
        modality_list: tuple[int, ...] = (4, 5, 6),
        # Optional per-modality sampling weights, parallel to modality_list.
        # None = uniform (legacy). e.g. lang run: list (7,4,5,6) w (0.8,...).
        modality_weights: tuple[float, ...] | None = None,
        # Language modality (id 7): manifest with per-scene target phrases +
        # the instruction pattern. Both None = id 7 unavailable (legacy).
        lang_manifest: str | None = None,
        lang_pattern: str | None = None,   # e.g. "move toward the {x}"
        first_step_only: bool = False,     # t=0-only probe view (see below)
        keep_steps_max: int | None = None, # keep only t <= this (see below)
        image_only_max_distance_steps: int = 20,
        min_goal_distance_steps: int = 1,
        max_goal_distance_steps: int = 20,
        metric_waypoint_spacing: float = 0.167,
        waypoint_spacing: int = 1,
        image_size: tuple[int, int] = (224, 224),
        random_hflip: bool = True,
        goals_per_obs: int = 1,
        emit_cctv: bool = True,
        # --- CCTV joint-training knobs (see thesis/methods_brief.md 6.5 'sampling
        #     knobs'; the 2026-06-01 CHANGES handoff was removed 2026-09-07) ---
        # perception_dropout = (p_both, p_ego_only, p_cctv_only); must sum to 1.0.
        #   (1,0,0) = both observation views always present (Run 1/3).
        perception_dropout: tuple[float, float, float] = (1.0, 0.0, 0.0),
        # Diagnostic ablation: feed an all-black onboard frame (block stays
        # PRESENT, so the guard above is untouched) to force reliance on CCTV.
        ego_blackout: bool = False,
        # use_cctv_goal: when True, feed a CCTV goal image (a 4th image slot) and
        #   couple goal-image views to perceived views. When False, only the ego
        #   goal image exists (3 image slots) — Runs 1/2.
        use_cctv_goal: bool = False,
        # --- 4-CCTV experiment knobs (default = legacy 1-cam behaviour) ---
        # cctv_cam_names: per-view dir names recorded under each traj. ("cctv",)
        #   keeps the 1-cam runs unchanged; the 4 array-cam names give 4 cctv views.
        # no_goal_image: drop the ego-goal + cctv-goal IMAGE slots — goal is the
        #   pose vector only (slot layout becomes [ego_obs, cctv_0..N-1, pose]).
        cctv_cam_names: tuple[str, ...] = ("cctv",),
        no_goal_image: bool = False,
        # Per-camera CCTV dropout: each cctv view is independently masked
        # (block_present=0) with this probability. Ego is ALWAYS present. 0.0 =
        # off (use the coarse perception_dropout). Use 0.0 for val.
        cctv_cam_dropout: float = 0.0,
        # With this probability the goal is pinned to the trajectory's FINAL
        # frame (n-1) instead of a uniformly sampled offset. Rationale (cf
        # campaign, 2026-07-02): intermediate goals sit ON the oracle's detour
        # path and leak the turn direction; a final-frame goal forces the model
        # to explain the detour from perception (the forklift) alone. 0.0 =
        # legacy behaviour, and the branch consumes NO rng draw at 0.0 so
        # existing runs stay byte-reproducible.
        final_goal_prob: float = 0.0,
        # --- Counterfactual CCTV blockade-signal ablation (default off) ---
        # When True, every real CCTV frame is replaced by a solid black (the
        # aisle that camera owns is SEALED) or white (clear) image — a clean
        # per-aisle blockade bit. Requires the with_4cctv_2x3 corner cams and
        # counterfactual (_cf<suffix>) scene names; incompatible with
        # random_hflip (guarded below). See the module-level helpers.
        cctv_blockade_signal: bool = False,
        # --- Absolute-pose token (default off) ---
        # When True, each sample also carries "abs_pose": [x, y, cos(yaw), sin(yaw),
        # goal_x, goal_y] in the WORLD frame, xy normalised to [-1, 1] via
        # abs_pose_bounds = {"x": [lo, hi], "y": [lo, hi]} (warehouse extents, set
        # in the training YAML). Consumed by a separate ProprioProjector-style
        # encoder -> one extra token appended after the pose token (same presence
        # flag). Leak-free: cf variants share start/goal. Incompatible with
        # random_hflip (world coordinates don't mirror).
        abs_pose: bool = False,
        abs_pose_bounds: dict | None = None,
        # --- Per-camera responsibility crop (default off) ---
        # Crop each real CCTV frame to its camera's fixed responsibility box
        # (own aisle, entrance -> far rack-end) BEFORE the resize: virtual zoom
        # on far forklifts + cuts the through-the-D-gap view of the other row
        # (top edge = projection-derived separating line; work-log 2026-07-06).
        # {cam: (x0, y0, x1, y1)} in pixels of a cctv_crop_ref_size-square
        # frame; scaled if stored frames differ.
        cctv_crop: dict | None = None,
        cctv_crop_ref_size: int = 720,
        predict_stop_token: bool = True,
        seed: int = 42,
        # --- Fixed sampling (val sets; default off = paper behaviour) ---
        # When True, every stochastic per-sample choice in __getitem__ (goal
        # step k / final-goal coin, modality, perception dropout) is drawn from
        # a fresh RNG seeded on the sample's IDENTITY (seed, traj_idx, t)
        # instead of the shared running RNG. Result: the sample set is a pure
        # function of the data + seed — byte-identical on every pass — so val
        # loss differences reflect the weights, not sampling luck. Train keeps
        # the paper's per-call resampling (gnm_dataset.py:177), where one
        # (traj, t) feeding many (goal, modality) variants is the point.
        deterministic_sampling: bool = False,
        # --- Step-set view over the index (default off; DatasetManager) ---
        # {traj_dir_name: set of observation steps t} — after the normal index
        # build, keep ONLY samples whose (traj, t) is listed here; trajectories
        # not named contribute nothing. This is how counterfactual_dataset_2's
        # DatasetManager expresses "same trajectories, decision-window view"
        # without a subclass. A name not present under traj_root raises (a
        # silent no-match means the caller's mining and this dataset disagree
        # about the data). None = keep the full index (every existing config).
        allowed_steps_by_name: dict | None = None,
    ) -> None:
        if num_actions_chunk != NUM_ACTIONS_CHUNK:
            raise ValueError(
                f"num_actions_chunk={num_actions_chunk} must equal "
                f"prismatic.vla.constants.NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}; "
                "the pretrained action head's output dim is fixed."
            )

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        self.context_size = context_size
        self.num_actions_chunk = num_actions_chunk
        self.modality_list = list(modality_list)
        if modality_weights is not None:
            assert len(modality_weights) == len(self.modality_list), (
                f"modality_weights {modality_weights} must parallel "
                f"modality_list {self.modality_list}")
        self.modality_weights = (list(modality_weights)
                                 if modality_weights is not None else None)
        # scene name -> human-readable target phrase ("traffic cone"), joined
        # from the data-gen manifest; run-dir name maps to scene name by
        # stripping the "__pose__<ts>" suffix. Instructions deliberately live
        # in the manifest, NOT the recorder output (labels decoupled from data).
        self.lang_pattern = lang_pattern
        self._phrase_by_scene: dict[str, str] = {}
        if lang_manifest is not None:
            _man = json.loads(Path(lang_manifest).read_text())
            self._phrase_by_scene = {m["name"]: m["target_phrase"]
                                     for m in _man["scenes"]}
        if 7 in self.modality_list:
            assert self.lang_pattern and self._phrase_by_scene, (
                "modality 7 requires lang_manifest + lang_pattern")
        self.image_only_max_distance_steps = image_only_max_distance_steps
        self.min_goal_distance_steps = min_goal_distance_steps
        self.max_goal_distance_steps = max_goal_distance_steps
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.waypoint_spacing = waypoint_spacing
        self.image_size = tuple(image_size)
        self.random_hflip = random_hflip
        self.goals_per_obs = goals_per_obs
        self.emit_cctv = emit_cctv
        self.perception_dropout = perception_dropout
        self.ego_blackout = bool(ego_blackout)
        self.use_cctv_goal = use_cctv_goal
        self.cctv_cam_names = tuple(cctv_cam_names)
        self.no_goal_image = no_goal_image
        self.cctv_cam_dropout = float(cctv_cam_dropout)
        self.final_goal_prob = float(final_goal_prob)
        if not (0.0 <= self.final_goal_prob <= 1.0):
            raise ValueError(f"final_goal_prob={final_goal_prob} must be in [0, 1]")
        self.predict_stop_token = predict_stop_token
        # Guard: cctv goal images + MULTIPLE cctv cams (with ego goal) is not
        # supported — the dataset emits cctv obs as a flat list but no per-view
        # cctv-goal list, and the eval policy interleaves obs/goal per view, so
        # the slot orders would not match. Single-cctv cctv-goal is fine.
        if self.use_cctv_goal and not self.no_goal_image and len(self.cctv_cam_names) > 1:
            raise NotImplementedError(
                "use_cctv_goal=True with multiple CCTV cameras is not supported "
                "(slot order not aligned with the eval policy). Use a single cctv "
                "cam, or set use_cctv_goal=False.")
        # The onboard (ego) observation MUST always be present (never drop the
        # onboard view, per Lukas). perception_dropout=(p_both, p_ego_only,
        # p_cctv_only); only the cctv_only branch drops the ego, so require
        # p_cctv_only == 0. Guards both train and val dropout tuples.
        if self.perception_dropout[2] != 0.0:
            raise ValueError(
                f"perception_dropout={self.perception_dropout}: p_cctv_only must be 0 "
                f"— the onboard ego observation must never be dropped (train + inference).")

        # --- Counterfactual CCTV blockade-signal ablation setup ---
        self.cctv_blockade_signal = bool(cctv_blockade_signal)
        self._cctv_aisle: dict[str, str] = {}
        if self.cctv_blockade_signal:
            # hflip mirrors the world (ego image + actions + goal) but the
            # black/white signal is computed for the UNmirrored aisle layout, and
            # a solid image is flip-invariant — so the signal would silently stop
            # tracking the mirror in ~50% of samples. Forbid the combination.
            if self.random_hflip:
                raise ValueError(
                    "cctv_blockade_signal=True is incompatible with "
                    "random_hflip=True (the black/white per-aisle signal cannot "
                    "follow the mirror). Set random_hflip=False — the cf configs "
                    "already do.")
            # Map each configured CCTV cam -> the transit aisle it owns. Error
            # early if a cam has no mapping (e.g. flag set for a non-2x3 cam set).
            for cam in self.cctv_cam_names:
                if cam not in _CCTV_CORNER_TO_AISLE_2X3:
                    raise ValueError(
                        f"cctv_blockade_signal: no aisle mapping for CCTV cam "
                        f"{cam!r} (known: {sorted(_CCTV_CORNER_TO_AISLE_2X3)}). "
                        "This mode targets the with_4cctv_2x3 corner cameras.")
                self._cctv_aisle[cam] = _CCTV_CORNER_TO_AISLE_2X3[cam]

        # --- Counterfactual onboard-bundle swap (opt-in; 2026-07-12) ---
        # When onboard_swap_map is set (by counterfactual_dataset.
        # build_onboard_swap_map after init), an in-window sample (ti, t) may
        # read its ONBOARD bundle — ego obs image, ego goal image, goal_pose —
        # from a cf SIBLING trajectory at the same frame index, while the CCTV
        # frames and the action labels stay the sample's own. Purpose: inside
        # decision windows the onboard view differs from the sibling's only by
        # the pre-divergence drift (the copycat tilt), so serving either twin's
        # bundle with the sample's own label makes the onboard channel
        # non-predictive there and forces the diverging labels to be explained
        # by the CCTV stream. Map: {(traj_idx, t): [sibling traj_idx, ...]};
        # onboard_swap_p gates a UNIFORM draw over the family (own included):
        # p=1.0 fully randomizes the bundle source on eligible draws.
        self.onboard_swap_map: dict | None = None
        self.onboard_swap_p: float = 0.0
        # EXTRA-SAMPLE variant (Lukas 2026-07-12, replaces in-place swapping
        # for the diagonal-invisible fine-tune): honest samples stay untouched;
        # each eligible (traj, t) additionally appears ONCE as a forced-swap
        # duplicate [(traj_idx, t, sibling_idx), ...] appended after
        # samples_index. Deterministic — both versions seen every epoch.
        self.onboard_swap_extra: list = []
        # NB: incompatible with abs_pose (world-frame vec would mix own pose
        # with the sibling's goal) — enforced in the builder functions, the
        # only places these are attached.

        # --- Absolute-pose token setup ---
        self.abs_pose = bool(abs_pose)
        self.abs_pose_bounds = abs_pose_bounds
        if self.abs_pose:
            # World coordinates cannot follow an image mirror (same class of
            # problem as the blockade signal above) — forbid the combination.
            if self.random_hflip:
                raise ValueError(
                    "abs_pose=True is incompatible with random_hflip=True "
                    "(absolute world coordinates cannot be mirrored).")
            if (not isinstance(abs_pose_bounds, dict)
                    or "x" not in abs_pose_bounds or "y" not in abs_pose_bounds):
                raise ValueError(
                    "abs_pose=True requires abs_pose_bounds={'x': [lo, hi], "
                    f"'y': [lo, hi]}} (warehouse extents); got {abs_pose_bounds!r}")

        # --- Per-camera responsibility crop setup ---
        self.cctv_crop = None
        self.cctv_crop_ref_size = int(cctv_crop_ref_size)
        if cctv_crop is not None:
            if self.cctv_blockade_signal:
                raise ValueError(
                    "cctv_crop is incompatible with cctv_blockade_signal=True "
                    "(solid signal frames have no geometry to crop).")
            # Same class of problem as the blockade signal / abs_pose above: a
            # mirrored frame no longer matches its camera's fixed crop box, and
            # a mirrored NE view shows what looks like NW's aisle.
            if self.random_hflip:
                raise ValueError(
                    "cctv_crop is incompatible with random_hflip=True (a mirrored "
                    "frame no longer matches its camera's fixed crop box).")
            missing = [c for c in self.cctv_cam_names if c not in cctv_crop]
            if missing:
                raise ValueError(f"cctv_crop: missing boxes for cams {missing}")
            for cam, box in cctv_crop.items():
                x0, y0, x1, y1 = box
                if not (0 <= x0 < x1 <= self.cctv_crop_ref_size
                        and 0 <= y0 < y1 <= self.cctv_crop_ref_size):
                    raise ValueError(f"cctv_crop[{cam!r}]={box} not a valid box "
                                     f"within ref size {self.cctv_crop_ref_size}")
            self.cctv_crop = {c: tuple(int(v) for v in cctv_crop[c])
                              for c in self.cctv_cam_names}

        # Deterministic shuffle order across runs with same seed (assuming
        # the input dir contents are also stable — which list_successful_*
        # ensures by name-sorting).
        self._rng = random.Random(seed)
        # Kept for deterministic_sampling: __getitem__ re-seeds per sample from
        # (base_seed, traj_idx, t), so the draw survives index re-ordering.
        self._base_seed = seed
        self.deterministic_sampling = bool(deterministic_sampling)

        # ---- Build trajectory cache + sample index ----
        traj_dirs = list_successful_trajectories(Path(traj_root), max_n=max_trajectories)
        if not traj_dirs:
            raise RuntimeError(f"no successful trajectories under {traj_root}")
        self.trajectories: list[_TrajectoryCache] = [
            _TrajectoryCache(d, self.cctv_cam_names, self.cctv_blockade_signal)
            for d in traj_dirs
        ]

        # Sample index: list of (traj_idx, t) where t is the *observation*
        # step. Valid t requires:
        #   - At least 1 future step for action_chunk[0] (t + 1 ≤ n_frames - 1).
        #   - At least `min_goal_distance_steps` future steps for a valid goal.
        # We always need t+min_goal_distance_steps ≤ n_frames-1 → t ≤ n_frames-1-min_goal.
        # Lower bound on t: 0 (action chunk pads with last pose if trajectory
        # ends early; this matches gnm_dataset.py's behavior at line 243-244).
        # We don't restrict t by context_size because the pretrained image
        # backbone uses only current+goal (context_size is a vestigial param).
        self.samples_index: list[tuple[int, int]] = []
        for ti, traj in enumerate(self.trajectories):
            t_max = traj.n_frames - 1 - max(self.min_goal_distance_steps, self.num_actions_chunk)
            if t_max < 0:
                continue                                   # too short
            for t in range(0, t_max + 1):
                for _ in range(self.goals_per_obs):
                    self.samples_index.append((ti, t))

        # --- First-frame-only probe view (2026-07-21, Lukas) ---
        # Keep ONLY t=0 samples: the cold-start commitment probe. Tests whether
        # the optimizer can wire language->initiation when no later-step motion
        # shortcut competes for gradient (the lang80 run left t=0 at the neutral
        # prior while fitting everything else — see session log).
        if first_step_only:
            self.samples_index = [(ti, t) for (ti, t) in self.samples_index if t == 0]

        # --- Early-frame view (2026-07-22, Lukas) ---
        # Keep only t <= keep_steps_max. Generalises first_step_only (which is
        # keep_steps_max=0) to the first few frames. Motivation: only t=0 is a
        # genuinely counterfactual frame on langv2 (from t=1 the robot has begun
        # rotating, so its own yaw leaks the aisle choice — twin yaw spread
        # 0.00deg at t=0 -> 3.41 at t=1 -> 19.90 at t=2). Restricting to the
        # early frames stops the optimiser spending capacity on the ~98% of
        # samples that are solvable from proprioception alone.
        if keep_steps_max is not None:
            self.samples_index = [(ti, t) for (ti, t) in self.samples_index
                                  if t <= int(keep_steps_max)]

        # --- Optional step-set view (see the constructor comment) ---
        # Applied AFTER the normal build so every validity rule above (chunk
        # future, min-goal headroom) still holds for the kept samples.
        self.allowed_steps_by_name = allowed_steps_by_name
        if allowed_steps_by_name is not None:
            name_to_idx = {traj.traj_dir.name: ti
                           for ti, traj in enumerate(self.trajectories)}
            unknown = sorted(set(allowed_steps_by_name) - set(name_to_idx))
            if unknown:
                raise ValueError(
                    f"allowed_steps_by_name: {len(unknown)} name(s) not found "
                    f"under {traj_root} (first: {unknown[:3]}) — mining output "
                    "and traj_root disagree")
            allowed_by_idx = {name_to_idx[name]: steps
                              for name, steps in allowed_steps_by_name.items()}
            self.samples_index = [
                (ti, t) for (ti, t) in self.samples_index
                if ti in allowed_by_idx and t in allowed_by_idx[ti]
            ]
            if not self.samples_index:
                raise RuntimeError(
                    "allowed_steps_by_name filtered the index to 0 samples")

        n_samples = len(self.samples_index)
        n_trajs = len(self.trajectories)
        print(f"[IsaacSimDataset] {n_trajs} trajectories → {n_samples} samples "
              f"(avg {n_samples / n_trajs:.1f} per trajectory)")

    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        # onboard_swap_extra (2026-07-12): forced-swap DUPLICATES of eligible
        # decision frames appended after the honest samples — see
        # counterfactual_dataset.build_onboard_swap_extra.
        return len(self.samples_index) + len(self.onboard_swap_extra)

    # ----------------------------------------------------------------------
    def chunk_theta_deg(self) -> np.ndarray:
        """|chord angle| (deg) of the GT action chunk for EVERY sample index.

        Pure numpy over the cached positions/yaws (no image loads, no rng) —
        cheap enough to run once at DataLoader construction. Consumed by the
        curvature-balanced WeightedRandomSampler in train_omnivla_sim_cctv_v2
        (data.curvature_balance in the train YAML).
        """
        out = np.empty(len(self.samples_index), dtype=np.float64)
        for i, (traj_idx, t) in enumerate(self.samples_index):
            traj = self.trajectories[traj_idx]
            out[i] = _chunk_abs_theta_deg(
                traj.positions, traj.yaws, t, self.num_actions_chunk)
        return out

    # ----------------------------------------------------------------------
    def _reseed_for_sample(self, traj_idx: int, t: int) -> None:
        """deterministic_sampling: replace the running RNG with one seeded on
        this sample's identity, so every draw in __getitem__ (goal step,
        modality, dropout) is identical on every pass. Seeded on (traj_idx, t)
        — not the positional idx — so the draw survives re-ordering or
        re-filtering of samples_index. str seed: a tuple would raise TypeError
        under Python 3.11+. No-op when the flag is off (train datasets)."""
        if self.deterministic_sampling:
            self._rng = random.Random(f"{self._base_seed}:{traj_idx}:{t}")

    def _sample_goal_t(self, t: int, n: int) -> int:
        """Draw the goal step for a sample at frame t of an n-frame trajectory.

        Goal-step range: [t+min_d, t+max_d] clamped to [t+min_d, n-1].
        final_goal_prob branch: pin the goal to the trajectory END (n-1) with
        probability p, bypassing max_goal_distance_steps by design (the point
        is the full remaining horizon; the index build guarantees
        n-1-t >= min_goal_distance_steps). The `> 0.0` guard means p=0 configs
        consume NO extra rng draw -> legacy runs stay byte-reproducible.
        NOTE: with p>0 the extra draw shifts the rng stream, which breaks the
        _replay_goal_t() coupling in the inspect_* debug scripts (logged in
        the project's suspicion log 2026-07-02).
        """
        if self.final_goal_prob > 0.0 and self._rng.random() < self.final_goal_prob:
            return n - 1
        max_k_offset = min(self.max_goal_distance_steps, n - 1 - t)
        return t + self._rng.randint(self.min_goal_distance_steps, max_k_offset)

    # ----------------------------------------------------------------------
    def _cctv_crop_box(self, cam: str, frame_size) -> tuple[int, int, int, int]:
        """Scale `cam`'s ref-frame crop box to this frame's actual pixel size.
        e.g. box (200, 254, 540, 648) @ ref 720 on a 360x360 frame -> (100, 127, 270, 324)."""
        w, h = frame_size
        sx = w / self.cctv_crop_ref_size
        sy = h / self.cctv_crop_ref_size
        x0, y0, x1, y1 = self.cctv_crop[cam]
        return (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))

    def __getitem__(self, idx: int) -> dict:
        # Forced-swap EXTRA samples live past the honest index range.
        forced_src_ti = None
        if idx >= len(self.samples_index):
            traj_idx, t, forced_src_ti = self.onboard_swap_extra[
                idx - len(self.samples_index)]
        else:
            traj_idx, t = self.samples_index[idx]
        traj = self.trajectories[traj_idx]
        n = traj.n_frames

        self._reseed_for_sample(traj_idx, t)

        # --- Choose goal step `k` and modality ---
        goal_t = self._sample_goal_t(t, n)
        k_offset = goal_t - t
        # Distance in *steps* — matches the units gnm_dataset.py uses for
        # the image-only threshold (see iter1_image_pose.yaml comments).
        distance_steps = k_offset // self.waypoint_spacing

        if distance_steps <= self.image_only_max_distance_steps:
            allowed = self.modality_list
        else:
            # Drop image-only at long goal distance — paper convention,
            # gnm_dataset.py:394. Image goals beyond ~3 m are OOD for the
            # vision backbone.
            # Filter by explicit ID (= 6 for image-only) rather than slice
            # index, so the behaviour is robust to the order of modality_list
            # in the YAML.
            allowed = [m for m in self.modality_list if m != 6]
            if not allowed:
                allowed = self.modality_list   # nothing to drop
        if self.modality_weights is None:
            modality_id = self._rng.choice(allowed)
        else:
            # Weighted draw restricted to `allowed` (weights renormalise
            # implicitly in random.choices).
            w = [self.modality_weights[self.modality_list.index(m)] for m in allowed]
            modality_id = self._rng.choices(allowed, weights=w, k=1)[0]

        # --- Counterfactual onboard-bundle swap source (see __init__ note) ---
        # src supplies the ONBOARD bundle (ego obs image, ego goal image,
        # goal_pose); traj keeps supplying actions + CCTV. Swaps happen ONLY on
        # FINAL-goal draws: the twins share the physical goal, so the sibling's
        # own final frame is the same goal location and the swapped bundle
        # stays task-coherent. An intermediate goal point lies on the sibling's
        # BRANCH and would contradict the own-branch label — those draws keep
        # their own bundle (they leak the route through the goal anyway, so
        # they carry no counterfactual pressure to protect). The draw comes
        # after _reseed_for_sample => pinned under deterministic_sampling;
        # guarded so runs without the feature consume no extra rng draws.
        src = traj
        src_t, src_goal_t = t, goal_t
        if forced_src_ti is not None:
            # Extra-sample path: swap is deterministic (no coin, no map lookup).
            # build_onboard_swap_extra guarantees final_goal_prob == 1.0, so
            # goal_t == n-1 here and the sibling's own arrival frame is the
            # same physical goal.
            src = self.trajectories[forced_src_ti]
            src_t = min(t, src.n_frames - 1)
            src_goal_t = src.n_frames - 1
        elif self.onboard_swap_map is not None and goal_t == n - 1:
            siblings = self.onboard_swap_map.get((traj_idx, t))
            if siblings and self._rng.random() < self.onboard_swap_p:
                # p = probability of serving a SIBLING's bundle; p=0.5 => 50/50
                # own/sibling for twin pairs = bundle source uniform over the
                # family = onboard content carries zero label information in
                # windows. Do NOT set p near 1.0: own pairings would vanish and
                # bundle/label become ANTI-correlated — an equally learnable
                # shortcut, just inverted.
                src = self.trajectories[self._rng.choice(siblings)]
                src_t = min(t, src.n_frames - 1)
                src_goal_t = src.n_frames - 1        # sibling's OWN arrival frame

        # --- Sample which OBSERVATION views are present (perception dropout) ---
        # At least one view is always present. (1,0,0) default => both on.
        p_both, p_ego, _p_cctv = self.perception_dropout
        r_perc = self._rng.random()
        if r_perc < p_both:
            ego_present, cctv_present = True, True
        elif r_perc < p_both + p_ego:
            ego_present, cctv_present = True, False
        else:
            ego_present, cctv_present = False, True

        # Per-camera CCTV dropout: each cctv view independently kept with prob
        # (1 - cctv_cam_dropout). Ego stays as the coarse flag (always True with
        # the [1,0,0] default + onboard-always guard). cctv_view_present is a
        # per-view list used in the multi-cctv block layout; None => fall back to
        # the single coarse cctv_present for every view (legacy behaviour).
        if self.cctv_cam_dropout > 0.0:
            cctv_view_present = [self._rng.random() >= self.cctv_cam_dropout
                                 for _ in range(len(traj.cctv_dirs))]
        else:
            cctv_view_present = None

        # --- Build action chunk: N future poses in local frame at step t ---
        N = self.num_actions_chunk
        end_idx = min(t + N + 1, n)
        positions = traj.positions[t:end_idx].copy()       # (n_avail, 2)
        yaws = traj.yaws[t:end_idx].copy()                 # (n_avail,)

        # Pad with terminal pose if trajectory ends before t+N+1.
        # Matches gnm_dataset.py:242-244.
        if positions.shape[0] < N + 1:
            pad_n = N + 1 - positions.shape[0]
            positions = np.concatenate(
                [positions, np.tile(positions[-1:], (pad_n, 1))], axis=0)
            yaws = np.concatenate(
                [yaws, np.tile(yaws[-1:], pad_n)], axis=0)

        waypoints_local = _to_local_xy(positions, positions[0], yaws[0])   # (N+1, 2)
        yaws_local = yaws[1:] - yaws[0]                                     # (N,)

        # Normalise positions: actions[:,:2] /= metric_waypoint_spacing * waypoint_spacing.
        # Matches gnm_dataset.py:257.
        actions_dxy = waypoints_local[1:] / (self.metric_waypoint_spacing * self.waypoint_spacing)
        actions_3d = np.concatenate([actions_dxy, yaws_local[:, None]], axis=-1)  # (N, 3)
        actions = _actions_to_sincos(actions_3d)                                  # (N, 4)

        # --- Goal pose (relative to t, in local frame, normalised) ---
        # Onboard-swap: the whole egocentric bundle comes from src — src's
        # goal frame relative to src's OWN pose at t — so image and pose stay
        # mutually consistent (src == traj in the default path).
        goal_xy_world = src.positions[src_goal_t][None, :]                # (1, 2)
        goal_xy_local = _to_local_xy(goal_xy_world, src.positions[src_t], src.yaws[src_t])[0]
        goal_xy_local = goal_xy_local / (self.metric_waypoint_spacing * self.waypoint_spacing)
        # DE-LEAK (2026-06-17): see isaac_sim_dataset.py — the goal-step recorded
        # heading encodes the detour side and handed it to the policy at t≈0. Use
        # the trajectory START heading so the goal pose carries no which-way info.
        goal_yaw_local = src.yaws[0] - src.yaws[src_t]
        goal_pose = np.array([
            goal_xy_local[0], goal_xy_local[1],
            math.cos(goal_yaw_local), math.sin(goal_yaw_local),
        ], dtype=np.float32)

        # --- Absolute-pose vector (world frame; opt-in, see __init__) ---
        # [x, y, cos(yaw), sin(yaw), goal_x, goal_y]; xy scaled to [-1, 1] by the
        # warehouse extents. positions[0]/yaws[0] = current step t in WORLD frame;
        # goal_xy_world = position at the sampled goal index (same frame).
        abs_pose_vec = None
        if self.abs_pose:
            xb, yb = self.abs_pose_bounds["x"], self.abs_pose_bounds["y"]
            nx = lambda v: 2.0 * (v - xb[0]) / (xb[1] - xb[0]) - 1.0
            ny = lambda v: 2.0 * (v - yb[0]) / (yb[1] - yb[0]) - 1.0
            abs_pose_vec = np.array([
                nx(positions[0, 0]), ny(positions[0, 1]),
                math.cos(yaws[0]), math.sin(yaws[0]),
                nx(goal_xy_world[0, 0]), ny(goal_xy_world[0, 1]),
            ], dtype=np.float32)

        # --- Images: load + resize (ego views from src — onboard swap) ---
        cur_pil = Image.open(src.frames_dir / src.graph.nodes[src_t].image_filename).convert("RGB")
        goal_pil = Image.open(src.frames_dir / src.graph.nodes[src_goal_t].image_filename).convert("RGB")
        cur_pil_resized = cur_pil.resize(self.image_size, Image.BILINEAR)
        goal_pil_resized = goal_pil.resize(self.image_size, Image.BILINEAR)

        # Load CCTV BEFORE the flip block so all visual inputs share the
        # same flip decision (see correctness note below).
        if self.emit_cctv and traj.has_cctv:
            if self.cctv_blockade_signal:
                # Counterfactual ablation: replace each real CCTV frame with a
                # solid black (this cam's aisle is sealed) or white (clear)
                # image — a clean per-aisle blockade bit, stripped of all other
                # content. Index-aligned to self.cctv_cam_names == traj.cctv_dirs
                # order, so it lines up with cctv_stack / block_present downstream.
                cctv_pils = []
                for cam in self.cctv_cam_names:
                    shade = _blockade_shade(self._cctv_aisle[cam], traj.blocked_aisles)
                    cctv_pils.append(Image.new("RGB", self.image_size, (shade, shade, shade)))
            else:
                # 1 (legacy) or N (4-CCTV) frames; traj.cctv_dirs is index-aligned
                # to self.cctv_cam_names (see _TrajectoryCache), so zipping keeps
                # each crop box with its own camera's frame.
                cctv_pils = []
                for cam, d in zip(self.cctv_cam_names, traj.cctv_dirs):
                    img = Image.open(d / traj.graph.nodes[t].image_filename).convert("RGB")
                    if self.cctv_crop is not None:
                        img = img.crop(self._cctv_crop_box(cam, img.size))
                    cctv_pils.append(img.resize(self.image_size, Image.BILINEAR))
        else:
            cctv_pils = None

        # CCTV goal frame (at the goal step). Legacy 1-cam path only; 4-CCTV runs
        # set no_goal_image/use_cctv_goal=False so there is no cctv-goal slot.
        if self.emit_cctv and traj.has_cctv and self.use_cctv_goal:
            cctv_goal_path = traj.cctv_dirs[0] / traj.graph.nodes[goal_t].image_filename
            cctv_goal_pil = Image.open(cctv_goal_path).convert("RGB").resize(self.image_size, Image.BILINEAR)
        else:
            cctv_goal_pil = None

        # Random horizontal flip. Matches gnm_dataset.py:369-385 — negates
        # the lateral action and sin(dyaw) components, plus the goal_pose
        # lateral + sin components. Doubles effective left/right symmetry.
        #
        # Correctness note: when CCTV is enabled, the CCTV image MUST also
        # be flipped to stay consistent with ego + actions. Otherwise the
        # model gets contradictory signals (ego shows mirrored world, CCTV
        # shows actual world, actions in robot frame are mirrored). Flipping
        # CCTV is philosophically a "physically impossible" view (no real
        # CCTV pose would produce that exact image), but the model only
        # needs to extract situational info from it — it doesn't need to
        # know the camera pose. All visual inputs in one sample must
        # depict the same (possibly-mirrored) world.
        if self.random_hflip and self._rng.random() > 0.5:
            cur_pil_resized = cur_pil_resized.transpose(Image.FLIP_LEFT_RIGHT)
            goal_pil_resized = goal_pil_resized.transpose(Image.FLIP_LEFT_RIGHT)
            if cctv_pils is not None:
                cctv_pils = [p.transpose(Image.FLIP_LEFT_RIGHT) for p in cctv_pils]
            if cctv_goal_pil is not None:
                cctv_goal_pil = cctv_goal_pil.transpose(Image.FLIP_LEFT_RIGHT)
            actions[:, 1] = -actions[:, 1]
            actions[:, 3] = -actions[:, 3]
            goal_pose[1] = -goal_pose[1]
            goal_pose[3] = -goal_pose[3]

        if self.ego_blackout:
            cur_pil_resized = Image.new("RGB", cur_pil_resized.size, (0, 0, 0))
        pixel_values = self.image_transform(cur_pil_resized)
        pixel_values_goal = self.image_transform(goal_pil_resized)

        if cctv_pils is not None:
            cctv_stack = [self.image_transform(p) for p in cctv_pils]     # len = #cctv views
        else:
            cctv_stack = [torch.zeros_like(pixel_values) for _ in traj.cctv_dirs]
        cctv_pixel_values = cctv_stack[0]                                 # legacy single-cctv slot

        if cctv_goal_pil is not None:
            cctv_pixel_values_goal = self.image_transform(cctv_goal_pil)
        else:
            cctv_pixel_values_goal = torch.zeros_like(pixel_values)

        # --- Couple goal-image views to perceived views, then enforce validity ---
        # modality_id (4/5/6) says WHICH goal info exists (pose? image?); the
        # perception flags say WHICH views the image goal is shown in.
        modality_pose = modality_id in (4, 5)
        modality_img  = modality_id in (5, 6)
        ego_goal_present  = ego_present  and modality_img
        cctv_goal_present = cctv_present and modality_img and self.use_cctv_goal

        # Validity: a sample must carry at least one goal. If this (perception,
        # modality, use_cctv_goal) combo yields none (e.g. cctv-only obs +
        # image-goal task + cctv-goal disabled), fall back to a coordinate/pose
        # goal — Lukas-confirmed rule ("just use coordinates for that").
        # Language-only (id 7) carries its goal in the INSTRUCTION — no pose or
        # image goal is not a defect. Without this clause the fallback silently
        # rewrote every id-7 draw to id 4 (caught by the trainer's modality
        # sanity check, 2026-07-21 — the 80% language mix arrived as 94% pose).
        modality_lang = (modality_id == 7) and bool(self.lang_pattern)
        if not (modality_pose or ego_goal_present or cctv_goal_present or modality_lang):
            modality_pose = True
            modality_id = 4            # pose-only; also the scalar the action head sees
            ego_goal_present = False
            cctv_goal_present = False

        # Per-view cctv presence: per-camera dropout list if enabled, else the
        # single coarse flag broadcast to every view.
        cctv_flags = (cctv_view_present if cctv_view_present is not None
                      else [cctv_present] * len(cctv_stack))

        # Per-block presence flags, in the EXACT order the collate concatenates
        # images (ORDER IS LOAD-BEARING — see potential_bugs.md 2026-06-01).
        # 3-image run (use_cctv_goal=False): [ego_obs, ego_goal, cctv_obs, pose]
        # 4-image run (use_cctv_goal=True):  [ego_obs, ego_goal, cctv_obs, cctv_goal, pose]
        if self.no_goal_image:
            # 4-CCTV: [ego_obs, cctv_0..N-1, pose] — pose-only goal, no goal-image slots.
            block_flags = [ego_present] + cctv_flags + [modality_pose]
        elif self.emit_cctv and len(cctv_stack) > 1:
            # ego goal + MULTI-cctv obs, no cctv goal:
            #   [ego_obs, ego_goal, cctv_0..N-1, pose]
            # ego_obs+ego_goal -> ego encoder (num_ego_images=2); the N cctv obs
            # -> shared cctv encoder. use_cctv_goal with multi-cctv is unsupported
            # (guarded in __init__): the per-view obs/goal interleave order isn't
            # aligned with the eval policy yet.
            block_flags = ([ego_present, ego_goal_present]
                           + cctv_flags
                           + [modality_pose])
        else:
            block_flags = [ego_present, ego_goal_present]
            if self.emit_cctv:
                block_flags.append(cctv_flags[0])
                if self.use_cctv_goal:
                    block_flags.append(cctv_goal_present)
            block_flags.append(modality_pose)
        block_present = np.array(block_flags, dtype=np.float32)

        # --- LLM prompt (mirror gnm_dataset.py:336-366) ---
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        current_action = actions_torch[0]
        future_actions = actions_torch[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        prompt_builder = self.prompt_builder_fn("openvla")
        if modality_id in (7, 8) and self.lang_pattern:
            # LeLaN training format exactly (lelan_dataset.py:433-440):
            # "move toward {X}" wrapped in the fixed question, lowercased.
            scene = traj.traj_dir.name.split("__pose__")[0]
            if scene.startswith("rec_"):  # recovery dirs = rec_<scene>__pose__<ts>_t<k>;
                scene = scene[4:]  # same scene family -> same instruction/phrase
            phrase = self._phrase_by_scene[scene]   # KeyError = wrong manifest, fail loud
            lang = self.lang_pattern.format(x=phrase).lower()
            prompt_builder.add_turn(
                "human", f"What action should the robot take to {lang}?")
        else:
            prompt_builder.add_turn("human", "No language instruction")
        prompt_builder.add_turn("gpt", action_chunk_string)
        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True,
        ).input_ids
        labels = list(input_ids)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        # Mask everything before the action tokens with IGNORE_INDEX so the
        # loss is computed only on action positions. The last (action_chunk_len+1)
        # positions are: action_chunk_len action tokens + 1 EOS.
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # --- Collator-required dummy fields (MBRA pathway we don't use) ---
        # Shapes are minimal-but-consistent across the batch so collator.stack works.
        # cur_image is normally (3*context_size, 64, 85) for MBRA's image-hist input.
        cur_image_dummy = np.zeros((3 * self.context_size, 64, 85), dtype=np.float32)
        # goal_image_8 normally a single 3xHxW frame for MBRA's goal input.
        goal_image_8_dummy = np.zeros((3, 64, 85), dtype=np.float32)
        # temp_dist is a scalar used elsewhere.
        temp_dist_dummy = np.array(0.0, dtype=np.float32)
        # action_select_mask = 1.0 means "use raw action" (not MBRA-synthetic).
        action_select_mask = np.array(1.0, dtype=np.float32)
        # obj_pose_norm = zeros (no LeLaN object-reaching loss).
        obj_pose_norm = np.array([0.0, 0.0], dtype=np.float32)

        sample = dict(
            pixel_values=pixel_values,
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
            # Per-block attention-presence vector (length n_img + 1).
            block_present=block_present,
        )
        if abs_pose_vec is not None:
            sample["abs_pose"] = abs_pose_vec
        if self.no_goal_image:
            # 4-CCTV: pose-only goal; the N cctv views become slots after the ego
            # obs. The collator concatenates [ego_obs, *cctv_pixel_values_list].
            sample["cctv_pixel_values_list"] = cctv_stack
        elif self.emit_cctv and len(cctv_stack) > 1:
            # ego goal + MULTI-cctv obs (no cctv goal). Collator concatenates
            # [ego_obs, ego_goal, *cctv_pixel_values_list].
            sample["pixel_values_goal"] = pixel_values_goal
            sample["gimg_PIL"] = goal_pil_resized
            sample["cctv_pixel_values_list"] = cctv_stack
        else:
            # Legacy single-cctv path: ego goal image (+ optional cctv goal).
            # Emitting cctv goal conditionally is what makes the collate build a
            # 3- vs 4-image tensor.
            sample["pixel_values_goal"] = pixel_values_goal
            sample["gimg_PIL"] = goal_pil_resized
            if self.emit_cctv:
                sample["cctv_pixel_values"] = cctv_pixel_values
            if self.use_cctv_goal:
                sample["cctv_pixel_values_goal"] = cctv_pixel_values_goal
        return sample


# ---------------------------------------------------------------------------
# CLI sanity-check entrypoint: shows index stats + a sample's shapes.
# Run with: `python -m prismatic.vla.datasets.isaac_sim_dataset <traj_root>`
# Requires action_tokenizer / base_tokenizer / etc. to be available, so
# this CLI is a stub — real smoke test lives in isaac_sim/train/smoke_*.py.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print("Use isaac_sim/train/smoke_test_dataset.py for the real smoke test.")
    print(f"Args received: {sys.argv}")
