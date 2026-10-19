
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


VALID_MODALITY_TOKENS = frozenset({"sat", "lang", "pose", "image"})

MODALITY_TO_ID: dict[frozenset[str], int] = {
    frozenset({"sat"}):                  0,
    frozenset({"sat", "pose"}):          1,
    frozenset({"sat", "image"}):         2,
    frozenset({"sat", "pose", "image"}): 3,
    frozenset({"pose"}):                 4,
    frozenset({"pose", "image"}):        5,
    frozenset({"image"}):                6,
    frozenset({"lang"}):                 7,
    frozenset({"lang", "pose"}):         8,
}


@dataclass
class Task:
    name: str
    modality: frozenset[str]

    # goal_xy is optional and currently not read by the sim loop: the goal is
    # the env's effective (standoff) goal pose, which accounts for the goal
    # prop's AABB + parking buffer, or the raw goal point when no prop is
    # placed.
    goal_xy: tuple[float, float] | None = None

    # Default mirrors the LeLaN "move toward X" training pattern (as in
    # upstream OmniVLA). The full prompt seen by the tokenizer is
    # "What action should the robot take to {lang}?". Override per task / per
    # run (lang_instruction) to match the goal prop placed by the env.
    lang_instruction: str = "move toward the goal"

    # Currently unread — the runtime goal heading comes from the env's
    # effective goal pose. Setting it has no effect.
    goal_heading_deg: float = 0.0

    # For "image" modality, set EXACTLY ONE of these:
    #   goal_image_path  — single goal image (OmniVLA's ≤3 m case)
    #   goal_graph_path  — JSON manifest of a topological graph (OmniVLA's
    #                      >3 m case). Recorded by IsaacSimEnv.set_task at
    #                      run start.
    goal_image_path: str | None = None
    goal_graph_path: str | None = None

    # Graph-recording knobs consumed by IsaacSimEnv._record_image_graph.
    goal_graph_node_spacing_m: float = None
    goal_graph_node_spacing_freq: float = 1 
    goal_graph_camera_name: str = "front_cam"  # Hydra dict key, not the camera's internal .name
    goal_graph_sampling_mode: str = "static_arclength"

    max_steps: int = 200
    # Tight — the goal we measure against is the *standoff* position, which
    # already parks the robot in open space in front of the prop. 0.3 m is
    # "image-localisation slop".
    success_dist: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.modality, frozenset):
            self.modality = frozenset(self.modality)

        unknown = self.modality - VALID_MODALITY_TOKENS
        if unknown:
            raise ValueError(
                f"Task {self.name!r}: unknown modality tokens {sorted(unknown)}. "
                f"Valid tokens: {sorted(VALID_MODALITY_TOKENS)}."
            )

        if self.modality not in MODALITY_TO_ID:
            valid = ", ".join(
                "{" + ",".join(sorted(combo)) + "}" for combo in MODALITY_TO_ID
            )
            raise ValueError(
                f"Task {self.name!r}: modality combo {set(self.modality)} is "
                f"not one of the 9 the model supports. Pick one of: {valid}."
            )

        if "sat" in self.modality:
            raise ValueError(
                f"Task {self.name!r}: 'sat' modality is not yet plumbed "
                f"through the sim bridge — no satellite imagery source. "
                f"Drop 'sat' or wire it up first."
            )

        if "image" in self.modality:
            has_image = self.goal_image_path is not None
            has_graph = self.goal_graph_path is not None
            if not (has_image or has_graph):
                raise ValueError(
                    f"Task {self.name!r}: modality includes 'image' but "
                    f"neither goal_image_path nor goal_graph_path is set."
                )
            if has_image and has_graph:
                raise ValueError(
                    f"Task {self.name!r}: set exactly one of goal_image_path "
                    f"(single image, ≤3 m) or goal_graph_path (topological, >3 m)."
                )

        # parents[1] = repo root. task.py lives at <repo>/isaac_sim/task.py,
        # so one hop up the parent chain lands on the repo root — the anchor
        # for YAML paths that start with "isaac_sim/".
        repo_root = Path(__file__).resolve().parents[1]
        for attr in ("goal_image_path", "goal_graph_path"):
            val = getattr(self, attr)
            if val is None:
                continue
            p = Path(val)
            if not p.is_absolute():
                p = (repo_root / val).resolve()
                setattr(self, attr, str(p))
            # goal_graph_path is created by the env at runtime, so
            # existence is NOT a precondition. goal_image_path must exist upfront.
            if attr == "goal_image_path" and not Path(getattr(self, attr)).exists():
                raise FileNotFoundError(
                    f"Task {self.name!r}: {attr} "
                    f"{getattr(self, attr)!r} does not exist."
                )

    @property
    def modality_id(self) -> int:
        return MODALITY_TO_ID[self.modality]

    def to_modality_flags(self) -> dict[str, bool]:
        """Booleans matching run_omnivla's module-scope flags."""
        return {
            "satellite":  "sat"   in self.modality,
            "lan_prompt": "lang"  in self.modality,
            "pose_goal":  "pose"  in self.modality,
            "image_goal": "image" in self.modality,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modality"] = sorted(self.modality)
        d["modality_id"] = self.modality_id
        return d
    
    def set_node_spacing(self, speed):
        self.goal_graph_node_spacing_m = speed / self.goal_graph_node_spacing_freq
