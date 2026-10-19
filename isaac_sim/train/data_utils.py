"""Shared helpers for Isaac Sim trajectory data.

Consumed by prismatic/vla/datasets/cctv_isaac_sim_dataset.py (training).

Per-trajectory layout (produced by isaac_sim/data_gen.py):
  <traj>/
    config.json     — env config that produced the trajectory
    meta.json       — outcome, n_frames, sample_hz, astar_path_length_m, ...
    astar_path.json — A* smoothed-path waypoints + total length
    frames.json     — GoalGraph: per-sample (xy, yaw, image_filename, cmd_v, cmd_w)
    frames/NNN.png  — robot egocentric POV
    scene/NNN.png   — third-person chase cam (for videos only, not training)
    cctv/NNN.png    — fixed infrastructure camera (phase 6 — emitted now, used later)
    cctv.png        — single CCTV snapshot at trajectory start

This module deliberately does NOT touch images or tensors — that's the
dataset class's job. We only parse the JSON sidecars and surface a clean
filter for "trajectories worth training on."
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrajectoryMeta:
    """Parsed meta.json. Subset of fields used downstream."""
    outcome: str                  # "reached" (success) | "stuck" | "path_deviation" | "off_floor" | "timeout"
    success: bool                 # convenience: outcome == "reached"
    n_frames: int                 # number of sample nodes in frames.json
    sample_hz: float              # 3.0 for paper-matched data
    n_steps: int                  # number of physics ticks executed
    astar_path_length_m: float    # planned-path length, for the A*-ratio metric
    env_config: str               # source env_config JSON path (replayability)
    elapsed_s: float              # wall-clock to collect


@dataclass(frozen=True)
class AstarPath:
    """A* planned path: LOS-smoothed waypoints + total length."""
    path_xy: list[tuple[float, float]]
    path_length_m: float
    n_waypoints: int


def load_meta(traj_dir: Path) -> TrajectoryMeta:
    """Parse result.json from a trajectory directory."""
    d = json.loads((Path(traj_dir) / "result.json").read_text())
    return TrajectoryMeta(
        outcome=d["outcome"],
        success=(d["outcome"] == "reached"),  # result.json has no `success` field
        n_frames=int(d["n_frames"]),
        sample_hz=float(d["sample_hz"]),
        n_steps=int(d["steps"]),  # result.json calls it `steps`
        astar_path_length_m=float(d["astar_path_length_m"]),
        env_config=str(d.get("env_config", "")),  # not in result.json; unused downstream
        elapsed_s=float(d.get("elapsed_s", 0.0)),  # not in result.json; unused downstream
    )


def load_astar_path(traj_dir: Path) -> AstarPath:
    """Parse astar_path.json from a trajectory directory.

    Used by eval to compute the A*-ratio: policy_path_length / astar_length.
    """
    d = json.loads((Path(traj_dir) / "astar_path.json").read_text())
    return AstarPath(
        path_xy=[(float(p[0]), float(p[1])) for p in d["path_xy"]],
        path_length_m=float(d["path_length_m"]),
        n_waypoints=int(d["n_waypoints"]),
    )


def is_successful(traj_dir: Path) -> bool:
    """True iff this trajectory reached the goal cleanly.

    Why this matters: outcomes != "success" mean the controller hit a
    collision, timeout, or got stuck. Using those as training labels
    would teach the model the failing behavior (e.g., a "stuck" sample
    pairs an image with a near-zero action label under nonzero commanded
    velocity — the model would learn "command motion, do nothing").
    Filter them out.

    Tolerant to missing/malformed meta.json — returns False rather than
    raising, so a broken trajectory dir doesn't crash dataset construction.
    """
    try:
        return load_meta(traj_dir).success
    except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError):
        return False


def list_successful_trajectories(
    root: Path,
    max_n: int | None = None,
) -> list[Path]:
    """Successful trajectory dirs under `root`, sorted by name.

    Sorting matters: the YAML's `seed` only does its job (deterministic
    shuffle across runs) if the *input order* to the shuffler is also
    deterministic. Name-sorting gives us that — re-running training on
    the same root produces the same sample order modulo the YAML seed.

    `max_n` caps the result. Used for the val/test 20-trajectory caps
    (we over-collect 30 each, then take the first 20 successful).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"trajectory root not found: {root}")
    # Skip per-config stdout files written by the old collect_trajectories.sh
    # (removed 2026-06; legacy data dirs may still contain such files).
    candidates = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.endswith(".stdout")
    )
    successful = [d for d in candidates if is_successful(d)]
    if max_n is not None:
        successful = successful[:max_n]
    return successful


def summarise_split(root: Path) -> dict:
    """Diagnostic summary of a trajectory root. Useful for sanity checks
    and the train-launch banner.

    Returns counts by outcome plus aggregates (mean frames per traj,
    total A* path length). Cheap — just parses sidecars.
    """
    root = Path(root)
    candidates = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith(".stdout"))
    by_outcome: dict[str, int] = {}
    n_frames_total = 0
    astar_length_total = 0.0
    n_success = 0
    for d in candidates:
        try:
            m = load_meta(d)
        except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError):
            by_outcome["missing_or_malformed"] = by_outcome.get("missing_or_malformed", 0) + 1
            continue
        by_outcome[m.outcome] = by_outcome.get(m.outcome, 0) + 1
        if m.success:
            n_success += 1
            n_frames_total += m.n_frames
            astar_length_total += m.astar_path_length_m
    return {
        "root": str(root),
        "n_dirs": len(candidates),
        "by_outcome": by_outcome,
        "n_success": n_success,
        "mean_frames_per_success": (n_frames_total / n_success) if n_success else 0.0,
        "total_astar_length_m": astar_length_total,
    }


if __name__ == "__main__":
    # CLI diagnostic: `python -m isaac_sim.train.data_utils <traj_root>`
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m isaac_sim.train.data_utils <traj_root>", file=sys.stderr)
        sys.exit(2)
    summary = summarise_split(Path(sys.argv[1]))
    print(json.dumps(summary, indent=2))
