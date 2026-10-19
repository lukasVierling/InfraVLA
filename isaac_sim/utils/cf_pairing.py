"""Counterfactual (cf0 vs cfX) pairing + decision-frame geometry.

Pure numpy / filesystem helpers for counterfactual-window mining and paired
probes. Kept model-free and torch-free so the logic is unit-testable on CPU
without loading a VLA.

Background — the counterfactual construction (build_2x3.py::generate_cf):
from one base scene the generator emits a family of
variants that share start / goal / heading / distractors *byte-for-byte* and
differ ONLY in which transit aisles a forklift seals. `<base>_cf0` is the
unblocked twin; `<base>_cfB`, `_cfC`, `_cfBE`, ... seal the named aisle(s). At
the decision point the seal is not resolvable from the onboard (ego) view — only
the fixed CCTV cameras see it — so the correct turn is disambiguated by CCTV
alone. That is the whole experiment: a cf0/cfX pair is a controlled A/B where the
only changed input is the CCTV stream.

This module provides:
  - find_cf_pairs()   : discover (cf0, cfX) pairs in a split directory.
  - load_pose_stream(): read a trajectory's world (x, y, yaw) stream.
  - decision_frame()  : the earliest obs step whose future path diverges between
                        the two variants — where the model must already commit.
  - chunk geometry    : final-waypoint heading / lateral + signed separations,
                        the scalars the probe compares across models.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

# isaac_sim must be importable for GoalGraph; this file lives under isaac_sim/
# so the repo root is three parents up.
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from isaac_sim.planning.image_graph.graph import GoalGraph  # noqa: E402


# A run dir is "<scene>__pose__YYYY-MM-DD_HH-MM-SS"; the scene is
# "warehouse_2x3_<4-digit base>_cf<suffix>". "0" = unblocked twin, otherwise a
# subset of the transit aisles B/C/E/F.
_SCENE_RE = re.compile(r"^(warehouse_2x3_\d{4})_cf([A-Z0]+)__pose__")
# Language-task twins (build_2x3_lang.py): same family semantics, suffix
# tA..tD (one per aisle object) instead of a seal pattern; optional langvN
# prefix. A separate regex so the seal parser above is unaffected.
_SCENE_RE_LANG = re.compile(r"^(warehouse_2x3_lang(?:v\d+)?_\d{4})_t([A-D])__pose__")
# Real-robot Spot families: NN_a / NN_b / NN_c takes from a shared spawn
# (0.14 m / 3 deg median spread), letter = target object. Same
# family semantics as the sim variants, so the miner works unchanged.
_SCENE_RE_SPOT = re.compile(r"^(\d{2})_([abc])_\d{4}_")


def _parse_run_dir(name: str) -> tuple[str, str] | None:
    """(base, variant_suffix) from a run-dir name, or None if it doesn't match.
    e.g. "warehouse_2x3_0904_cfC__pose__2026-07-04_..." -> ("warehouse_2x3_0904", "C");
         "warehouse_2x3_lang_0000_tA__pose__..." -> ("warehouse_2x3_lang_0000", "A")."""
    m = _SCENE_RE.match(name) or _SCENE_RE_LANG.match(name) or _SCENE_RE_SPOT.match(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def find_cf_pairs(split_dir: str | Path) -> list[dict]:
    """Find (cf0, cfX) pairs in a split directory."""
    split_dir = Path(split_dir)
    by_base: dict[str, dict[str, Path]] = {}
    #parse the names and add them into a dictionary that contains another dict per suffix
    for d in sorted(split_dir.iterdir()):
        if not d.is_dir():
            continue
        parsed = _parse_run_dir(d.name)
        if parsed is None:
            continue
        base, suffix = parsed
        by_base.setdefault(base, {})[suffix] = d

    pairs: list[dict] = []
    for base in sorted(by_base):
        variants = by_base[base]
        cf0_dir = variants.get("0")
        if cf0_dir is None:
            continue                              # no control twin -> skip family
        for suffix in sorted(variants):
            if suffix == "0":
                continue
            pairs.append({
                "base": base,
                "blocked": suffix,
                "blocked_set": frozenset(suffix),
                "cf0_dir": cf0_dir,
                "cfx_dir": variants[suffix],
                "cf0_scene": f"{base}_cf0",
                "cfx_scene": f"{base}_cf{suffix}",
            })
    return pairs


def load_pose_stream(traj_dir: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """given a trajectory path it returns the positions, yaws and names of the images"""
    graph = GoalGraph.load(Path(traj_dir) / "frames.json")
    positions = np.array([[nd.xy[0], nd.xy[1]] for nd in graph.nodes], dtype=np.float64)
    yaws = np.array([nd.yaw_rad for nd in graph.nodes], dtype=np.float64)
    names = [nd.image_filename for nd in graph.nodes]
    return positions, yaws, names


def decision_frame(
    pos_cf0: np.ndarray,
    pos_cfx: np.ndarray,
    n_chunk: int,
    dev_thresh_m: float = 0.75,
    min_t: int = 0,
) -> tuple[int | None, float, float]:
    """looks for a decision frame and returns the time deviation and start offset"""
    start_offset = float(np.linalg.norm(pos_cf0[0] - pos_cfx[0]))
    n = min(len(pos_cf0), len(pos_cfx))
    #furthest lookaehad based on n_chunk
    t_max = n - 1 - n_chunk
    for t in range(min_t, max(min_t, t_max) + 1):
        #check at each time step
        fut0 = pos_cf0[t + 1: t + 1 + n_chunk]
        futx = pos_cfx[t + 1: t + 1 + n_chunk]
        m = min(len(fut0), len(futx))
        if m == 0:
            break
        #get the deviation
        dev = float(np.linalg.norm(fut0[:m] - futx[:m], axis=1).max())
        #check whether deviation exceeds limit
        if dev > dev_thresh_m:
            return t, dev, start_offset
    return None, 0.0, start_offset


def decision_frame_yaw(
    yaw_a: np.ndarray,
    yaw_b: np.ndarray,
    eps_deg: float = 1.0,
    min_t: int = 0,
) -> tuple[int | None, int | None, float]:
    """Crux by HEADING divergence — the leak-aware twin of decision_frame().

    decision_frame() asks "when do the two futures separate in POSITION?".
    That is the wrong question for detecting a proprioceptive leak: on langv2
    the robot rotates in place before it translates, so its own yaw gives the
    answer away a frame or two BEFORE any position gap opens (measured: twin
    yaw spread 0.00deg at t=0 -> 3.41 at t=1 -> 19.90 at t=2, while position
    separation at t=1 is only 0.02 m). A model can read that heading off its
    own observation history and score perfectly without using the instruction.

    So here we find `t_div` = the first frame at which the two twins' headings
    differ by more than `eps_deg`, and define the CRUX as `t_div - 1`: the LAST
    frame at which the robot has NOT yet initiated the diverging turn. A window
    ending at (or before) the crux is provably un-leaked — the two twins are in
    the same physical state there, so only the instruction can separate them.

    eps_deg is a tolerance, not a strict equality: identical oracle plans still
    accumulate ~1e-3 deg of integrator noise.

    Returns (t_crux, t_div, max_dev_deg). Both are None if the headings never
    diverge (a vacuous pair). t_crux is clamped to >= min_t.
    """
    n = min(len(yaw_a), len(yaw_b))
    if n == 0:
        return None, None, 0.0
    # Wrap the DIFFERENCE into [-pi, pi] — not each stream separately. Two
    # headings of +pi and -pi are identical, but unwrapping each stream on its
    # own leaves them 2pi apart and reports a bogus 360 deg divergence at frame
    # 0.
    diff = yaw_a[:n] - yaw_b[:n]
    diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
    dev = np.abs(np.degrees(diff))
    over = np.where(dev[min_t:] > float(eps_deg))[0]
    if len(over) == 0:
        return None, None, float(dev.max())
    t_div = int(over[0]) + min_t
    t_crux = max(min_t, t_div - 1)
    return t_crux, t_div, float(dev.max())


def family_decision_windows_yaw(
    yaws_by_variant: dict[str, np.ndarray],
    positions_by_variant: dict[str, np.ndarray],
    eps_deg: float,
    pre_steps: int,
    post_steps: int,
    max_start_offset_m: float = 0.5,
) -> tuple[dict[str, set[int]], list[dict]]:
    """family_decision_windows() but with the yaw crux (see decision_frame_yaw).

    Window is [t_crux - pre_steps, t_crux + post_steps]. NOTE post_steps > 0
    reaches PAST the crux into frames where the headings have already split —
    i.e. leaked frames. post_steps=0 is the provably-clean choice; 1 includes
    only the first barely-over-eps frame.
    """
    suffixes = sorted(yaws_by_variant)
    windows: dict[str, set[int]] = {s: set() for s in suffixes}
    pair_log: list[dict] = []
    for ai in range(len(suffixes)):
        for bi in range(ai + 1, len(suffixes)):
            a, b = suffixes[ai], suffixes[bi]
            pos_a, pos_b = positions_by_variant[a], positions_by_variant[b]
            start_off = float(np.linalg.norm(pos_a[0] - pos_b[0]))
            t_crux, t_div, dev = decision_frame_yaw(
                yaws_by_variant[a], yaws_by_variant[b], eps_deg=eps_deg)
            entry = {"a": a, "b": b, "t_star": t_crux, "t_div": t_div,
                     "dev_m": dev, "start_offset_m": start_off}
            if start_off > max_start_offset_m:
                entry["status"] = "start_mismatch"
            elif t_crux is None:
                entry["status"] = "vacuous"
            else:
                entry["status"] = "ok"
                lo = max(0, t_crux - pre_steps)
                hi = t_crux + post_steps
                for suffix, pos in ((a, pos_a), (b, pos_b)):
                    windows[suffix].update(range(lo, min(hi, len(pos) - 1) + 1))
            pair_log.append(entry)
    return windows, pair_log


# ---------------------------------------------------------------------------
# Counterfactual FAMILIES (generalisation of the cf0/cfX pairs above).
#
# The balanced build_2x3 datagen emits each base as a pair of BLOCKED variants
# with different sealed sets (e.g. cfCE vs cfCF) and no cf0 control —
# find_cf_pairs() returns nothing on such a split. A "family" is the general
# unit: every variant sharing one base, pairwise-comparable because the
# generator gives all of them byte-identical start/goal/distractors. Consumed by
# the decision-window mining in prismatic/vla/datasets/dataset_utils/.
# ---------------------------------------------------------------------------

def group_cf_families(run_dir_names: list[str]) -> tuple[dict[str, dict[str, int]], list[int]]:
    """Group run dirs into cf families {base: {suffix: index}}; also return indices that did not parse."""
    families: dict[str, dict[str, int]] = {}
    non_cf: list[int] = []
    for i, name in enumerate(run_dir_names):
        parsed = _parse_run_dir(name)
        if parsed is None:
            non_cf.append(i)
            continue
        base, suffix = parsed\
        #add base to the family dict and then add another entry for the suffix with index of trajectory
        families.setdefault(base, {})[suffix] = i
    return families, non_cf


def family_decision_windows(
    positions_by_variant: dict[str, np.ndarray],
    n_chunk: int,
    dev_thresh_m: float,
    pre_steps: int,
    post_steps: int,
    max_start_offset_m: float = 0.5,
) -> tuple[dict[str, set[int]], list[dict]]:
    """Per-variant decision windows from every pair in a family, plus a per-pair log."""
    suffixes = sorted(positions_by_variant)
    windows: dict[str, set[int]] = {s: set() for s in suffixes}
    pair_log: list[dict] = []
    for ai in range(len(suffixes)):
        for bi in range(ai + 1, len(suffixes)):
            a, b = suffixes[ai], suffixes[bi]
            pos_a, pos_b = positions_by_variant[a], positions_by_variant[b]
            t_star, dev, start_off = decision_frame(
                pos_a, pos_b, n_chunk=n_chunk, dev_thresh_m=dev_thresh_m)
            entry = {"a": a, "b": b, "t_star": t_star, "dev_m": dev,
                     "start_offset_m": start_off}
            if start_off > max_start_offset_m:
                entry["status"] = "start_mismatch"
            elif t_star is None:
                entry["status"] = "vacuous"
            else:
                entry["status"] = "ok"
                lo = max(0, t_star - pre_steps)
                hi = t_star + post_steps                   # inclusive; capped to the trajectory length below
                for suffix, pos in ((a, pos_a), (b, pos_b)):
                    windows[suffix].update(range(lo, min(hi, len(pos) - 1) + 1))
            pair_log.append(entry)
    return windows, pair_log


# ---------------------------------------------------------------------------
# Action-chunk geometry.
#
# A chunk is (N, 4) = [dx, dy, cos(dyaw), sin(dyaw)] in the robot-local frame at
# the observation step, x = forward, y = LEFT, positions CUMULATIVE (waypoint k
# is the pose after k+1 steps, not a per-step delta). So chunk[-1, :2] is the net
# displacement over the whole chunk — "which way, how far" in one vector. We read
# the committed direction off that final waypoint.
# ---------------------------------------------------------------------------

def final_wp_lateral(chunk: np.ndarray) -> float:
    """Return dy (lateral) of the final waypoint."""
    return float(chunk[-1, 1])


def final_wp_forward(chunk: np.ndarray) -> float:
    """Return dx (forward) of the final waypoint."""
    return float(chunk[-1, 0])


def final_wp_heading(chunk: np.ndarray) -> float:
    """Return the bearing (radians) of the final waypoint."""
    return math.atan2(chunk[-1, 1], chunk[-1, 0])


def wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def lateral_separation(chunk_x: np.ndarray, chunk_0: np.ndarray) -> float:
    """Lateral difference between two chunks at their final waypoint."""
    return final_wp_lateral(chunk_x) - final_wp_lateral(chunk_0)


def heading_separation(chunk_x: np.ndarray, chunk_0: np.ndarray) -> float:
    """returns the heading separation for two chunks based on their final waypoint."""
    return wrap_pi(final_wp_heading(chunk_x) - final_wp_heading(chunk_0))
