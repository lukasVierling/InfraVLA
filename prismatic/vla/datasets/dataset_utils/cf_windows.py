"""Counterfactual decision-window mining and its two consumers.

mine_cf_windows        — mine per-trajectory decision-window step sets
apply_cf_window_filter — cut a built dataset's samples_index to the windows
cf_window_sample_mask  — boolean per-sample mask for weighted resampling

See the notes block at the end of this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from isaac_sim.utils.cf_pairing import (  # noqa: E402
    family_decision_windows, family_decision_windows_yaw, group_cf_families,
)


def mine_cf_windows(
    traj_names: list[str],
    traj_positions: list[np.ndarray],
    n_chunk: int,
    dev_thresh_m: float,
    pre_steps: int,
    post_steps: int,
    criterion: str = "distance",
    traj_yaws: list[np.ndarray] | None = None,
    yaw_eps_deg: float = 1.0,
) -> dict:
    """
    Returns a dictionary with the mapping trajectory to where the cf scenes live and some extra info for verifying things
    
    Args:
        - traj_name: list of strings with the names of the trajectories (file names)
        - traj_position: the XY of the trajecotry at i
        - n_chunk: how far ahead does the decision frame look to find out where deviation happens
        - dev_thresh_m: threshold to decide for a divergence or not
        - pre_steps: how many t after the divergence should be considered
        - post_steps: how many t before the divergence should be considered

    Returns:
        -dict including a mapping trajectory to allowed indices and data fields for verification
    """
    #check whether pre and post steps are valid 
    if criterion not in ("distance", "yaw"):
        raise ValueError(f"criterion must be 'distance' or 'yaw', got {criterion!r}")
    if criterion == "yaw" and traj_yaws is None:
        raise ValueError("criterion='yaw' requires traj_yaws")
    if pre_steps < 0 or post_steps < 0:
        raise ValueError(
            f"decision window steps must be >= 0, got pre={pre_steps} "
            f"post={post_steps}")
    # get the family names and find out which ones have no cf counterpart
    families, non_cf_idx = group_cf_families(traj_names)
    #get the singletons that have no counterfactual partner
    singleton_idx = [next(iter(m.values()))
                     for m in families.values() if len(m) < 2]
    #mapping from index to a set of allowed scenes
    windows_by_traj: dict[int, set[int]] = {i: set() for i in range(len(traj_names))}
    pair_log: list[dict] = []
    for base, members in sorted(families.items()):
        #check whether we have more or eq to 2 scenes 
        if len(members) < 2:
            continue
        #each suffix maps to their trajectory expressed in coordinates
        positions_by_variant = {
            suffix: traj_positions[i] for suffix, i in members.items()
        }
        #find the decision points and have window around it
        if criterion == "yaw":
            # crux = last frame BEFORE the twins' headings diverge (leak-aware);
            # see cf_pairing.decision_frame_yaw for why position is the wrong test.
            yaws_by_variant = {
                suffix: traj_yaws[i] for suffix, i in members.items()
            }
            windows, fam_log = family_decision_windows_yaw(
                yaws_by_variant, positions_by_variant,
                eps_deg=yaw_eps_deg, pre_steps=pre_steps, post_steps=post_steps,
            )
        else:
            windows, fam_log = family_decision_windows(
                positions_by_variant,
                n_chunk=n_chunk,
                dev_thresh_m=dev_thresh_m,
                pre_steps=pre_steps,
                post_steps=post_steps,
            )
        #union over the new window with existing windows
        for suffix, i in members.items():
            windows_by_traj[i] |= windows[suffix]
        #log the entry with their respective base 
        for entry in fam_log:
            pair_log.append({**entry, "base": base})
    #return results
    return {
        "windows_by_traj": windows_by_traj,
        "pair_log": pair_log,
        "non_cf_idx": non_cf_idx,
        "singleton_idx": singleton_idx,
        "n_families": sum(1 for m in families.values() if len(m) >= 2),
    }


def print_pair_warnings(pair_log: list[dict], tag: str) -> None:
    """prints warnings if any of the pair logs are not ok"""
    for entry in pair_log:
        if entry["status"] != "ok":
            print(f"[{tag}]   WARNING {entry['status']}: "
                  f"{entry['base']} cf{entry['a']} vs cf{entry['b']} "
                  f"(start_offset={entry['start_offset_m']:.2f} m)")


def apply_cf_window_filter(dataset, *, pre_steps: int, post_steps: int,
                           dev_thresh_m: float, tag: str = "cf_window_filter",
                           criterion: str = "distance",
                           yaw_eps_deg: float = 1.0,
                           ) -> dict:
    """Apply window mining and print the final summary of mining. Return the mining results"""
    #Get the mining results
    mining = mine_cf_windows(
        traj_names=[tr.traj_dir.name for tr in dataset.trajectories],
        traj_positions=[tr.positions for tr in dataset.trajectories],
        n_chunk=dataset.num_actions_chunk,
        dev_thresh_m=float(dev_thresh_m),
        pre_steps=int(pre_steps),
        post_steps=int(post_steps),
        criterion=str(criterion),
        traj_yaws=[tr.yaws for tr in dataset.trajectories],
        yaw_eps_deg=float(yaw_eps_deg),
    )
    windows = mining["windows_by_traj"]
    #how many samples before filtering
    n_before = len(dataset.samples_index)
    #get listof index and trajectory if they are in the wndows
    dataset.samples_index = [
        (ti, t) for (ti, t) in dataset.samples_index if t in windows[ti]
    ]
    #get length after mining
    n_after = len(dataset.samples_index)
    #error if no window found
    if n_after == 0:
        raise RuntimeError(
            f"{tag}: no samples survived decision-window filtering — is this "
            "really a counterfactual (_cf) split?")
    n_pairs_by_status: dict[str, int] = {}
    #proceed to next status for pair log
    for entry in mining["pair_log"]:
        n_pairs_by_status[entry["status"]] = \
            n_pairs_by_status.get(entry["status"], 0) + 1
    #how many trajectories are kept
    kept_traj = sum(1 for ts in windows.values() if ts)
    #summary
    summary = {
        "n_traj_total": len(dataset.trajectories),
        "n_traj_kept": kept_traj,
        "n_traj_non_cf": len(mining["non_cf_idx"]),
        "n_traj_singleton": len(mining["singleton_idx"]),
        "n_families": mining["n_families"],
        "pairs_by_status": n_pairs_by_status,
        "n_samples_before": n_before,
        "n_samples_after": n_after,
    }
    dataset.cf_windows_by_traj = windows
    dataset.cf_pair_log = mining["pair_log"]
    dataset.cf_summary = summary
    #print all the important info
    print(f"[{tag}] {summary['n_families']} families "
          f"({kept_traj}/{len(dataset.trajectories)} trajs keep windows; "
          f"{summary['n_traj_non_cf']} non-cf, "
          f"{summary['n_traj_singleton']} singleton dropped) | "
          f"pairs {n_pairs_by_status} | samples {n_before} -> {n_after} "
          f"({100.0 * n_after / max(n_before, 1):.1f}%) | "
          f"window pre={pre_steps} post={post_steps} thresh={dev_thresh_m}m")
    print_pair_warnings(mining["pair_log"], tag)
    return mining


def cf_window_sample_mask(
    dataset,
    *,
    pre_steps: int,
    post_steps: int,
    dev_thresh_m: float,
    criterion: str = "distance",
    yaw_eps_deg: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Mine the windows and return the mined result and a mask over the original datset."""
    #get the windows
    mining = mine_cf_windows(
        traj_names=[traj.traj_dir.name for traj in dataset.trajectories],
        traj_positions=[traj.positions for traj in dataset.trajectories],
        n_chunk=dataset.num_actions_chunk,
        dev_thresh_m=dev_thresh_m,
        pre_steps=pre_steps,
        post_steps=post_steps,
        criterion=criterion,
        traj_yaws=[traj.yaws for traj in dataset.trajectories],
        yaw_eps_deg=yaw_eps_deg,
    )
    #get windows from mining output (ignore datafields for verification)
    windows = mining["windows_by_traj"]
    #lastly obtain a mask over the datset
    mask = np.array([t in windows[ti] for ti, t in dataset.samples_index],
                    dtype=bool)
    return mask, mining
