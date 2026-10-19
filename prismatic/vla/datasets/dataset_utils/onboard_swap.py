"""Counterfactual onboard-bundle swap builders.

build_onboard_swap_map   — in-place probabilistic swap inside decision windows
build_onboard_swap_extra — forced-swap EXTRA samples (diagonal-invisible gate)

Both attach state to an existing IsaacSimDataset (onboard_swap_map /
onboard_swap_p / onboard_swap_extra) that its __getitem__ consumes. See the
notes block at the end of this file for the gate-design rationale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from isaac_sim.utils.cf_pairing import (  # noqa: E402
    decision_frame, group_cf_families,
)


def _aligned_ego_diff(traj_a, traj_b, t: int, size: int = 224) -> float:
    """Mean abs pixel diff (0-255) between the twins' ego frames at step t.
    At an aligned step (~identical poses) this measures CONTENT difference
    only — i.e. whether the differing seal forklift is in the onboard view."""
    from PIL import Image
    a = np.asarray(Image.open(traj_a.frames_dir / traj_a.graph.nodes[t].image_filename)
                   .convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)
    b = np.asarray(Image.open(traj_b.frames_dir / traj_b.graph.nodes[t].image_filename)
                   .convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)
    return float(np.abs(a - b).mean())


def build_onboard_swap_map(
    dataset,
    *,
    pre_steps: int,
    post_steps: int,
    dev_thresh_m: float,
    max_offset_m: float,
    p: float,
    junction_filter: str = "second",
    split_y_boundary: float | None = None,
    ego_diff_thresh_px: float | None = None,
    first_frames: int | None = None,
    graft_err_cap_m: float | None = None,
) -> dict:
    """Attach an onboard-swap map (+ probability p) to `dataset`.

    For every ok cf pair passing the junction gate (junction_filter
    "second"/"first"/"any", classified by the pair's t* y-position against
    split_y_boundary — required unless "any"), every in-window frame t with
    twin pose offset < max_offset_m becomes swappable in BOTH directions:
        swap_map[(traj_idx, t)] -> [sibling_idx, ...]
    ego_diff_thresh_px (optional): additionally require the aligned-frame ego
    diff below the threshold. Returns a diagnostics dict (per-pair gate
    results + counts). Window definition matches mine_cf_windows exactly.

    first_frames (real-robot mode, 2026-08-21): bypass decision-frame mining and
    junction gates entirely - swap is allowed ONLY at t in [0, first_frames),
    i.e. the spawn frames where the twins are physically interchangeable. The
    per-frame gate becomes graft_err_cap_m: an estimate of the grafted label's
    endpoint error, |dpos| + reach * |dyaw| (reach ~= 1.05 m chunk end), which
    is what actually bounds how wrong the swapped supervision can be (measured
    on this data: median 0.107 m, outliers up to 0.32 m all from spawn-offset
    families 10/17/24).
    """
    if getattr(dataset, "abs_pose", False):
        raise ValueError("onboard swap is incompatible with abs_pose=True "
                         "(world-frame vec would mix own pose with sibling goal)")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"onboard swap p must be in [0, 1], got {p}")
    if junction_filter not in ("second", "first", "any"):
        raise ValueError(f"junction_filter must be second|first|any, got {junction_filter!r}")
    if junction_filter != "any" and split_y_boundary is None:
        raise ValueError("split_y_boundary is required with a junction filter "
                         "(zone-derived, e.g. (A.y_lo + D.y_hi)/2 = 16.9 for 2x3)")

    names = [tr.traj_dir.name for tr in dataset.trajectories]
    families, _ = group_cf_families(names)
    n_chunk = dataset.num_actions_chunk

    swap_map: dict[tuple[int, int], list[int]] = {}
    pair_diag: list[dict] = []
    for base, members in sorted(families.items()):
        if len(members) < 2:
            continue
        sfx = sorted(members)
        for ai in range(len(sfx)):
            for bi in range(ai + 1, len(sfx)):
                ia, ib = members[sfx[ai]], members[sfx[bi]]
                tra, trb = dataset.trajectories[ia], dataset.trajectories[ib]
                n_min = min(tra.n_frames, trb.n_frames)
                if first_frames is not None:
                    # spawn-frame mode: no crux, no junction; gate per frame on
                    # the estimated graft endpoint error below.
                    start_off = float(np.linalg.norm(tra.positions[0] - trb.positions[0]))
                    if start_off > 0.5:
                        continue
                    lo, hi = 0, min(first_frames - 1, n_min - 1)
                    pair_diag.append(dict(base=base, a=sfx[ai], b=sfx[bi], t_star=None,
                                          junction="spawn", t_aligned=None,
                                          ego_diff=None, eligible=True))
                    for t in range(lo, hi + 1):
                        dp = float(np.linalg.norm(tra.positions[t] - trb.positions[t]))
                        if graft_err_cap_m is not None:
                            dyaw = abs(float(np.arctan2(
                                np.sin(tra.yaws[t] - trb.yaws[t]),
                                np.cos(tra.yaws[t] - trb.yaws[t]))))
                            ok = (dp + 1.05 * dyaw) < graft_err_cap_m
                        else:
                            ok = dp < max_offset_m
                        if ok:
                            swap_map.setdefault((ia, t), []).append(ib)
                            swap_map.setdefault((ib, t), []).append(ia)
                    continue
                t_star, _dev, start_off = decision_frame(
                    tra.positions, trb.positions, n_chunk, dev_thresh_m=dev_thresh_m)
                if t_star is None or start_off > 0.5:
                    continue
                junction = ("first" if tra.positions[t_star][1] > (split_y_boundary or 0.0)
                            else "second")
                eligible = junction_filter == "any" or junction == junction_filter
                t_al, diff = None, None
                if eligible and ego_diff_thresh_px is not None:
                    for k in range(min(t_star, n_min - 1), -1, -1):
                        if np.linalg.norm(tra.positions[k] - trb.positions[k]) < 0.01:
                            t_al = k
                            break
                    if t_al is None:
                        eligible = False
                    else:
                        diff = _aligned_ego_diff(tra, trb, t_al)
                        eligible = diff < ego_diff_thresh_px
                pair_diag.append(dict(base=base, a=sfx[ai], b=sfx[bi], t_star=t_star,
                                      junction=junction, t_aligned=t_al,
                                      ego_diff=diff, eligible=eligible))
                if not eligible:
                    continue
                lo = max(0, t_star - pre_steps)
                hi = min(t_star + post_steps, n_min - 1)
                for t in range(lo, hi + 1):
                    if np.linalg.norm(tra.positions[t] - trb.positions[t]) < max_offset_m:
                        swap_map.setdefault((ia, t), []).append(ib)
                        swap_map.setdefault((ib, t), []).append(ia)

    dataset.onboard_swap_map = swap_map
    dataset.onboard_swap_p = float(p)

    n_pairs = len(pair_diag)
    n_elig = sum(1 for d in pair_diag if d["eligible"])
    sample_keys = set(dataset.samples_index)
    n_swappable = sum(1 for k in swap_map if k in sample_keys)
    diag = dict(pairs_total=n_pairs, pairs_eligible=n_elig,
                swappable_samples=n_swappable, n_samples=len(dataset),
                pair_diag=pair_diag,
                knobs=dict(pre_steps=pre_steps, post_steps=post_steps,
                           dev_thresh_m=dev_thresh_m, max_offset_m=max_offset_m,
                           junction_filter=junction_filter,
                           split_y_boundary=split_y_boundary,
                           ego_diff_thresh_px=ego_diff_thresh_px, p=p,
                           first_frames=first_frames,
                           graft_err_cap_m=graft_err_cap_m))
    print(f"[onboard_swap] pairs {n_elig}/{n_pairs} eligible "
          f"(junction={junction_filter}"
          + (f", ego-diff<{ego_diff_thresh_px}px" if ego_diff_thresh_px is not None else "")
          + f") | swappable samples {n_swappable}/{len(dataset)} | "
          f"p={p} offset_cap={max_offset_m}m")
    return diag


_AHEAD_AISLE = {"B": "E", "C": "F"}


def _diff_south_seals(a: str, b: str) -> set:
    sa = set() if a == "0" else set(a)
    sb = set() if b == "0" else set(b)
    return (sa ^ sb) & {"E", "F"}


def build_onboard_swap_extra(
    dataset,
    *,
    pre_steps: int,
    post_steps: int,
    dev_thresh_m: float,
    max_offset_m: float,
    split_x_boundary: float,
    split_y_boundary: float,
) -> dict:
    """Attach forced-swap EXTRA samples to `dataset` (diagonal-invisible gate).

    For every ok cf pair whose differing south seal is diagonal to the exit
    aisle (x at t* against split_x_boundary: B/E column west, C/F east), every
    in-range frame [t*-pre, t*+post] that is an actual dataset sample and has
    twin pose offset < max_offset_m is appended (BOTH directions) to
    dataset.onboard_swap_extra as (traj_idx, t, sibling_idx).
    Requires final_goal_prob == 1.0 (the swap serves the sibling's arrival
    frame as the goal — only coherent when goals are always final).
    """
    if getattr(dataset, "abs_pose", False):
        raise ValueError("onboard swap is incompatible with abs_pose=True")
    if float(getattr(dataset, "final_goal_prob", 0.0)) < 1.0:
        raise ValueError(
            "build_onboard_swap_extra requires final_goal_prob=1.0 — forced "
            f"swaps serve the sibling's arrival frame as the goal (got "
            f"{dataset.final_goal_prob})")

    names = [tr.traj_dir.name for tr in dataset.trajectories]
    families, _ = group_cf_families(names)
    n_chunk = dataset.num_actions_chunk
    sample_keys = set(dataset.samples_index)

    extra: list[tuple[int, int, int]] = []
    pair_diag: list[dict] = []
    for base, members in sorted(families.items()):
        if len(members) < 2:
            continue
        sfx = sorted(members)
        for ai in range(len(sfx)):
            for bi in range(ai + 1, len(sfx)):
                ia, ib = members[sfx[ai]], members[sfx[bi]]
                tra, trb = dataset.trajectories[ia], dataset.trajectories[ib]
                t_star, _dev, start_off = decision_frame(
                    tra.positions, trb.positions, n_chunk, dev_thresh_m=dev_thresh_m)
                if t_star is None or start_off > 0.5:
                    continue
                if tra.positions[t_star][1] > split_y_boundary:
                    continue
                exit_aisle = ("B" if tra.positions[t_star][0] < split_x_boundary
                              else "C")
                dsouth = _diff_south_seals(sfx[ai], sfx[bi])
                eligible = bool(dsouth) and _AHEAD_AISLE[exit_aisle] not in dsouth
                pair_diag.append(dict(base=base, a=sfx[ai], b=sfx[bi],
                                      t_star=t_star, exit_aisle=exit_aisle,
                                      diff_south=sorted(dsouth),
                                      eligible=eligible))
                if not eligible:
                    continue
                n_min = min(tra.n_frames, trb.n_frames)
                lo = max(0, t_star - pre_steps)
                hi = min(t_star + post_steps, n_min - 1)
                for t in range(lo, hi + 1):
                    if np.linalg.norm(tra.positions[t] - trb.positions[t]) >= max_offset_m:
                        continue
                    if (ia, t) in sample_keys:
                        extra.append((ia, t, ib))
                    if (ib, t) in sample_keys:
                        extra.append((ib, t, ia))

    dataset.onboard_swap_extra = extra
    n_elig = sum(1 for d in pair_diag if d["eligible"])
    diag = dict(pairs_total=len(pair_diag), pairs_eligible=n_elig,
                n_extra=len(extra), n_honest=len(dataset.samples_index),
                pair_diag=pair_diag,
                knobs=dict(pre_steps=pre_steps, post_steps=post_steps,
                           dev_thresh_m=dev_thresh_m, max_offset_m=max_offset_m,
                           split_x_boundary=split_x_boundary))
    print(f"[onboard_swap_extra] pairs {n_elig}/{len(pair_diag)} eligible "
          f"(diagonal-invisible gate) | +{len(extra)} forced-swap extra samples "
          f"on {len(dataset.samples_index)} honest ({len(dataset)} total) | "
          f"range [t*-{pre_steps}, t*+{post_steps}]")
    return diag
