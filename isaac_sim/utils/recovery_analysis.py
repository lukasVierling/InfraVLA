"""Pure geometry for the recovery pipeline (recovery.py) — no sim, no torch.

point_to_polyline_dist — min distance from points to a polyline
find_derailment        — derailment verdict + onset + last-aligned index

See the notes block at the end of this file.
"""

from __future__ import annotations

import numpy as np


def point_to_polyline_dist(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Min distance from each point (n, 2) to a polyline (m, 2). Returns (n,).

    Distance to the nearest SEGMENT (projection clamped per segment), not the
    nearest vertex — a rollout hugging the path between two widely spaced GT
    samples must read as on-path.
    """
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    if len(polyline) == 0:
        raise ValueError("empty polyline")
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)

    a = polyline[:-1]
    ab = polyline[1:] - a
    ab_len2 = (ab ** 2).sum(axis=1)
    ab_len2 = np.where(ab_len2 == 0.0, 1e-12, ab_len2)

    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None, :, :]).sum(axis=2) / ab_len2[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return d.min(axis=1)


def find_derailment(
    xy: np.ndarray,
    gt_polyline: np.ndarray,
    *,
    dev_thresh_m: float,
    sustain_steps: int,
    aligned_eps_m: float,
) -> dict:
    """Derailment verdict for one rollout trace against the GT path.

    Derailed iff some step's distance-to-GT exceeds dev_thresh_m and STAYS
    above it for sustain_steps consecutive steps (an excursion running into
    the end of the trace counts). Same divergence principle as the original
    counterfactual mining, applied rollout-vs-GT instead of twin-vs-twin.

    Returns dict(derailed, onset_idx, last_aligned_idx, deviation, max_dev_m):
      onset_idx        — first step of the sustained excursion (None if not derailed)
      last_aligned_idx — last step BEFORE onset with deviation <= aligned_eps_m
                         ("the last time the trajectories were together"; 0 if
                         the trace was never that close). The recovery spawn is
                         sampled uniformly in [last_aligned_idx, onset_idx].
    """
    xy = np.asarray(xy, dtype=np.float64)
    if sustain_steps < 1:
        raise ValueError(f"sustain_steps must be >= 1, got {sustain_steps}")

    dev = point_to_polyline_dist(xy, gt_polyline)
    above = dev > dev_thresh_m

    onset_idx = None
    for i in range(len(above)):
        if above[i] and above[i:i + sustain_steps].all():
            onset_idx = i
            break

    if onset_idx is None:
        return dict(derailed=False, onset_idx=None, last_aligned_idx=None,
                    deviation=dev,
                    max_dev_m=float(dev.max()) if len(dev) else 0.0)

    aligned = np.where(dev[:onset_idx] <= aligned_eps_m)[0]
    last_aligned_idx = int(aligned[-1]) if len(aligned) else 0

    return dict(derailed=True, onset_idx=int(onset_idx),
                last_aligned_idx=last_aligned_idx, deviation=dev,
                max_dev_m=float(dev.max()))

import re as _re

_REC_DIRNAME = _re.compile(r"^rec_(?P<src>.+)_t(?P<t>\d+)(?:_s(?P<k>\d+))?$")


def parse_rec_dirname(name: str) -> tuple[str, int, int]:
    """(source_traj_dirname, sample_t, spawn_k) from a stitched rec_* dirname.

    spawn_k is 0 for the legacy single-spawn naming (no _s suffix); extra
    spawns of the same derailment (n_spawns > 1, 2026-07-17) carry _s1, _s2...
    Keep ALL consumers on this helper — the t-last convention breaks the
    moment someone parses with rsplit("_t") against an _s-suffixed name.
    """
    m = _REC_DIRNAME.match(name)
    if not m:
        raise ValueError(f"not a rec_* dirname: {name!r}")
    return m.group("src"), int(m.group("t")), int(m.group("k") or 0)


def rejoin_index(rec_xy, plan_xy, *, eps_m: float) -> int | None:
    """First recovery-segment index within eps_m of the original plan
    ("back on track"); None if never. ONE rule, no sustain window (Lukas
    2026-07-17: simple rules — every extra parameter is one more thing to
    tune and defend). Known trade, accepted: a transient path CROSSING
    during the corrective S-curve counts as rejoined and cuts the anchor
    span there.
    """
    import numpy as _np
    dev = point_to_polyline_dist(_np.asarray(rec_xy), _np.asarray(plan_xy))
    hits = _np.nonzero(dev < eps_m)[0]
    return int(hits[0]) if len(hits) else None


def find_excursions(xy, gt_polyline, *, dev_thresh_m: float,
                    aligned_eps_m: float) -> list[dict]:
    """ALL excursions of a rollout vs the GT route, with REJOIN semantics.

    An excursion opens when deviation exceeds dev_thresh_m and closes only when
    the rollout returns to within aligned_eps_m of the route (or the trace
    ends). A transient blip that rejoins is therefore its OWN excursion and
    cannot hijack the onset of a later, real failure — the first-crossing
    semantics of find_derailment cannot distinguish the two (2026-08-23
    analysis: 5/12 spawn-start rollouts had a spawn blip before the junction
    failure).

    Returns, per excursion, in ROLLOUT indices:
      last_aligned_idx  last step before onset with deviation <= aligned_eps_m
                        (0 if never), so the oracle-spawn band is
                        [last_aligned_idx, onset_idx] exactly as in
                        find_derailment.
      onset_idx         first step with deviation > dev_thresh_m
      end_idx           where the excursion closed (rejoin, or trace end)
      peak_dev_m        max deviation over [onset, end] — the SEVERITY of this
                        excursion, used to tell a real wrong-aisle commitment
                        from tracking slop (2026-08-24: 5.31-5.43 m vs
                        0.58-0.64 m on the BEfullrun probes).
      rejoined          True iff it closed by RETURNING to the route rather
                        than by the trace running out — i.e. the policy
                        self-corrected. Perfectly anti-correlated with
                        peak_dev_m >= 2.5 on that probe (12 vs 5, 0 exceptions),
                        which is what validates the threshold as non-arbitrary.
                        NOT used as the filter itself: an excursion truncated
                        by the rollout budget looks un-rejoined when it would
                        have self-corrected, whereas peak_dev_m is a property
                        of the excursion alone.
    """
    xy = np.asarray(xy, dtype=np.float64)
    dev = point_to_polyline_dist(xy, gt_polyline)
    out, i, last_aligned, n = [], 0, 0, len(dev)
    while i < n:
        if dev[i] <= aligned_eps_m:
            last_aligned = i
        if dev[i] > dev_thresh_m:
            onset = i
            j = i
            while j < n and dev[j] > aligned_eps_m:
                j += 1
            end = int(min(j, n - 1))
            out.append(dict(last_aligned_idx=int(last_aligned),
                            onset_idx=int(onset), end_idx=end,
                            peak_dev_m=float(dev[onset:end + 1].max()),
                            rejoined=bool(j < n and dev[end] <= aligned_eps_m)))
            i = j
        else:
            i += 1
    return out


def project_to_gt_indices(xy, gt_positions) -> np.ndarray:
    """Monotone nearest-GT-frame index for every rollout point.

    Monotonicity (search resumes at the previous match) guards against the
    projection snapping to a different pass where the route runs near itself.
    Used to express WHERE along the route an excursion happened — robust to
    timing drift, unlike comparing rollout step counts to GT frame numbers.
    """
    gt = np.asarray(gt_positions, dtype=np.float64)[:, :2]
    js, j0 = [], 0
    for pnt in np.asarray(xy, dtype=np.float64)[:, :2]:
        d = np.linalg.norm(gt[j0:] - pnt, axis=1)
        j0 = j0 + int(np.argmin(d))
        js.append(j0)
    return np.asarray(js, dtype=int)
