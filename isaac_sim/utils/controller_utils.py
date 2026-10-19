# Shared controller-side helpers used by both eval_policy.py (VLA rollouts)
# and data_gen.py (A* oracle rollouts). Kept deliberately small — only
# things that need to behave identically across the two rollout drivers
# belong here.

import math


def _sign(a):
    # np.sign semantics WITHOUT importing numpy: -1 / 0 / +1. The saturation
    # block below relies on sign(0)==0 (a zero linear vel must stay zero after
    # the turn-radius rescale), so plain math.copysign (which has no zero) would
    # be wrong here.
    return (a > 0) - (a < 0)


def world_to_robot(tx, ty, x, y, yaw):
    """World-frame target (tx, ty) -> robot-frame (dx, dy) at pose (x, y, yaw).

    dx = forward, dy = left. This is the rotation R(-yaw) applied to the world
    error (R orthogonal so inverse = transpose). e.g. robot facing +x at origin,
    target (1, 0) -> (dx, dy) = (1, 0); target (0, 1) -> (0, 1) (1 m to the left).
    """
    dx_w = tx - x
    dy_w = ty - y
    c, s = math.cos(yaw), math.sin(yaw)
    return dx_w * c + dy_w * s, -dx_w * s + dy_w * c


def pure_pursuit_target(path, pos, cursor, lookahead_m):
    """Two-phase pure-pursuit lookup over an (x, y) polyline. Returns
    (target_xy, new_cursor).

    Phase 1 (ratchet): advance `cursor` while the NEXT node is closer than the
    current one — monotonic arc-length progress that never decreases, so it's
    immune to path self-loops. Phase 2 (lookahead): return the first node at or
    beyond `lookahead_m` from `pos`; if the whole remaining path is inside the
    radius, return the final node. Squared distances throughout (no sqrt).

    Shared by the A* oracle (data_gen) and the VLA follower (eval) so both track
    a path with identical geometry — only the path SOURCE differs.
    """
    x, y = pos
    last = len(path) - 1

    # Phase 1 — ratchet the cursor forward.
    while cursor < last:
        dx_here = path[cursor][0] - x
        dy_here = path[cursor][1] - y
        dx_next = path[cursor + 1][0] - x
        dy_next = path[cursor + 1][1] - y
        if (dx_next * dx_next + dy_next * dy_next) < (dx_here * dx_here + dy_here * dy_here):
            cursor += 1
        else:
            break

    # Phase 2 — first node outside the lookahead radius.
    look2 = lookahead_m * lookahead_m
    j = cursor
    while j < last:
        dx = path[j][0] - x
        dy = path[j][1] - y
        if (dx * dx + dy * dy) >= look2:
            return path[j], cursor
        j += 1
    return path[last], cursor


def solve_twist(dx, dy, horizon_s, eps, maxv, maxw):
    """Robot-frame target offset (dx, dy) -> diff-drive twist (v, w).

    "Reach the target in `horizon_s` seconds." horizon_s is the time to traverse
    the lookahead at cruise speed (lookahead_m / max_v), NOT one control tick —
    using a tick here at high control rates demands ~50 m/s and pins w to its cap
    (limit-cycle / zigzag). Forward-only: if the target is behind (dx<=0) we turn
    in place toward it (v=0) rather than reversing; atan2(dy,dx) makes the turn
    go the right way for dx<0. Caller passes the (already arrival-braked, if any)
    maxv/maxw.

    The saturation is a JOINT clamp that preserves the turn radius rd = v/w, so a
    capped command still aims at the same arc — not an independent per-axis clip.
    e.g. dx=2, dy=0, horizon_s=2, maxv=1 -> v=1 (capped from 1.0), w=0.
    """
    if abs(dx) < eps and abs(dy) < eps:
        # Exactly on the target — pure-pursuit never lands here (target is
        # lookahead_m ahead by construction), so just hold still.
        v = 0.0
        w = 0.0
    elif abs(dx) < eps:
        v = 0.0
        w = _sign(dy) * math.pi / (2.0 * horizon_s)
    else:
        v = dx / horizon_s if dx > 0.0 else 0.0
        w = math.atan2(dy, dx) / horizon_s

    # Joint saturation (preserves rd = v/w). Copied math from the paper's
    # run_omnivla.py controller; shared so oracle + follower clamp identically.
    if abs(v) <= maxv:
        if abs(w) <= maxw:
            return v, w
        rd = v / w
        return maxw * _sign(v) * abs(rd), maxw * _sign(w)
    else:
        if abs(w) <= 0.001:
            return maxv * _sign(v), 0.0
        rd = v / w
        if abs(rd) >= maxv / maxw:
            return maxv * _sign(v), maxv * _sign(w) / abs(rd)
        else:
            return maxw * _sign(v) * abs(rd), maxw * _sign(w)


def is_stuck(recent_poses, step, window, bbox_m, grace_steps):
    # Bounding-box semantics: if the xy extent over the last `window` steps
    # stays inside an axis-aligned `bbox_m` square, the robot is stuck.
    # Tolerates slow turns-in-place (no net displacement) without false-
    # positiving on a single tiny step. Returns False during the initial
    # `grace_steps` so the empty/short pose buffer never trips at t=0.
    if step < grace_steps or len(recent_poses) < window:
        return False
    xs = [p[0] for p in recent_poses]
    ys = [p[1] for p in recent_poses]
    # e.g. recent_poses = [(0.10, 0.20), (0.11, 0.21), (0.10, 0.22)]
    #      max(xs)-min(xs) = 0.01, max(ys)-min(ys) = 0.02 — both < 0.20 → stuck.
    return (max(xs) - min(xs)) < bbox_m and (max(ys) - min(ys)) < bbox_m
