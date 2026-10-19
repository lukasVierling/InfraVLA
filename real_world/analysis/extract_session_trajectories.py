
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recordings", required=True, help="dir of <take>/ bag dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="/odometry")
    args = ap.parse_args()

    from mcap_ros2.reader import read_ros2_messages   # container-only dependency

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    dirs = sorted(d for d in glob.glob(os.path.join(args.recordings, "*")) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no take directories under {args.recordings}")

    runs, frames, t_start = [], set(), time.time()
    for i, d in enumerate(dirs, 1):
        name = os.path.basename(d)
        rec = {"name": name, "prompt": None, "started_unix": None,
               "n_odom": 0, "t": [], "x": [], "y": [], "yaw": []}
        lab = os.path.join(d, "label.json")
        if os.path.exists(lab):
            L = json.load(open(lab))
            rec["prompt"] = (L.get("prompt") or "").strip()
            rec["started_unix"] = L.get("started_unix")
            rec["duration_s"] = L.get("duration_s")
        mcaps = sorted(glob.glob(os.path.join(d, "*.mcap")))
        if mcaps:
            try:
                for m in read_ros2_messages(mcaps[0], topics=[args.topic]):
                    r = m.ros_msg
                    frames.add(f"{r.header.frame_id}->{r.child_frame_id}")
                    p = r.pose.pose.position
                    rec["t"].append(m.log_time_ns / 1e9)
                    rec["x"].append(p.x); rec["y"].append(p.y)
                    rec["yaw"].append(yaw_of(r.pose.pose.orientation))
            except Exception as exc:                      # a truncated bag must not be silent
                rec["error"] = f"{type(exc).__name__}: {exc}"
        else:
            rec["error"] = "no .mcap in the take directory"
        rec["n_odom"] = len(rec["x"])
        runs.append(rec)
        print(f"[{i}/{len(dirs)}] {name:34s} prompt={rec['prompt']!r:22s} n={rec['n_odom']}"
              + (f"  !! {rec['error']}" if "error" in rec else ""), flush=True)

    dead = [r["name"] for r in runs if r["n_odom"] < 5]
    json.dump({"frames": sorted(frames), "runs": runs}, open(args.out, "w"))
    print(f"\nwrote {args.out}  ({len(runs)} takes, {time.time() - t_start:.1f}s)")
    print(f"frames seen: {sorted(frames)}")
    if dead:
        print(f"NOTE {len(dead)} take(s) carry <5 odometry samples and will be dropped "
              f"downstream: {', '.join(dead)}")


if __name__ == "__main__":
    main()
