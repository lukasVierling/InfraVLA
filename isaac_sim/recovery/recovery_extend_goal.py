"""Extend stitched recovery trajectories to the original goal.

Takes a recovery output root (rec_* dirs from recovery.py) and writes a new
root <root>_extgoal/ where each trajectory = the stitched run + the SOURCE
trajectory's GT tail from the snippet endpoint to the original final goal.
Under final_goal_prob=1.0 the recovery samples' goal then IS the original
goal (deployment-matched), instead of the nearby snippet endpoint.

Seam: the tail starts at the source frame NEAREST the recovery's final pose
(searched from the sample step onward), so the frame-to-frame step across the
join stays at normal spacing — no label spike.

Non-destructive: the input root is untouched. Plain python, no sim.

Usage:
    python isaac_sim/recovery/recovery_extend_goal.py <recovery_output_root>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from isaac_sim.planning.a_star.goal_graph import GoalGraph, GraphNode  # noqa: E402


def extend_one(rec_dir: Path, split_dir: Path, out_root: Path) -> dict:
    src_name = rec_dir.name[len("rec_"):].rsplit("_t", 1)[0]
    t = int(rec_dir.name.rsplit("_t", 1)[1])
    src_dir = split_dir / src_name

    g_rec = GoalGraph.load(rec_dir / "frames.json")
    g_src = GoalGraph.load(src_dir / "frames.json")
    src_pos = np.array([n.xy for n in g_src.nodes])

    # Seam: source frame nearest the recovery's final pose, at/after the
    # sample step (the recovery ends within success_dist of gt[snippet_end]).
    end_xy = np.array(g_rec.nodes[-1].xy)
    dists = np.linalg.norm(src_pos[t:] - end_xy, axis=1)
    j = t + int(dists.argmin())
    seam_gap = float(dists.min())

    out_dir = out_root / rec_dir.name
    out_dir.mkdir(parents=True)
    cam_dirs = sorted(d.name for d in rec_dir.iterdir()
                      if d.is_dir() and any(f.suffix == ".png" for f in d.iterdir()))
    for cam in cam_dirs:
        (out_dir / cam).mkdir()

    # stitched frames first (from rec_dir), then the source tail j+1..end.
    sources = [(rec_dir, n) for n in g_rec.nodes] + \
              [(src_dir, n) for n in g_src.nodes[j + 1:]]
    nodes = []
    for i, (d, node) in enumerate(sources):
        new_name = f"{i:03d}.png"
        for cam in cam_dirs:
            shutil.copy2(d / cam / node.image_filename, out_dir / cam / new_name)
        nodes.append(GraphNode(xy=node.xy, yaw_rad=node.yaw_rad,
                               image_filename=new_name,
                               cmd_v=node.cmd_v, cmd_w=node.cmd_w))

    GoalGraph(env_config_name=g_rec.env_config_name,
              env_config_hash=g_rec.env_config_hash,
              node_spacing_m=0.0, robot_inflation_m=g_rec.robot_inflation_m,
              sampling_mode="recovery_stitched_extgoal",
              target_speed_mps=g_rec.target_speed_mps,
              sample_hz=g_rec.sample_hz, nodes=nodes).save(out_dir / "frames.json")

    res = json.loads((rec_dir / "result.json").read_text())
    res["n_frames"] = res["steps"] = len(nodes)
    res["stitched"]["n_gt_tail_frames"] = len(g_src.nodes) - (j + 1)
    res["stitched"]["seam_gap_m"] = round(seam_gap, 3)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    shutil.copy2(rec_dir / "astar_path.json", out_dir / "astar_path.json")
    shutil.copy2(rec_dir / "config.yaml", out_dir / "config.yaml")
    return dict(name=rec_dir.name, n_frames=len(nodes), seam_gap_m=seam_gap,
                n_tail=len(g_src.nodes) - (j + 1))


def main(root: Path) -> None:
    manifest = json.loads((root / "recovery_manifest.json").read_text())
    split_dir = Path(manifest["split_dir"])
    out_root = root.parent / (root.name + "_extgoal")
    if out_root.exists():
        raise SystemExit(f"{out_root} already exists — remove it first (non-destructive by design)")
    out_root.mkdir()

    stats = []
    for rec_dir in sorted(root.glob("rec_*")):
        s = extend_one(rec_dir, split_dir, out_root)
        stats.append(s)
        print(f"  {s['name']}: {s['n_frames']} frames "
              f"(+{s['n_tail']} GT tail, seam {s['seam_gap_m']:.2f} m)")

    manifest["extgoal"] = dict(
        source_root=str(root),
        note="stitched runs extended with the source GT tail to the original "
             "goal (recovery_extend_goal.py); seam = nearest source frame to "
             "the recovery endpoint",
        max_seam_gap_m=max(s["seam_gap_m"] for s in stats),
    )
    (out_root / "recovery_manifest.json").write_text(json.dumps(manifest, indent=2))
    gaps = [s["seam_gap_m"] for s in stats]
    print(f"\n{len(stats)} trajectories -> {out_root}")
    print(f"seam gaps: max {max(gaps):.3f} m, mean {np.mean(gaps):.3f} m "
          f"(frame spacing is 0.333 m — max should be well under one step)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
