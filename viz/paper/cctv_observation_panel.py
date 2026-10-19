
from __future__ import annotations

import pathlib
import sys

import numpy as np
import hydra

_REPO = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "isaac_sim"))

OUT = _REPO / "isaac_sim" / "outputs" / "cctv_observation_panel"

# Map-consistent 2x2 arrangement (north up): row 0 = north wall cams, row 1 =
# south wall cams; west column left.
PANEL = [["cctv_NW", "cctv_NE"],
         ["cctv_SW", "cctv_SE"]]

INK = "#0b0b0b"


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg):
    from isaac_sim.sim.isaac_sim_env import IsaacSimEnv
    from omegaconf import OmegaConf
    # scene yaml is `# @package env` -> its keys sit directly under cfg.env
    assert not OmegaConf.select(cfg, "env.obstacles", default=None), \
        "cctv_observation_panel expects the BARE scene (env/scene=warehouse_2x3_bare)"
    print("[obs_panel] building env (boots Isaac)...", flush=True)
    env = IsaacSimEnv(cfg.env, record_hz=cfg.record_hz, physics_hz=cfg.physics_hz)
    env.reset()
    # A few warm-up render steps so every sensor buffer is filled before capture.
    for _ in range(30):
        env.world.step(render=True)
    obs = env.get_observations(capture_images=True)
    images = obs["images"]
    names = [n for row in PANEL for n in row]
    missing = [n for n in names if n not in images]
    assert not missing, f"camera(s) {missing} not in observations: {list(images)}"

    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    for n in names + [k for k in ("front_cam",) if k in images]:
        Image.fromarray(np.asarray(images[n])).save(OUT / f"{n}.png")

    # --- the 2x2 panel ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.8), facecolor="white",
                             gridspec_kw=dict(wspace=0.02, hspace=0.02))
    for r in range(2):
        for c in range(2):
            name = PANEL[r][c]
            ax = axes[r][c]
            ax.imshow(np.asarray(images[name]))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(0.03, 0.97, name, transform=ax.transAxes, fontsize=8.5,
                    color=INK, ha="left", va="top", family="monospace",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                              boxstyle="round,pad=0.25"))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"cctv_panel_2x2.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    scene_name = OmegaConf.select(cfg, "env.name", default="?")
    focal = OmegaConf.select(cfg, "env.cctv_NE.intrinsics.focal_length_mm", default="?")
    ap = OmegaConf.select(cfg, "env.cctv_NE.intrinsics.horizontal_aperture_mm", default="?")
    print(f"[obs_panel] scene={scene_name}  cams={names}", flush=True)
    print(f"[obs_panel] intrinsics: focal={focal} mm, aperture={ap} mm", flush=True)
    print(f"[out] {OUT}/cctv_panel_2x2.png", flush=True)
    print()
    print("--- suggested caption clause ---")
    print("Observations of the four fixed CCTV cameras (16 mm, 720x720), arranged "
          "by mount corner (north up: NW/NE top row, SW/SE bottom row). Each camera "
          "is mounted in a corner of the hall at 7.9 m height and pitched into the "
          "interior. The scene contains no task objects; the robot is visible at "
          "its spawn in the north strip (zone A) in the two north views. The "
          "camera-to-aisle responsibility used elsewhere is an imposed convention "
          "of our setup, not calibrated geometry.")
    env.close()


if __name__ == "__main__":
    main()
