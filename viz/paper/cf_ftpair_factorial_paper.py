
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "viz/_lib")); sys.path.insert(0, str(_REPO / "vla-scripts"))
import cf_counterfactual_probe as P
import train_omnivla_sim_cctv_v2 as T2
import cf_ftpair_decision_gallery as M
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_CCTV

OUT = M.OUT / "factorial"

ARMS = [
    ("cf-upsampled", "#1f77b4",
     M.CFGS / "2026-07-11_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_cfwinboost8_15k.yaml",
     M.RUNS / "2026-07-11_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_cfwinboost8_15k_v0/checkpoints/step_15000",
     15000),
    ("no upsampling", "#d62728",
     M.CFGS / "2026-07-30_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_noups_15k.yaml",
     M.RUNS / "2026-07-30_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_noups_15k_v0/checkpoints/step_15000",
     15000),
    ("obswap", "#2ca02c",
     M.CFGS / "2026-07-12_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg100_cfwin_obswap_ft20k.yaml",
     M.RUNS / "2026-07-12_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg100_cfwin_obswap_ft20k_v0/checkpoints/step_20000",
     20000),
]

# swap conditions: which channels come from the SEALED twin. Order = single
# swaps first, then pairs, then all — reads left-to-right as increasing dose.
CONDS = [((), "none"),
         (("pose",), "pose"), (("ego",), "ego"), (("cctv",), "CCTV"),
         (("pose", "ego"), "pose+ego"), (("pose", "cctv"), "pose+CCTV"),
         (("ego", "cctv"), "ego+CCTV"),
         (("pose", "ego", "cctv"), "all")]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "pdf.fonttype": 42,
})


def lateral(chunk):
    return M.final_wp_xy(np.asarray(chunk, np.float32))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="warehouse_2x3_1616")
    ap.add_argument("--split", default="train")
    ap.add_argument("--open-suf", default="B", help="the twin whose shortcut is open")
    ap.add_argument("--sealed-suf", default="E")
    ap.add_argument("--frames", default="53,55")
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    frames = [int(v) for v in args.frames.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"factorial_{args.base}_{args.split}.json"

    if args.from_cache:
        data = json.loads(cache.read_text())
    else:
        cfgs = [yaml.safe_load(c.read_text()) for _, _, c, _, _ in ARMS]
        models = []
        for (name, _col, _c, ckpt, step), cfg in zip(ARMS, cfgs):
            print(f"[load] {name} <- {ckpt}", flush=True)
            models.append((name, P.build_model(cfg, ckpt, step, args.device)))
        proc0 = models[0][1][0]
        root = M.SPLIT_ROOT / args.split
        ds = P.build_dataset(cfgs[0], proc0, root, int(cfgs[0]["train"].get("seed", 42)))
        coll = PaddedCollatorForActionPrediction_Nav_CCTV(
            proc0.tokenizer.model_max_length, proc0.tokenizer.pad_token_id,
            padding_side="right", num_img=cfgs[0]["model"]["num_images_in_input"])
        pad, mx = proc0.tokenizer.pad_token_id, proc0.tokenizer.model_max_length
        n2t = P._name_to_traj_index(ds)
        tis = {}
        for nm, ti in n2t.items():
            if nm.startswith(f"{args.base}_cf{args.open_suf}__"):
                tis["open"] = ti
            if nm.startswith(f"{args.base}_cf{args.sealed_suf}__"):
                tis["sealed"] = ti
        assert len(tis) == 2, f"twins not found: {tis}"

        data = {"frames": {}}
        for t in frames:
            s = {"open": P.sample_pose_final_goal(ds, tis["open"], t),
                 "sealed": P.sample_pose_final_goal(ds, tis["sealed"], t)}

            def hybrid(swaps):
                # base = the OPEN twin's sample; listed channels come from SEALED
                h = dict(s["sealed"] if "ego" in swaps else s["open"])
                h["goal_pose"] = s["sealed" if "pose" in swaps else "open"]["goal_pose"]
                h["cctv_pixel_values_list"] = \
                    s["sealed" if "cctv" in swaps else "open"]["cctv_pixel_values_list"]
                return h

            rec = {"gt_open_chunk": np.asarray(s["open"]["actions"], np.float32).tolist(),
                   "gt_sealed_chunk": np.asarray(s["sealed"]["actions"], np.float32).tolist(),
                   "conds": {}}
            for swaps, lbl in CONDS:
                batch = T2.pad_and_mask_batch(coll([hybrid(swaps)]), pad, mx)
                rec["conds"][lbl] = {}
                with torch.inference_mode():
                    for name, bits in models:
                        _p, vla, pp, ah, npat, app = bits
                        chunk = P.predict_chunk(batch, vla, pp, ah, npat,
                                                args.device, abs_pose_projector=app)
                        rec["conds"][lbl][name] = chunk.tolist()
                print(f"  t={t} {lbl:>9}: " + "  ".join(
                    f"{n}={lateral(rec['conds'][lbl][n]):+.2f}" for n, _ in models), flush=True)
            data["frames"][str(t)] = rec
        cache.write_text(json.dumps(data, indent=1))
        print(f"cached {cache}")

    # ------- figure: rows = frames; per row [ego frames | 4 swap panels] -------
    # Each swap panel shows the THREE models' predictions under that condition
    # (colour = model, as in the lock-in figure) + the two GT branches in grey.
    import cf_pooling_pred_gallery as G
    from PIL import Image

    SWAP_COLS = [("all",  "sealed scene"),
                 ("CCTV", "CCTV sealed\nego open"),
                 ("ego",  "CCTV open\nego sealed"),
                 ("none", "open scene")]

    # twin trajectory dirs (for the onboard frames)
    root = M.SPLIT_ROOT / args.split
    tdirs = {}
    for d in root.iterdir():
        if d.name.startswith(f"{args.base}_cf{args.open_suf}__"):
            tdirs["open"] = d
        if d.name.startswith(f"{args.base}_cf{args.sealed_suf}__"):
            tdirs["sealed"] = d

    def draw_chunk(ax, chunk, color, ls="-", lw=1.5, z=3):
        f, l = G.chunk_to_local_m(np.asarray(chunk, np.float32))
        ax.plot(-l, f, ls, color=color, lw=lw, zorder=z)

    nrow = len(frames)
    fig = plt.figure(figsize=(7.4, 2.9 * nrow + 0.5))
    gs = fig.add_gridspec(nrow, 6, width_ratios=[0.85, 0.14, 1, 1, 1, 1],
                          wspace=0.12, hspace=0.22,
                          left=0.055, right=0.99, top=0.93, bottom=0.17)
    for r, t_ in enumerate(frames):
        rec = data["frames"][str(t_)]

        # --- the two onboard views, stacked ---
        sub = gs[r, 0].subgridspec(2, 1, hspace=0.25)
        for j, (key, lab) in enumerate((("open", "shortcut open"),
                                        ("sealed", "shortcut sealed"))):
            ax = fig.add_subplot(sub[j, 0])
            img = tdirs[key] / "frames" / f"{t_:03d}.png"
            ax.imshow(np.asarray(Image.open(img)))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(lab, fontsize=7, pad=1.5)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_edgecolor("0.35")
            if j == 0:
                ax.text(-0.16, 0.0, rf"$t = {t_}$", transform=ax.transAxes,
                        rotation=90, ha="right", va="center", fontsize=10)

        # --- the four swap panels ---
        for c, (lbl, title) in enumerate(SWAP_COLS):
            ax = fig.add_subplot(gs[r, c + 2])
            if lbl == "none":
                draw_chunk(ax, rec["gt_open_chunk"], "0.68", lw=2.4, z=1)
            elif lbl == "all":
                draw_chunk(ax, rec["gt_sealed_chunk"], "0.68", lw=2.4, z=1)
            for name, col, *_rest in ARMS:
                draw_chunk(ax, rec["conds"][lbl][name], col, lw=1.5, z=3)
            ax.plot(0, 0, marker="^", color="black", ms=4.5, zorder=5)
            ax.set_aspect("equal")
            ax.set_xlim(-0.35, 1.1); ax.set_ylim(-0.18, 2.95)
            ax.grid(alpha=0.22, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(title, fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel("forward (m)", fontsize=8)
            else:
                ax.set_yticklabels([])
            if r == nrow - 1:
                ax.set_xlabel("lateral (m)", fontsize=8)
            else:
                ax.set_xticklabels([])

    handles = [Line2D([], [], color=c, lw=1.7, label=n) for n, c, *_ in ARMS]
    handles += [Line2D([], [], color="0.68", lw=2.2,
                       label="ground truth (pure scenes only)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7.5, columnspacing=1.5, handlelength=2.0,
               bbox_to_anchor=(0.5, 0.01))
    out = OUT / f"factorial_chunks_paper_{args.base}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
        print(f"wrote {out}.{ext}")


if __name__ == "__main__":
    main()
