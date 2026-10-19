
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

_REPO = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "viz/_lib"))
import cf_ftpair_decision_gallery as M
import cf_pooling_pred_gallery as G

ARM_A, COL_A = M.ARMS[0][0], "#1f77b4"     # upsampled
ARM_B, COL_B = M.ARMS[1][0], "#d62728"     # no upsampling
LBL_A, LBL_B = "cf-upsampled", "no upsampling"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,                    # embed TrueType -> editable in the PDF
})


def alpha_of(row, arm):
    g = M.final_wp_xy(row["gt"])[0]
    tw = M.final_wp_xy(row["twin_gt"])[0]
    p = M.final_wp_xy(row["preds"][arm])[0]
    return (p - tw) / (g - tw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="warehouse_2x3_1616")
    ap.add_argument("--split", default="train")
    ap.add_argument("--unsure", type=int, default=-1)
    ap.add_argument("--locked", type=int, default=1)
    ap.add_argument("--extra-tag", default=None,
                    help="overlay a third arm from <OUT>/<tag>/preds_<base>_<split>.json")
    ap.add_argument("--extra-label", default="obswap")
    ap.add_argument("--extra-color", default="#2ca02c")
    ap.add_argument("--show-gt", action="store_true",
                    help="draw the ground-truth chunks in grey under the predictions")
    args = ap.parse_args()

    root = M.OUT / M.PAIR_TAG
    fam = json.loads((root / f"preds_{args.base}_{args.split}.json").read_text())
    # optional third arm from a separate cache tag; merged into each row's preds
    if args.extra_tag:
        efam = json.loads((M.OUT / args.extra_tag /
                           f"preds_{args.base}_{args.split}.json").read_text())
        for suf in efam:
            erows = {r["off"]: r for r in efam[suf]["rows"]}
            for r in fam[suf]["rows"]:
                er = erows.get(r["off"])
                if er:
                    # single-arm cache -> exactly one key in er["preds"]
                    (ename, chunk), = er["preds"].items()
                    r["preds"][args.extra_label] = chunk
    sufs = sorted(fam)                     # e.g. ['B', 'E']
    roles = {s: fam[s]["role"] for s in sufs}
    style = {s: ("-" if roles[s] == "turn" else "--") for s in sufs}
    by = {s: {r["off"]: r for r in fam[s]["rows"]} for s in sufs}
    t_star = fam[sufs[0]]["t_star"]

    fig = plt.figure(figsize=(7.6, 3.9))
    # [imgs | plot | gap | imgs | plot]; small spacer column keeps the two
    # panels visually separate without a frame
    gs = fig.add_gridspec(1, 5, width_ratios=[1.05, 1.0, 0.06, 1.05, 1.0],
                          wspace=0.26, left=0.012, right=0.985,
                          top=0.90, bottom=0.22)

    panels = [("(a)", args.unsure, 0),
              ("(b)", args.locked, 3)]
    for tag, off, c0 in panels:
        rows = {s: by[s][off] for s in sufs}
        t = rows[sufs[0]]["t"]

        # --- ego views, stacked, label ABOVE each image (no cf codes) ---
        sub = gs[0, c0].subgridspec(2, 1, hspace=0.30)
        for j, suf in enumerate(sufs):
            ax = fig.add_subplot(sub[j, 0])
            p = pathlib.Path(fam[suf]["traj_dir"]) / "frames" / f"{rows[suf]['t']:03d}.png"
            ax.imshow(np.asarray(Image.open(p)))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_edgecolor("0.35")
            lab = ("shortcut open" if roles[suf] == "turn" else "shortcut sealed")
            ax.set_title(lab, fontsize=7.5, pad=2)

        # --- predictions ---
        ax = fig.add_subplot(gs[0, c0 + 1])
        xs, ys = [], []
        arm_list = [(ARM_A, COL_A), (ARM_B, COL_B)]
        if args.extra_tag:
            arm_list.append((args.extra_label, args.extra_color))
        if args.show_gt:
            for suf in sufs:
                gf, gl = G.chunk_to_local_m(
                    np.asarray(rows[suf]["gt"], dtype=np.float32))
                ax.plot(-gl, gf, style[suf], color="0.62", lw=2.2, zorder=1)
                xs.append(-gl); ys.append(gf)
        for suf in sufs:
            for arm, col in arm_list:
                pf, pl = G.chunk_to_local_m(
                    np.asarray(rows[suf]["preds"][arm], dtype=np.float32))
                ax.plot(-pl, pf, style[suf], color=col, lw=1.5)
                xs.append(-pl); ys.append(pf)
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        ax.plot(0, 0, marker="^", color="black", ms=5, zorder=5)   # the robot
        ax.set_xlim(min(-0.35, xs.min() - 0.15), max(1.05, xs.max() + 0.15))
        ax.set_ylim(-0.18, ys.max() + 0.25)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_xlabel("lateral (m)", fontsize=8)
        if c0 == 0:
            ax.set_ylabel("forward (m)", fontsize=8)
        ax.set_title(rf"{tag}  $t = {rows[sufs[0]]['t']}$", fontsize=9, pad=4)
    # --- one shared legend: all four model x twin combinations, 2x2 ---
    pairs = [(LBL_A, COL_A), (LBL_B, COL_B)]
    if args.extra_tag:
        pairs.append((args.extra_label, args.extra_color))
    if args.show_gt:
        pairs.append(("ground truth", "0.62"))
    handles = []
    for lbl, col in pairs:
        handles.append(Line2D([], [], color=col, lw=1.6, ls="-",
                              label=f"{lbl} — shortcut open"))
        handles.append(Line2D([], [], color=col, lw=1.6, ls="--",
                              label=f"{lbl} — shortcut sealed"))
    fig.legend(handles=handles, loc="lower center", ncol=len(pairs),
               frameon=False, fontsize=7.5, columnspacing=2.0,
               handlelength=2.4, bbox_to_anchor=(0.5, 0.005))

    suffix = f"_plus_{args.extra_label}" if args.extra_tag else ""
    out = root / f"lockin_paper_{args.base}{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
        print(f"wrote {out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
