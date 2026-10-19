
from __future__ import annotations

import argparse, math, pathlib, sys

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_REPO = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "viz/_lib")); sys.path.insert(0, str(_REPO / "vla-scripts"))

import cf_counterfactual_probe as P
import train_omnivla_sim_cctv_v2 as T2
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_CCTV
from isaac_sim.utils.cf_pairing import (load_pose_stream, decision_frame,
                                        group_cf_families)

OUT = _REPO / "isaac_sim/outputs/pooling_pred_gallery"
SPLIT_ROOT = _REPO / "runs/data/splits/2026-07-11_warehouse_2x3_cfv4"
RUNS = _REPO / "runs/training/runs"
CFGS = _REPO / "isaac_sim/train/configs"

_AP64 = ("avgpool64", "#1f77b4",
         CFGS / "2026-07-11_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_cfwinboost8_15k.yaml",
         RUNS / "2026-07-11_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_cfwinboost8_15k_v0/checkpoints/step_15000")
PRESETS = {
    "pooling": [
        _AP64,
        ("avgpool16", "#d62728",
         CFGS / "2026-07-24_cctv4_sharedenc_2x3cfv4_egogoal_avgpool16_r16_fg50_cfwinboost8_15k.yaml",
         RUNS / "2026-07-24_cctv4_sharedenc_2x3cfv4_egogoal_avgpool16_r16_fg50_cfwinboost8_15k_v0/checkpoints/step_15000"),
    ],
    "upsampling": [
        ("WITH upsampling", "#1f77b4", _AP64[2], _AP64[3]),
        ("NO upsampling", "#d62728",
         CFGS / "2026-07-30_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_noups_15k.yaml",
         RUNS / "2026-07-30_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_noups_15k_v0/checkpoints/step_15000"),
    ],
}
ARMS = PRESETS["pooling"]        # overridden in main() from --preset
GT_COL = "green"
WPS = 0.333                     # metric_waypoint_spacing: chunk units -> metres
CAMS = ["cctv_NW", "cctv_NE", "cctv_SW", "cctv_SE"]
NCHUNK, DEV_THRESH = 8, 0.75
# PRE-CRUX ramp: twin GT delta grows ~0.02 -> ~0.27 m over t*-6..t*-1 before
# jumping to ~1.4 m at t*. Those small-delta frames are where the commit is
# formed and where derailing is decided; later frames are already determined.
ROW_OFFSETS = [-8, -6, -5, -4, -3, -2, -1, 0]
ZB = {"B": ((-6.155, -2.295), (8.8, 25.0)), "C": ((-1.2, 2.66), (8.8, 25.0)),
      "E": ((-6.155, -2.295), (-15.2, 1.0)), "F": ((-1.2, 2.66), (-15.2, 1.0))}


def chunk_to_local_m(chunk):
    """Cumulative normalized robot-frame chunk (N,4) -> (fwd_m, left_m) arrays,
    with the robot's own origin prepended so the polyline starts at (0,0)."""
    fwd = np.concatenate([[0.0], chunk[:, 0] * WPS])
    left = np.concatenate([[0.0], chunk[:, 1] * WPS])
    return fwd, left


def chunk_to_world(chunk, x, y, yaw):
    fwd, left = chunk_to_local_m(chunk)
    c, s = math.cos(yaw), math.sin(yaw)
    return x + fwd * c - left * s, y + fwd * s + left * c


def cctv_tile(traj_dir, t, px=180):
    tiles = []
    for cam in CAMS:
        p = traj_dir / cam / f"{t:03d}.png"
        im = (Image.open(p).convert("RGB").resize((px, px)) if p.exists()
              else Image.new("RGB", (px, px), (40, 40, 40)))
        tiles.append(im)
    canvas = Image.new("RGB", (2 * px + 4, 2 * px + 4), (255, 255, 255))
    for k, im in enumerate(tiles):
        canvas.paste(im, ((k % 2) * (px + 4), (k // 2) * (px + 4)))
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--preset", default="pooling", choices=sorted(PRESETS),
                    help="which pair of arms to compare")
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--families", default=None,
                    help="comma-separated family bases, e.g. warehouse_2x3_1753")
    ap.add_argument("--variants", default=None,
                    help="comma-separated cf suffixes to pair, e.g. B,BE")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    global ARMS, OUT
    ARMS = PRESETS[args.preset]
    OUT = OUT.parent / f"pred_gallery_{args.preset}"
    cfgs = [yaml.safe_load(c.read_text()) for _, _, c, _ in ARMS]
    if cfgs[0]["data"] != cfgs[1]["data"]:
        diff = {k: (cfgs[0]["data"].get(k), cfgs[1]["data"].get(k))
                for k in set(cfgs[0]["data"]) | set(cfgs[1]["data"])
                if cfgs[0]["data"].get(k) != cfgs[1]["data"].get(k)}
        print(f"[warn] arms differ in data cfg (not pooling-only!): {diff}")
    print(f"[gallery] out_tokens: "
          f"{[c['model']['cctv_downsample']['out_tokens'] for c in cfgs]}")

    models = []
    for (name, col, _cfgp, ckpt), cfg in zip(ARMS, cfgs):
        print(f"[gallery] loading {name} <- {ckpt}", flush=True)
        bits = P.build_model(cfg, ckpt, 15000, args.device)
        models.append((name, col, cfg, bits))

    # ONE dataset (data cfgs identical) -> both models see bit-identical inputs.
    proc0 = models[0][3][0]
    VAL = SPLIT_ROOT / args.split
    ds = P.build_dataset(cfgs[0], proc0, VAL, int(cfgs[0]["train"].get("seed", 42)))
    collator = PaddedCollatorForActionPrediction_Nav_CCTV(
        proc0.tokenizer.model_max_length, proc0.tokenizer.pad_token_id,
        padding_side="right", num_img=cfgs[0]["model"]["num_images_in_input"])
    pad_id, max_len = proc0.tokenizer.pad_token_id, proc0.tokenizer.model_max_length
    name_to_ti = P._name_to_traj_index(ds)

    # counterfactual families in val, with their decision frame
    dirs = sorted(d for d in VAL.iterdir() if d.is_dir())
    names = [d.name for d in dirs]
    fam, _ = group_cf_families(names)
    streams = {d.name: load_pose_stream(d) for d in dirs}
    picks = []
    want = set(args.families.split(",")) if args.families else None
    want_v = args.variants.split(",") if args.variants else None
    for base, members in sorted(fam.items()):
        if len(members) < 2:
            continue
        if want is not None and base not in want:
            continue
        suf = sorted(members)
        if want_v is not None and all(v in members for v in want_v):
            a, b = want_v[0], want_v[1]
        else:
            a, b = suf[0], suf[1]
        da, db = dirs[members[a]], dirs[members[b]]
        pa, _, _ = streams[da.name]
        pb, _, _ = streams[db.name]
        t_star, _, _ = decision_frame(pa, pb, n_chunk=NCHUNK, dev_thresh_m=DEV_THRESH)
        if t_star is None:
            continue
        for suffix, d, twin_s, twin_d in ((a, da, b, db), (b, db, a, da)):
            picks.append((base, suffix, d, t_star, twin_s, twin_d))
        if len({p[0] for p in picks}) >= args.scenes:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    for base, suffix, tdir, t_star, twin_suffix, twin_dir in picks:
        ti = name_to_ti.get(tdir.name)
        if ti is None:
            print(f"[skip] {tdir.name} not in dataset"); continue
        pos, yaws, _ = streams[tdir.name]
        frames = []
        for o in ROW_OFFSETS:
            t = t_star + o
            if (ti, t) in ds.samples_index:
                frames.append(t)
        if not frames:
            continue

        rows = []
        for t in frames:
            s = P.sample_pose_final_goal(ds, ti, t)
            batch = T2.pad_and_mask_batch(collator([s]), pad_id, max_len)
            gt = np.asarray(s["actions"], dtype=np.float32)
            tw_ti = name_to_ti.get(twin_dir.name)
            twin_gt = None
            if tw_ti is not None and (tw_ti, t) in ds.samples_index:
                twin_gt = np.asarray(
                    P.sample_pose_final_goal(ds, tw_ti, t)["actions"], dtype=np.float32)
            preds = {}
            for name, col, cfg, bits in models:
                processor, vla, pp, ah, npat, app = bits
                preds[name] = P.predict_chunk(batch, vla, pp, ah, npat,
                                              args.device, abs_pose_projector=app)
            rows.append((t, gt, preds, twin_gt))

        seal = "none (cf0)" if suffix == "0" else f"aisle {'+'.join(suffix)}"
        n = len(rows)
        fig, axes = plt.subplots(n, 4, figsize=(17, 3.5 * n),
                                 gridspec_kw={"width_ratios": [1.15, 1.15, 1, 1]})
        axes = np.atleast_2d(axes)
        print(f"\n{base} cf{suffix} ({seal})  t*={t_star}  frames={frames}")
        print(f"  {'frame':>5} | {'off':>4} | " +
              " | ".join(f"{nm} final-wp err (m)" for nm, _, _, _ in models))
        for r, (t, gt, preds, twin_gt) in enumerate(rows):
            p = tdir / "frames" / f"{t:03d}.png"
            if p.exists():
                axes[r, 0].imshow(np.asarray(Image.open(p)))
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            axes[r, 0].set_title(f"ego @ frame {t}", fontsize=9)
            axes[r, 1].imshow(cctv_tile(tdir, t))
            axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
            axes[r, 1].set_title("CCTV NW|NE / SW|SE  (the disambiguating input)",
                                 fontsize=9)

            ax = axes[r, 2]
            gf, gl = chunk_to_local_m(gt)
            if twin_gt is not None:
                tf, tl = chunk_to_local_m(twin_gt)
                ax.plot(-tl, tf, "--", color="0.55", lw=2.0, zorder=2,
                        label=f"GT of twin cf{twin_suffix}")
            ax.plot(-gl, gf, "-o", color=GT_COL, ms=4, lw=2.6, label="GT", zorder=3)
            errs = []
            for name, col, _c, _b in models:
                pf, pl = chunk_to_local_m(preds[name])
                ax.plot(-pl, pf, "-o", color=col, ms=3.5, lw=2.0, alpha=.9, label=name)
                errs.append(float(np.hypot(pf[-1] - gf[-1], pl[-1] - gl[-1])))
            ax.axhline(0, color="0.9", lw=.8); ax.axvline(0, color="0.9", lw=.8)
            xs = [-gl]; ys = [gf]
            for nm, _c2, _c3, _c4 in models:
                pf2, pl2 = chunk_to_local_m(preds[nm]); xs.append(-pl2); ys.append(pf2)
            if twin_gt is not None:
                tf2, tl2 = chunk_to_local_m(twin_gt); xs.append(-tl2); ys.append(tf2)
            xs = np.concatenate(xs); ys = np.concatenate(ys)
            mx = max(0.35, 0.15 * (xs.max() - xs.min() + ys.max() - ys.min()))
            ax.set_aspect("equal")
            ax.set_xlim(xs.min() - mx, xs.max() + mx)
            ax.set_ylim(ys.min() - mx, ys.max() + mx)
            ax.legend(fontsize=7, loc="lower left"); ax.tick_params(labelsize=7)
            ax.set_xlabel("left <- m -> right", fontsize=7)
            ax.set_ylabel("forward (m)", fontsize=7)
            twin_sep = float("nan")
            if twin_gt is not None:
                tf2, tl2 = chunk_to_local_m(twin_gt)
                twin_sep = float(np.hypot(tf2[-1] - gf[-1], tl2[-1] - gl[-1]))
            ax.set_title(f"@ {t} (t*{t - t_star:+d}) | TWIN delta {twin_sep:.2f} m | err "
                         + " / ".join(f"{nm}={e:.2f}" for (nm, _, _, _), e in zip(models, errs)),
                         fontsize=8.5, color=("red" if t == t_star else "0.3"))
            print(f"  {t:>5} | {t - t_star:>+4d} | " +
                  " | ".join(f"{e:>17.2f}" for e in errs))

            axm = axes[r, 3]
            for z, ((x0, x1), (y0, y1)) in ZB.items():
                axm.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="0.8"))
                axm.text((x0 + x1) / 2, (y0 + y1) / 2, z, color="0.8", ha="center", fontsize=7)
            axm.plot(pos[:, 0], pos[:, 1], "-", color="0.6", lw=1.1, alpha=.8)
            gx, gy = chunk_to_world(gt, pos[t, 0], pos[t, 1], yaws[t])
            axm.plot(gx, gy, "-", color=GT_COL, lw=2.4, zorder=3)
            for name, col, _c, _b in models:
                wx, wy = chunk_to_world(preds[name], pos[t, 0], pos[t, 1], yaws[t])
                axm.plot(wx, wy, "-", color=col, lw=1.9, alpha=.9)
            axm.plot(pos[t, 0], pos[t, 1], "o", color="black", ms=6)
            axm.set_xlim(-9, 5); axm.set_ylim(-21, 28); axm.set_aspect("equal")
            axm.tick_params(labelsize=6); axm.set_title(f"world @ {t}", fontsize=9)

        fig.suptitle(
            f"{base} cf{suffix} — sealed {seal}   |   CFV4 VAL (unseen)   |   "
            f"decision frame t*={t_star}\n"
            f"GT (green) vs {ARMS[0][0]} (blue) vs {ARMS[1][0]} (red); "
            "dashed grey = the TWIN's GT\n"
            "(gap between green and dashed = what CCTV must resolve). "
            "Identical inputs; the two arms differ only in the ablated knob.",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fp = OUT / f"pred_{base}_cf{suffix}_tstar{t_star}.png"
        fig.savefig(fp, dpi=95); plt.close(fig)
        print(f"wrote {fp}", flush=True)


if __name__ == "__main__":
    main()
