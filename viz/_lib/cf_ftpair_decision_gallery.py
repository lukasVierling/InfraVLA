
from __future__ import annotations

import argparse
import json
import pathlib
import sys

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
from isaac_sim.utils.cf_pairing import load_pose_stream, decision_frame, group_cf_families
# reuse the geometry / tiling helpers so there is ONE definition of each
import cf_pooling_pred_gallery as G

OUT = _REPO / "isaac_sim/outputs/cf_ftpair_decision"
SPLIT_ROOT = _REPO / "runs/data/splits/2026-07-11_warehouse_2x3_cfv4"
RUNS = _REPO / "runs/training/runs"
CFGS = _REPO / "isaac_sim/train/configs"

# Which ablation this run compares. Outputs are namespaced by it so caches from
# a different arm pair can never be silently reused.
PAIR_TAG = "upsampling"

# (label, colour, train-config, checkpoint dir, step) — the step is load-bearing:
# build_model reads pose_projector--{step}_checkpoint.pt from the ckpt dir.
# These two configs differ in EXACTLY one line: data.cf_window_balance.enabled.
ARMS = [
    ("WITH cf upsampling (boost8)", "#1f77b4",
     CFGS / "2026-07-11_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_cfwinboost8_15k.yaml",
     RUNS / "2026-07-11_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_cfwinboost8_15k_v0/checkpoints/step_15000",
     15000),
    ("NO cf upsampling", "#d62728",
     CFGS / "2026-07-30_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_noups_15k.yaml",
     RUNS / "2026-07-30_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_noups_15k_v0/checkpoints/step_15000",
     15000),
]

# The zone-D exit families in the cfv4 VAL split (see the scan in the 2026-08-22
# work-log): robot leaves aisle C at x~0.7, aisle ahead (F) open in both twins,
# the cross-over shortcut (E) sealed in exactly one.
DEFAULT_FAMILIES = ["warehouse_2x3_1753", "warehouse_2x3_1758"]

# aisle x-extents, north|south sharing a column (mirrors G.ZB)
WEST = (-6.155, -2.295)         # aisle B (north) over aisle E (south)
EAST = (-1.2, 2.66)             # aisle C (north) over aisle F (south)

NCHUNK, DEV_THRESH = 8, 0.75
SWEEP = range(-16, 9)               # the full training window [t*-16, t*+8]
GT_COL = "green"
TWIN_COL = "0.55"
# alpha is a ratio over the twin-GT gap; below this the gap is smaller than the
# models' own error floor (~0.05 m final-wp) and the ratio is pure noise.
ALPHA_MIN_GAP_M = 0.25


def parse_offsets(spec: str) -> list[int]:
    """"-10:2" -> [-10..2]; "-8,-4,0" -> [-8, -4, 0]."""
    if ":" in spec:
        lo, hi = (int(v) for v in spec.split(":"))
        return list(range(lo, hi + 1))
    return [int(v) for v in spec.split(",")]


def classify_family(pos, t_star, suf_a, suf_b):
    """Is this family the 'shortcut toggled' decision class, and which twin is
    which? Returns None when it is not.

    At t* the robot sits at an aisle mouth entering the cross-aisle. The aisle
    straight AHEAD (same column) must be open in both twins; the aisle in the
    OTHER column — the shorter way to the goal — must be sealed in exactly one.
    That twin is the 'resist' branch (must ignore the shortcut and go straight);
    its partner is the 'turn' branch (must cross over).

    e.g. cfB/cfBE at x=0.7: column EAST, ahead=F open in both, cross=E sealed in
         cfBE only -> roles {'B': 'turn', 'BE': 'resist'}
    """
    x, y = float(pos[t_star, 0]), float(pos[t_star, 1])
    if not (1.0 < y < 12.0):
        return None
    if WEST[0] <= x <= WEST[1]:
        col, front, cross, cross_side = "W", "E", "F", "E"
    elif EAST[0] <= x <= EAST[1]:
        col, front, cross, cross_side = "E", "F", "E", "W"
    else:
        return None

    sa = set() if suf_a == "0" else set(suf_a)
    sb = set() if suf_b == "0" else set(suf_b)
    if front in sa or front in sb:
        return None
    if (cross in sa) == (cross in sb):
        return None

    goal_x = float(pos[-1, 0])
    gside = "W" if goal_x <= WEST[1] else ("E" if goal_x >= EAST[0] else None)
    if gside != cross_side:
        return None

    roles = {suf_a: "resist" if cross in sa else "turn",
             suf_b: "resist" if cross in sb else "turn"}
    return {"col": col, "front": front, "cross": cross, "roles": roles}


def final_wp_xy(chunk: np.ndarray) -> tuple[float, float]:
    """Final waypoint of a chunk in metres, robot frame, as (right+, forward+).
    e.g. chunk final row [8.0, -1.5, ...] -> (+0.50 m right, +2.66 m forward)"""
    fwd, left = G.chunk_to_local_m(np.asarray(chunk, dtype=np.float32))
    return float(-left[-1]), float(fwd[-1])


def sweep_family(base, members, dirs, streams, ds, models, collator, pad_id,
                 max_len, name_to_ti, device):
    """Forward BOTH arms on every frame of the decision window, for BOTH twins.

    Returns {suffix: {"t_star": int, "rows": [ {t, off, gt, twin_gt, preds}, ...]}}
    with chunks kept as plain lists so the whole thing is JSON-dumpable."""
    suf = sorted(members)
    a, b = suf[0], suf[1]
    da, db = dirs[members[a]], dirs[members[b]]
    pa, _, _ = streams[da.name]
    pb, _, _ = streams[db.name]
    t_star, _, _ = decision_frame(pa, pb, n_chunk=NCHUNK, dev_thresh_m=DEV_THRESH)
    if t_star is None:
        return None

    cls = classify_family(pa, t_star, a, b)
    out = {}
    for suffix, tdir, twin_suffix, twin_dir in ((a, da, b, db), (b, db, a, da)):
        ti = name_to_ti.get(tdir.name)
        tw_ti = name_to_ti.get(twin_dir.name)
        if ti is None:
            print(f"[skip] {tdir.name} not in dataset")
            continue
        rows = []
        for off in SWEEP:
            t = t_star + off
            if (ti, t) not in ds.samples_index:
                continue
            s = P.sample_pose_final_goal(ds, ti, t)
            batch = T2.pad_and_mask_batch(collator([s]), pad_id, max_len)
            preds = {}
            with torch.inference_mode():
                for name, _col, _cfg, bits in models:
                    _proc, vla, pp, ah, npat, app = bits
                    preds[name] = P.predict_chunk(batch, vla, pp, ah, npat, device,
                                                  abs_pose_projector=app).tolist()
            twin_gt = None
            if tw_ti is not None and (tw_ti, t) in ds.samples_index:
                twin_gt = np.asarray(P.sample_pose_final_goal(ds, tw_ti, t)["actions"],
                                     dtype=np.float32).tolist()
            rows.append({"t": t, "off": off,
                         "gt": np.asarray(s["actions"], dtype=np.float32).tolist(),
                         "twin_gt": twin_gt, "preds": preds})
            print(f"  {tdir.name[:26]:>26} t={t:>3} (t*{off:+d}) done", flush=True)
        out[suffix] = {"t_star": t_star, "twin_suffix": twin_suffix,
                       "traj_dir": str(tdir), "twin_dir": str(twin_dir),
                       "role": (cls["roles"][suffix] if cls else None),
                       "cls": cls, "rows": rows}
    return out


def curves_for(rows, name):
    """(offsets, final-wp error, commitment alpha) for one arm on one twin."""
    offs = np.array([r["off"] for r in rows], dtype=float)
    lat_gt = np.array([final_wp_xy(r["gt"])[0] for r in rows])
    lat_tw = np.array([final_wp_xy(r["twin_gt"])[0] if r["twin_gt"] is not None
                       else np.nan for r in rows])
    lat_p = np.array([final_wp_xy(r["preds"][name])[0] for r in rows])
    err = np.array([float(np.hypot(*(np.subtract(final_wp_xy(r["preds"][name]),
                                     final_wp_xy(r["gt"]))))) for r in rows])
    gap = np.abs(lat_gt - lat_tw)
    alpha = (lat_p - lat_tw) / np.where(gap > 1e-9, lat_gt - lat_tw, np.nan)
    alpha = np.where(gap >= ALPHA_MIN_GAP_M, alpha, np.nan)
    return offs, err, alpha


def aggregate_figure(fams: dict, arm_names, arm_cols, split, out_path):
    """Median +/- IQR across ALL families, split by branch role.

    One column per role ('turn' = must cross to the open shortcut, 'resist' =
    shortcut sealed, must go straight), so the two demands are not averaged
    together — they behave oppositely and the mean of both is meaningless.
    """
    grid = np.arange(-16, 9)
    roles = ["turn", "resist"]
    # {role: {arm: (n_families, len(grid)) array of per-family curves}}
    err_by, alpha_by = {r: {n: [] for n in arm_names} for r in roles}, \
                       {r: {n: [] for n in arm_names} for r in roles}
    counts = {r: 0 for r in roles}
    for base, fam in sorted(fams.items()):
        for suffix, d in sorted(fam.items()):
            role = d.get("role")
            if role not in roles:
                continue
            counts[role] += 1
            for name in arm_names:
                offs, err, alpha = curves_for(d["rows"], name)
                # resample onto the common grid; frames missing at the edges -> nan
                e = np.full(grid.shape, np.nan); al = np.full(grid.shape, np.nan)
                idx = np.searchsorted(grid, offs)
                ok = (idx >= 0) & (idx < len(grid))
                e[idx[ok]] = err[ok]; al[idx[ok]] = alpha[ok]
                err_by[role][name].append(e); alpha_by[role][name].append(al)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, squeeze=False)
    for c, role in enumerate(roles):
        for name, col in zip(arm_names, arm_cols):
            for r, store, lab in ((0, err_by, "final-wp error vs GT (m)"),
                                  (1, alpha_by, "commitment alpha")):
                arr = np.asarray(store[role][name], dtype=float)
                if arr.size == 0:
                    continue
                with np.errstate(all="ignore"):
                    med = np.nanmedian(arr, axis=0)
                    q1 = np.nanpercentile(arr, 25, axis=0)
                    q3 = np.nanpercentile(arr, 75, axis=0)
                ax = axes[r, c]
                ax.fill_between(grid, q1, q3, color=col, alpha=.16, lw=0)
                ax.plot(grid, med, "-o", color=col, ms=3.5, lw=2.2, label=name)
                ax.set_ylabel(lab, fontsize=9)
        axes[0, c].set_title(
            f"{role} branch  (n={counts[role]} twins)" +
            ("  — shortcut OPEN, must cross over" if role == "turn"
             else "  — shortcut SEALED, must go straight"), fontsize=11)
        axes[1, c].axhspan(0.35, 0.65, color="#ffe08a", alpha=.5, zorder=0)
        axes[1, c].axhline(1.0, color=GT_COL, lw=1.2, ls=":")
        axes[1, c].axhline(0.0, color=TWIN_COL, lw=1.2, ls=":")
        axes[1, c].set_ylim(-0.6, 1.6)
        axes[1, c].set_xlabel("frame offset from decision frame t*", fontsize=9)
        for r in (0, 1):
            axes[r, c].axvline(0, color="red", lw=1.0, alpha=.6)
            axes[r, c].grid(alpha=.25)
            axes[r, c].legend(fontsize=8, loc="best")
    for r in (0, 1):
        axes[r, 1].sharey(axes[r, 0])

    fig.suptitle(
        f"CFV4 {split.upper()} — 'shortcut toggled' decision class, "
        f"{len(fams)} families aggregated\n"
        f"{arm_names[0]} (blue) vs {arm_names[1]} (red); median with IQR band. "
        f"alpha blanked where the twin gap < {ALPHA_MIN_GAP_M} m",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def timeline_figure(base: str, fam: dict, arm_names, arm_cols, out_path):
    """Three stacked time-axis panels per twin (twins = columns)."""
    sufs = sorted(fam)
    fig, axes = plt.subplots(3, len(sufs), figsize=(7.0 * len(sufs), 10.5),
                             sharex=True, squeeze=False)
    # rows 1 and 2 share y ACROSS the twin columns so the two branches' error
    # magnitudes can be compared by eye without re-reading the tick labels
    for r in (0, 1):
        for c in range(1, len(sufs)):
            axes[r, c].sharey(axes[r, 0])
    for c, suffix in enumerate(sufs):
        d = fam[suffix]
        rows = d["rows"]
        offs = np.array([r["off"] for r in rows], dtype=float)
        lat_gt = np.array([final_wp_xy(r["gt"])[0] for r in rows])
        lat_tw = np.array([final_wp_xy(r["twin_gt"])[0] if r["twin_gt"] is not None
                           else np.nan for r in rows])

        ax0, ax1, ax2 = axes[0, c], axes[1, c], axes[2, c]

        # row 1 — the two GT branches and where each arm's prediction sits
        ax0.fill_between(offs, lat_gt, lat_tw, color="0.85", zorder=1,
                         label="the decision gap (twin GT spread)")
        ax0.plot(offs, lat_tw, "--", color=TWIN_COL, lw=2.0, zorder=2,
                 label=f"GT of twin cf{d['twin_suffix']} (the WRONG branch here)")
        ax0.plot(offs, lat_gt, "-", color=GT_COL, lw=2.6, zorder=3, label="GT (this twin)")

        gap = np.abs(lat_gt - lat_tw)
        for name, col in zip(arm_names, arm_cols):
            lat_p = np.array([final_wp_xy(r["preds"][name])[0] for r in rows])
            err = np.array([float(np.hypot(*(np.subtract(final_wp_xy(r["preds"][name]),
                                             final_wp_xy(r["gt"]))))) for r in rows])
            # alpha = 0 at the wrong twin's GT, 1 at this twin's GT
            alpha = (lat_p - lat_tw) / np.where(gap > 1e-9, lat_gt - lat_tw, np.nan)
            alpha = np.where(gap >= ALPHA_MIN_GAP_M, alpha, np.nan)
            ax0.plot(offs, lat_p, "-o", color=col, ms=3.5, lw=2.0, zorder=4, label=name)
            ax1.plot(offs, err, "-o", color=col, ms=3.5, lw=2.0, label=name)
            ax2.plot(offs, alpha, "-o", color=col, ms=3.5, lw=2.0, label=name)

        ax2.axhspan(0.35, 0.65, color="#ffe08a", alpha=.5, zorder=0, label="hedging zone")
        ax2.axhline(1.0, color=GT_COL, lw=1.2, ls=":")
        ax2.axhline(0.0, color=TWIN_COL, lw=1.2, ls=":")
        ax2.set_ylim(-0.6, 1.6)

        for ax in (ax0, ax1, ax2):
            ax.axvline(0, color="red", lw=1.0, alpha=.6)
            ax.grid(alpha=.25)
            ax.tick_params(labelsize=8)
        ax0.set_title(f"cf{suffix} — sealed {'none (cf0)' if suffix == '0' else 'aisle ' + '+'.join(suffix)}",
                      fontsize=11)
        ax0.set_ylabel("final-wp lateral (m, + = right)", fontsize=9)
        ax1.set_ylabel("final-wp error vs GT (m)", fontsize=9)
        ax2.set_ylabel("commitment alpha", fontsize=9)
        ax2.set_xlabel("frame offset from decision frame t*", fontsize=9)
        ax0.legend(fontsize=7.5, loc="best")
        ax1.legend(fontsize=7.5, loc="best")
        ax2.legend(fontsize=7.5, loc="best")

    # numbers for the write-up: where each arm's error peaks inside the window
    # and how deep it dips toward the wrong branch
    print(f"\n[summary] {base}")
    print(f"  {'twin':>6} {'arm':>18} {'peak err (m)':>13} {'@off':>5} {'min alpha':>10} {'@off':>5}")
    for suffix in sufs:
        rows = fam[suffix]["rows"]
        offs = np.array([r["off"] for r in rows], dtype=float)
        lat_gt = np.array([final_wp_xy(r["gt"])[0] for r in rows])
        lat_tw = np.array([final_wp_xy(r["twin_gt"])[0] if r["twin_gt"] is not None
                           else np.nan for r in rows])
        gap = np.abs(lat_gt - lat_tw)
        for name in arm_names:
            lat_p = np.array([final_wp_xy(r["preds"][name])[0] for r in rows])
            err = np.array([float(np.hypot(*(np.subtract(final_wp_xy(r["preds"][name]),
                                             final_wp_xy(r["gt"]))))) for r in rows])
            alpha = (lat_p - lat_tw) / np.where(gap > 1e-9, lat_gt - lat_tw, np.nan)
            alpha = np.where(gap >= ALPHA_MIN_GAP_M, alpha, np.nan)
            i = int(np.argmax(err))
            j = int(np.nanargmin(alpha)) if np.any(np.isfinite(alpha)) else i
            amin = f"{alpha[j]:.2f}" if np.any(np.isfinite(alpha)) else "n/a"
            print(f"  {'cf' + suffix:>6} {name:>18} {err[i]:>13.3f} {offs[i]:>+5.0f} "
                  f"{amin:>10} {offs[j]:>+5.0f}")

    t_star = fam[sufs[0]]["t_star"]
    fig.suptitle(
        f"{base} — CFV4 VAL (unseen)   |   decision frame t*={t_star}\n"
        f"{arm_names[0]} (blue) vs {arm_names[1]} (red) — same arch, same inputs; "
        "the FT arm saw 5k extra steps of counterfactual windows only\n"
        "alpha=1 committed to the correct branch, 0 to the other twin's branch, "
        f"~0.5 hedging (blanked where the twin gap < {ALPHA_MIN_GAP_M} m)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def gallery_figure(base, suffix, d, offsets, arm_names, arm_cols, out_path,
                   split_label="VAL (unseen)"):
    """The classic 4-column per-frame gallery, rows restricted to `offsets`."""
    rows = [r for r in d["rows"] if r["off"] in offsets]
    if not rows:
        print(f"[skip] no frames in {offsets} for cf{suffix}")
        return
    tdir = pathlib.Path(d["traj_dir"])
    pos, yaws, _ = load_pose_stream(tdir)
    t_star = d["t_star"]
    role = d.get("role")
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(17, 3.3 * n),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1, 1]},
                             squeeze=False)
    for r, row in enumerate(rows):
        t, off = row["t"], row["off"]
        gt = np.asarray(row["gt"], dtype=np.float32)
        twin_gt = (np.asarray(row["twin_gt"], dtype=np.float32)
                   if row["twin_gt"] is not None else None)

        p = tdir / "frames" / f"{t:03d}.png"
        if p.exists():
            axes[r, 0].imshow(np.asarray(Image.open(p)))
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_title(f"ego @ frame {t}", fontsize=9)
        axes[r, 1].imshow(G.cctv_tile(tdir, t))
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        axes[r, 1].set_title("CCTV NW|NE / SW|SE  (the disambiguating input)", fontsize=9)

        ax = axes[r, 2]
        gf, gl = G.chunk_to_local_m(gt)
        xs, ys = [-gl], [gf]
        if twin_gt is not None:
            tf, tl = G.chunk_to_local_m(twin_gt)
            ax.plot(-tl, tf, "--", color=TWIN_COL, lw=2.0, zorder=2,
                    label=f"GT of twin cf{d['twin_suffix']}")
            xs.append(-tl); ys.append(tf)
        ax.plot(-gl, gf, "-o", color=GT_COL, ms=4, lw=2.6, zorder=3, label="GT")
        errs = []
        for name, col in zip(arm_names, arm_cols):
            pf, pl = G.chunk_to_local_m(np.asarray(row["preds"][name], dtype=np.float32))
            ax.plot(-pl, pf, "-o", color=col, ms=3.5, lw=2.0, alpha=.9, label=name)
            errs.append(float(np.hypot(pf[-1] - gf[-1], pl[-1] - gl[-1])))
            xs.append(-pl); ys.append(pf)
        ax.axhline(0, color="0.9", lw=.8); ax.axvline(0, color="0.9", lw=.8)
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        mx = max(0.35, 0.15 * (xs.max() - xs.min() + ys.max() - ys.min()))
        ax.set_aspect("equal")
        ax.set_xlim(xs.min() - mx, xs.max() + mx)
        ax.set_ylim(ys.min() - mx, ys.max() + mx)
        ax.legend(fontsize=7, loc="lower left"); ax.tick_params(labelsize=7)
        ax.set_xlabel("left <- m -> right", fontsize=7)
        ax.set_ylabel("forward (m)", fontsize=7)
        twin_sep = float("nan")
        if twin_gt is not None:
            tf2, tl2 = G.chunk_to_local_m(twin_gt)
            twin_sep = float(np.hypot(tf2[-1] - gf[-1], tl2[-1] - gl[-1]))
        ax.set_title(f"@ {t} (t*{off:+d}) | TWIN gap {twin_sep:.2f} m | err "
                     + " / ".join(f"{e:.2f}" for e in errs),
                     fontsize=8.5, color=("red" if off == 0 else "0.3"))

        axm = axes[r, 3]
        for z, ((x0, x1), (y0, y1)) in G.ZB.items():
            axm.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="0.8"))
            axm.text((x0 + x1) / 2, (y0 + y1) / 2, z, color="0.8", ha="center", fontsize=7)
        axm.plot(pos[:, 0], pos[:, 1], "-", color="0.6", lw=1.1, alpha=.8)
        gx, gy = G.chunk_to_world(gt, pos[t, 0], pos[t, 1], yaws[t])
        axm.plot(gx, gy, "-", color=GT_COL, lw=2.4, zorder=3)
        for name, col in zip(arm_names, arm_cols):
            wx, wy = G.chunk_to_world(np.asarray(row["preds"][name], dtype=np.float32),
                                      pos[t, 0], pos[t, 1], yaws[t])
            axm.plot(wx, wy, "-", color=col, lw=1.9, alpha=.9)
        axm.plot(pos[t, 0], pos[t, 1], "o", color="black", ms=6)
        axm.set_xlim(-9, 5); axm.set_ylim(-21, 28); axm.set_aspect("equal")
        axm.tick_params(labelsize=6); axm.set_title(f"world @ {t}", fontsize=9)

    seal = "none (cf0)" if suffix == "0" else f"aisle {'+'.join(suffix)}"
    fig.suptitle(
        f"{base} cf{suffix} — sealed {seal}   |   CFV4 {split_label}   |   t*={t_star}"
        + (f"   |   {role} branch" if role else "") + "\n"
        f"GT (green) vs {arm_names[0]} (blue) vs {arm_names[1]} (red); "
        "dashed grey = the TWIN's GT — the gap between green and dashed is what "
        "the CCTV must resolve\n"
        "identical inputs to both arms; per-panel 'err' = final-waypoint error in m, "
        f"{arm_names[0]} / {arm_names[1]}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.5 / n])
    fig.savefig(out_path, dpi=95)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                    help="comma-separated family bases, or 'all'")
    ap.add_argument("--offsets", default="-10:2",
                    help="gallery rows: 'lo:hi' range or comma list of t* offsets")
    ap.add_argument("--class-only", action="store_true",
                    help="auto-select every family in the 'shortcut toggled' class")
    ap.add_argument("--figures", default="timeline,gallery,aggregate",
                    help="comma list of: timeline, gallery, aggregate")
    ap.add_argument("--from-cache", action="store_true",
                    help="re-render figures from preds_*.json, no GPU / no model load")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    offsets = set(parse_offsets(args.offsets))
    want_fig = {f.strip() for f in args.figures.split(",") if f.strip()}
    arm_names = [a[0] for a in ARMS]
    arm_cols = [a[1] for a in ARMS]
    out_dir = OUT / PAIR_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[arms] {' vs '.join(arm_names)}  ->  {out_dir}")

    root = SPLIT_ROOT / args.split
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    fam_index, _ = group_cf_families([d.name for d in dirs])
    if args.class_only:
        # keep only families with the 'shortcut toggled' geometry (needs poses,
        # so this runs before the model load and costs nothing)
        wanted = []
        for base, members in sorted(fam_index.items()):
            if len(members) < 2:
                continue
            sa, sb = sorted(members)[:2]
            pa, _, _ = load_pose_stream(dirs[members[sa]])
            pb, _, _ = load_pose_stream(dirs[members[sb]])
            ts, _, _ = decision_frame(pa, pb, n_chunk=NCHUNK, dev_thresh_m=DEV_THRESH)
            if ts is None or ts < 16:
                continue
            if classify_family(pa, ts, sa, sb) is not None:
                wanted.append(base)
        print(f"[class] {len(wanted)} families in the shortcut-toggled class "
              f"({args.split}): {', '.join(wanted)}")
    else:
        wanted = (sorted(fam_index) if args.families == "all"
                  else args.families.split(","))

    if not args.from_cache:
        cfgs = [yaml.safe_load(c.read_text()) for _, _, c, _, _ in ARMS]
        models = []
        for (name, col, _cfgp, ckpt, step), cfg in zip(ARMS, cfgs):
            print(f"[load] {name} <- {ckpt} (step {step})", flush=True)
            models.append((name, col, cfg, P.build_model(cfg, ckpt, step, args.device)))
        proc0 = models[0][3][0]
        # ONE dataset instance (from the parent's data cfg) so both arms see
        # bit-identical inputs; sample_pose_final_goal pins fg=1.0 + modality 4.
        ds = P.build_dataset(cfgs[0], proc0, root, int(cfgs[0]["train"].get("seed", 42)))
        collator = PaddedCollatorForActionPrediction_Nav_CCTV(
            proc0.tokenizer.model_max_length, proc0.tokenizer.pad_token_id,
            padding_side="right", num_img=cfgs[0]["model"]["num_images_in_input"])
        pad_id, max_len = proc0.tokenizer.pad_token_id, proc0.tokenizer.model_max_length
        name_to_ti = P._name_to_traj_index(ds)
        streams = {d.name: load_pose_stream(d) for d in dirs}

    all_fams = {}
    for base in wanted:
        members = fam_index.get(base)
        if not members or len(members) < 2:
            print(f"[skip] {base}: no twin pair in {args.split}")
            continue
        cache = out_dir / f"preds_{base}_{args.split}.json"
        if args.from_cache:
            if not cache.exists():
                print(f"[skip] {base}: no cache at {cache}")
                continue
            fam = json.loads(cache.read_text())
        else:
            print(f"\n=== {base} ===", flush=True)
            fam = sweep_family(base, members, dirs, streams, ds, models, collator,
                               pad_id, max_len, name_to_ti, args.device)
            if fam is None:
                print(f"[skip] {base}: no decision frame")
                continue
            cache.write_text(json.dumps(fam))
            print(f"cached {cache}", flush=True)

        all_fams[base] = fam
        if "timeline" in want_fig:
            timeline_figure(base, fam, arm_names, arm_cols,
                            out_dir / f"timeline_{base}_{args.split}.png")
        if "gallery" in want_fig:
            for suffix, d in sorted(fam.items()):
                gallery_figure(base, suffix, d, offsets, arm_names, arm_cols,
                               out_dir / f"gallery_{base}_cf{suffix}_{args.split}.png",
                               split_label=("VAL (unseen)" if args.split == "val"
                                            else "TRAIN (seen by both arms)"))

    if "aggregate" in want_fig and all_fams:
        aggregate_figure(all_fams, arm_names, arm_cols, args.split,
                         out_dir / f"aggregate_{args.split}.png")


if __name__ == "__main__":
    main()
