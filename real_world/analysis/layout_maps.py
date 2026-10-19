
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import Patch, Wedge           # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from session_common import (COL, MRK, NICE, OBJS, REPO, STOP_OFFSET_M,  # noqa: E402
                            DEFAULT_TRAJ, load)

XL, YL = (-4.9, 4.0), (-1.2, 3.0)


def robot(ax, s=1.0):
    ax.plot(0, 0, marker="D", ms=6 * s, color="#1a1a1a", zorder=6)
    ax.annotate("", xy=(0, .55 * s), xytext=(0, 0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", lw=1.4 * s))


def rings(ax, rs=(1, 2, 3, 4), label=True):
    for r in rs:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="#bbb", lw=.6, ls=(0, (3, 3)), zorder=0))
        if label:
            ax.text(r * np.cos(np.radians(35)), r * np.sin(np.radians(35)), f"{r} m",
                    color="#999", fontsize=6.5, ha="left", va="bottom", zorder=1)


def frame(ax, xl=XL, yl=YL):
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")


def legend(target, **kw):
    h = [Line2D([], [], color=COL[o], marker=MRK[o], ls="-", ms=7, label=NICE[o]) for o in OBJS]
    h.append(Line2D([], [], color="#1a1a1a", marker="D", ls="", ms=6, label="robot start (fixed)"))
    target.legend(handles=h, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", type=pathlib.Path, default=DEFAULT_TRAJ)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "real_world/outputs/layout_2026_08_21")
    ap.add_argument("--stop-offset-m", type=float, default=STOP_OFFSET_M,
                    help="how far short of the object each take stops (default %(default)s)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs, _ = load(args.traj, args.stop_offset_m)
    fams = sorted({r["fam"] for r in runs})
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                         "grid.linewidth": .5, "axes.edgecolor": "#999"})

    # ---------------------------------------------------------- 01 overview
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=.18)
    ax = fig.add_subplot(gs[0]); rings(ax); robot(ax)
    for o in OBJS:
        P = np.array([r["obj_xy"] for r in runs if r["obj"] == o])
        ax.scatter(P[:, 0], P[:, 1], c=COL[o], marker=MRK[o], s=52, ec="white", lw=.8, zorder=5)
    for r in runs:
        ax.annotate(str(r["fam"]), r["obj_xy"], fontsize=5.5, color="#555",
                    xytext=(0, 7), textcoords="offset points", ha="center", zorder=4)
    frame(ax); legend(ax, loc="lower left", fontsize=8, framealpha=.95)
    ax.set_title(f"Object placements, all {len(fams)} scenes  (robot start-anchored frame)",
                 fontsize=10.5, loc="left")
    ax.set_xlabel("left  <-   metres   ->  right"); ax.set_ylabel("metres ahead")

    axp = fig.add_subplot(gs[1], projection="polar")
    axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
    axp.set_thetamin(-100); axp.set_thetamax(100)
    A = np.array([r["obj_xy"] for r in runs])
    rng, brg = np.hypot(A[:, 0], A[:, 1]), np.degrees(np.arctan2(A[:, 0], A[:, 1]))
    for o in OBJS:
        P = np.array([r["obj_xy"] for r in runs if r["obj"] == o])
        axp.scatter(np.arctan2(P[:, 0], P[:, 1]), np.hypot(P[:, 0], P[:, 1]),
                    c=COL[o], marker=MRK[o], s=44, ec="white", lw=.7, zorder=5)
    axp.set_rmax(5); axp.set_rticks([1, 2, 3, 4, 5]); axp.set_rlabel_position(103)
    axp.set_title(f"same, as range x bearing from the start pose\n"
                  f"(covered: {rng.min():.1f}-{rng.max():.1f} m, "
                  f"{brg.min():+.0f}..{brg.max():+.0f} deg)", fontsize=9.5)
    fig.suptitle(f"Spot session layout - {len(fams)} scenes, {len(runs)} takes",
                 fontsize=12, x=.09, ha="left", y=.98)
    fig.savefig(args.out / "01_layout_overview.png", dpi=170, bbox_inches="tight",
                facecolor="white"); plt.close(fig)

    # ---------------------------------------------------------- 02 scene grid
    ncol = 5; nrow = int(np.ceil(len(fams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 1.52 * nrow), sharex=True, sharey=True)
    for ax, f in zip(axes.ravel(), fams):
        rings(ax, (2, 4), label=False); robot(ax, .8)
        for r in [x for x in runs if x["fam"] == f]:
            ax.plot(r["X"], r["Y"], color=COL[r["obj"]], lw=1.5, alpha=.85, zorder=3)
            ax.scatter(*r["obj_xy"], c=COL[r["obj"]], marker=MRK[r["obj"]], s=60,
                       ec="white", lw=.9, zorder=5)
        frame(ax); ax.set_title(f"scene {f}", fontsize=9, pad=3); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    fig.subplots_adjust(hspace=.28, wspace=.06)
    legend(fig, loc="lower center", ncol=4, fontsize=9, framealpha=.95, bbox_to_anchor=(.5, -.02))
    fig.suptitle("Per-scene layout - trajectory per object, marker at the inferred object "
                 f"position (stop + {args.stop_offset_m:.2f} m along final heading)",
                 fontsize=12, y=1.0)
    fig.savefig(args.out / "02_scenes_grid.png", dpi=140, bbox_inches="tight",
                facecolor="white"); plt.close(fig)

    # ---------------------------------------------------------- 03 merged all
    fig, ax = plt.subplots(figsize=(11, 6.2)); rings(ax); robot(ax)
    for r in runs:
        ax.plot(r["X"], r["Y"], color=COL[r["obj"]], lw=1.1, alpha=.45, zorder=2)
    for o in OBJS:
        P = np.array([r["obj_xy"] for r in runs if r["obj"] == o])
        ax.scatter(P[:, 0], P[:, 1], c=COL[o], marker=MRK[o], s=62, ec="white", lw=.9, zorder=5)
    frame(ax); legend(ax, loc="lower left", fontsize=9, framealpha=.95)
    ax.set_title(f"All {len(runs)} takes merged - {len(fams)} scenes x {len(OBJS)} objects",
                 fontsize=12, loc="left")
    ax.set_xlabel("left  <-   metres   ->  right"); ax.set_ylabel("metres ahead")
    fig.savefig(args.out / "03_merged_all.png", dpi=170, bbox_inches="tight",
                facecolor="white"); plt.close(fig)

    # ---------------------------------------------------------- 04 per object
    for i, o in enumerate(OBJS):
        sel = [r for r in runs if r["obj"] == o]
        fig, ax = plt.subplots(figsize=(11, 6.2)); rings(ax); robot(ax)
        for r in sel:
            ax.plot(r["X"], r["Y"], color=COL[o], lw=1.3, alpha=.5, zorder=2)
            ax.scatter(*r["obj_xy"], c=COL[o], marker=MRK[o], s=70, ec="white", lw=.9, zorder=5)
            ax.annotate(str(r["fam"]), r["obj_xy"], fontsize=6.5, color="#444",
                        xytext=(0, 8), textcoords="offset points", ha="center", zorder=6)
        frame(ax)
        ax.set_title(f"{NICE[o]} - {len(sel)} placements, one per scene",
                     fontsize=12, loc="left", color=COL[o])
        ax.set_xlabel("left  <-   metres   ->  right"); ax.set_ylabel("metres ahead")
        fig.savefig(args.out / f"04{'abc'[i]}_merged_{o.replace(' ', '_')}.png",
                    dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # ---------------------------------------------------------- 05 coverage
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=.2)
    ax = fig.add_subplot(gs[0])
    s = np.sort(brg); gaps = np.diff(s); j = int(gaps.argmax())
    b0, b1, r_lo, r_hi = s[j], s[j + 1], rng.min(), rng.max()

    def wedge(a0, a1, q0, q1, **kw):
        return Wedge((0, 0), q1, 90 - a1, 90 - a0, width=q1 - q0, **kw)

    ax.add_patch(wedge(b0, b1, 0, 5.0, fc="#1565c0", alpha=.13, ec="#1565c0", lw=1, ls="--", zorder=1))
    ax.add_patch(wedge(-100, 100, 0, r_lo, fc="#6a1b9a", alpha=.13, ec="#6a1b9a", lw=1, ls="--", zorder=1))
    ax.add_patch(wedge(-100, 100, r_hi, 5.6, fc="#00838f", alpha=.13, ec="#00838f", lw=1, ls="--", zorder=1))
    rings(ax, (1, 2, 3, 4, 5), label=False)
    for o in OBJS:
        P = np.array([r["obj_xy"] for r in runs if r["obj"] == o])
        ax.scatter(P[:, 0], P[:, 1], c=COL[o], marker=MRK[o], s=48, ec="white", lw=.8, zorder=5)
    robot(ax)
    ax.text(0, 3.4, f"NEVER PLACED\nAHEAD\n({b1 - b0:.0f} deg wedge)", color="#1565c0",
            ha="center", va="center", fontsize=9, fontweight="bold", zorder=7)
    ax.text(0, r_lo * .75, f"never < {r_lo:.1f} m", color="#6a1b9a", ha="center",
            fontsize=8, fontweight="bold", zorder=7)
    ax.text(3.2, r_hi * .9, f"never > {r_hi:.1f} m", color="#00838f", ha="center",
            fontsize=8, fontweight="bold", zorder=7)
    ax.set_xlim(-5, 5); ax.set_ylim(-1.6, 5.2); ax.set_aspect("equal")
    ax.set_xlabel("left  <-   metres   ->  right"); ax.set_ylabel("metres ahead")
    ax.set_title("Where objects were never placed - the OOD budget", fontsize=11, loc="left")
    h = [Line2D([], [], color=COL[o], marker=MRK[o], ls="", ms=7, label=NICE[o]) for o in OBJS]
    h.append(Patch(fc="#1565c0", alpha=.2, ec="#1565c0", ls="--", label="uncovered region"))
    ax.legend(handles=h, loc="lower left", fontsize=8, framealpha=.95)

    axp = fig.add_subplot(gs[1], projection="polar")
    axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
    axp.set_thetamin(-110); axp.set_thetamax(110)
    rb, bb = np.arange(0, 5.51, .5), np.arange(-110, 111, 20)
    H, _, _ = np.histogram2d(rng, brg, bins=[rb, bb])
    cm = matplotlib.colormaps["YlGnBu"].copy(); cm.set_bad("white")
    m = axp.pcolormesh(np.radians(bb), rb, np.ma.masked_equal(H, 0), cmap=cm,
                       shading="flat", vmin=0)
    axp.set_rmax(5.5); axp.set_rticks([1, 2, 3, 4, 5]); axp.set_rlabel_position(115)
    axp.set_title("placement density\n(0.5 m x 20 deg cells; white = never)", fontsize=9.5)
    fig.colorbar(m, ax=axp, pad=.11, shrink=.75, label="placements")
    fig.suptitle("Coverage of the session, and the holes an OOD set can target",
                 fontsize=12, x=.09, ha="left", y=1.0)
    fig.savefig(args.out / "05_coverage_and_gaps.png", dpi=170, bbox_inches="tight",
                facecolor="white"); plt.close(fig)

    # ---------------------------------------------------------- table
    with open(args.out / "object_positions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "object", "bag", "stop_x_m", "stop_y_m", "obj_x_m", "obj_y_m",
                    "range_m", "bearing_deg", "path_len_m"])
        for r in runs:
            ox, oy = r["obj_xy"]
            w.writerow([r["fam"], NICE[r["obj"]], r["name"], f"{r['stop_xy'][0]:.3f}",
                        f"{r['stop_xy'][1]:.3f}", f"{ox:.3f}", f"{oy:.3f}",
                        f"{np.hypot(ox, oy):.3f}", f"{np.degrees(np.arctan2(ox, oy)):.1f}",
                        f"{r['path_m']:.2f}"])

    print(f"{len(runs)} takes, {len(fams)} scenes -> {args.out}")
    print(f"  range {rng.min():.2f}-{rng.max():.2f} m, bearing {brg.min():+.1f}..{brg.max():+.1f} deg")
    print(f"  largest empty bearing wedge: {b1 - b0:.1f} deg ({b0:+.1f} .. {b1:+.1f})")
    for f in sorted(args.out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
