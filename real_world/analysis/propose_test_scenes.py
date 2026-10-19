
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from session_common import COL, MRK, NICE, OBJS, REPO  # noqa: E402

CSV_OBJS = [NICE[o] for o in OBJS]


def load_training(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    sc = collections.defaultdict(dict)
    for r in rows:
        sc[int(r["scene"])][r["object"]] = (float(r["obj_x_m"]), float(r["obj_y_m"]))
    scenes = sorted(sc)
    missing = [s for s in scenes if any(o not in sc[s] for o in CSV_OBJS)]
    if missing:
        raise SystemExit(f"training scenes missing an object: {missing}")
    return np.array([[sc[s][o] for o in CSV_OBJS] for s in scenes]), scenes


def scene_distance(S, others):
    """THE metric, used for both novelty and selection.

        d(S, T) = mean_o || S_o - T_o ||_2

    Mean per-object Euclidean distance in metres. Objects are labelled, so plant
    compares to plant and no assignment step is needed. S is (3,2), others is
    (m,3,2), returns (m,).

    Deliberately the only scene metric in this file: novelty and the
    farthest-point selection both call it, so the paper has one definition to
    state rather than two that merely correlate.
    """
    return np.linalg.norm(others - S[None], axis=2).mean(axis=1)


def novelty_of(S, train):
    """Distance to the NEAREST training scene (not the mean over all of them)."""
    return float(scene_distance(S, train).min())


def build_pool(a, rng):
    """Feasible single-object slots, split into inside-FOV and outside-FOV."""
    n = a.pool
    x = rng.uniform(a.x_lo, a.x_hi, n)
    y = rng.uniform(a.y_lo, a.y_hi, n)
    r = np.hypot(x, y)
    b = np.degrees(np.arctan2(x, y))
    keep = (r >= a.r_min) & (r <= a.r_max) & (np.abs(b) <= a.b_max)
    x, y, b = x[keep], y[keep], b[keep]
    inside = np.abs(b) <= a.th_in
    outside = np.abs(b) >= a.th_out
    return np.c_[x, y][inside], np.c_[x, y][outside]


def sample_scene(k_vis, pin, pout, rng, sep_min, tries=400):
    """A scene with exactly k_vis objects inside the FOV, or None."""
    for _ in range(tries):
        want = rng.permutation([True] * k_vis + [False] * (3 - k_vis))
        pts = np.array([pin[rng.integers(len(pin))] if w else pout[rng.integers(len(pout))]
                        for w in want])
        d = [np.linalg.norm(pts[i] - pts[j]) for i in range(3) for j in range(i + 1, 3)]
        if min(d) >= sep_min:
            return pts, want
    return None, None


def farthest_point(cands, k, rng):
    """Greedy farthest-first traversal (Gonzalez) under scene_distance.

    d[i] always holds candidate i's distance to its nearest already-picked scene;
    each step takes the argmax and updates d by elementwise minimum. Greedily
    maximises the minimum separation among the picks, so a stratum is not filled
    with near-duplicates. Only the seed point is random.
    """
    if len(cands) <= k:
        return list(range(len(cands)))
    V = np.stack([c["S"] for c in cands])            # (n, 3, 2), not flattened
    picked = [int(rng.integers(len(V)))]
    d = scene_distance(V[picked[0]], V)
    while len(picked) < k:
        j = int(np.argmax(d))
        picked.append(j)
        d = np.minimum(d, scene_distance(V[j], V))
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = REPO / "real_world/outputs/layout_2026_08_21"
    p.add_argument("--train-csv", type=pathlib.Path, default=d / "object_positions.csv")
    p.add_argument("--out", type=pathlib.Path, default=REPO / "real_world/outputs/test_scenes")
    p.add_argument("--per-class", type=int, nargs=4, default=[5, 5, 5, 5],
                   metavar=("N0", "N1", "N2", "N3"),
                   help="scenes with 0/1/2/3 objects visible (default 5 5 5 5 = 20 scenes). "
                        "Balanced on purpose: equal n per class keeps the visibility factor "
                        "balanced for analysis, and avoids under-sampling the 3-visible class, "
                        "which is the one with zero training support and therefore the most "
                        "informative. The 3-visible class is geometrically tighter (~3.6k "
                        "feasible candidates vs 4k) but that is far more than enough to draw "
                        "5 scenes from.")
    p.add_argument("--strata", type=int, default=3, help="novelty strata per class")
    p.add_argument("--r-min", type=float, default=1.5, help="tube radius around the robot")
    p.add_argument("--r-max", type=float, default=4.6)
    p.add_argument("--b-max", type=float, default=100.0, help="max |bearing| (deg)")
    p.add_argument("--x-lo", type=float, default=-4.4); p.add_argument("--x-hi", type=float, default=3.5)
    p.add_argument("--y-lo", type=float, default=-0.4); p.add_argument("--y-hi", type=float, default=2.4)
    p.add_argument("--sep-min", type=float, default=1.5)
    # FOV is derived, not typed. FrontStitcher remaps onto a 110 deg canvas
    # (width 768 -> f = 268.9 px) and crops to the valid 599 px, so the real
    # horizontal field is 2*atan(599/2/268.9) = 96.2 deg (the 110 in
    # operator_loop's hfov_deg is the canvas). Half-FOV 48.1 deg with a +-15%
    # dead-band = [40.9, 55.3], rounded to [40, 55]: an object centred on the
    # FOV edge is half out of frame and sits where the projection is worst.
    p.add_argument("--th-in", type=float, default=40.0,
                   help="visible if |bearing| <= this (deg)")
    p.add_argument("--th-out", type=float, default=55.0,
                   help="not visible if |bearing| >= this (deg)")
    p.add_argument("--pool", type=int, default=400000)
    p.add_argument("--cands-per-class", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260825)
    a = p.parse_args()

    print(f"FOV thresholds: visible <= {a.th_in:.0f} deg, not visible >= {a.th_out:.0f} deg "
          f"(derived from 96.2 deg HFOV, half 48.1, +-15% band rounded)")

    rng = np.random.default_rng(a.seed)
    train, train_ids = load_training(a.train_csv)
    a.out.mkdir(parents=True, exist_ok=True)

    nn = []
    for i in range(len(train)):
        dd = np.linalg.norm(train - train[i][None], axis=2).mean(axis=1)
        dd[i] = np.inf
        nn.append(dd.min())
    nn = np.array(nn)
    print(f"{len(train)} training scenes | their own NN spacing: "
          f"min {nn.min():.2f}  med {np.median(nn):.2f}  max {nn.max():.2f} m")

    pin, pout = build_pool(a, rng)
    print(f"feasible slots: {len(pin)} inside FOV (|b|<={a.th_in}), {len(pout)} outside "
          f"(|b|>={a.th_out}), tube {a.r_min}-{a.r_max} m\n")

    chosen, report = [], []
    for k_vis, n_want in enumerate(a.per_class):
        if n_want == 0:
            continue
        cands = []
        for _ in range(a.cands_per_class):
            S, _ = sample_scene(k_vis, pin, pout, rng, a.sep_min)
            if S is None:
                continue
            cands.append({"S": S, "nov": novelty_of(S, train)})
        if not cands:
            print(f"  {k_vis} visible: NO FEASIBLE SCENE under these constraints - skipped")
            report.append((k_vis, 0, None, None))
            continue
        novs = np.array([c["nov"] for c in cands])
        edges = np.quantile(novs, np.linspace(0, 1, a.strata + 1))
        per = [n_want // a.strata] * a.strata
        for i in range(n_want - sum(per)):
            per[-(i + 1)] += 1
        got = 0
        for si in range(a.strata):
            lo, hi = edges[si], edges[si + 1] + (1e-9 if si == a.strata - 1 else 0)
            sub = [c for c in cands if lo <= c["nov"] <= hi]
            if not sub:
                continue
            for j in farthest_point(sub, per[si], rng):
                c = sub[j]
                chosen.append({**c, "k_vis": k_vis, "stratum": si,
                               "pct": float((nn < c["nov"]).mean() * 100)})
                got += 1
        report.append((k_vis, got, novs.min(), novs.max()))
        print(f"  {k_vis} visible: {len(cands)} candidates, novelty "
              f"{novs.min():.2f}-{novs.max():.2f} m -> {got} scenes across {a.strata} strata")

    chosen.sort(key=lambda c: (c["k_vis"], c["stratum"], -c["nov"]))
    for i, c in enumerate(chosen, 1):
        c["id"] = f"T{i:02d}"

    csv_path = a.out / "proposed_scenes.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene_id", "n_visible", "novelty_stratum", "novelty_m",
                    "novelty_pct_vs_train_nn", "object", "target_visible",
                    "x_m", "y_m", "range_m", "bearing_deg"])
        for c in chosen:
            for oi, o in enumerate(CSV_OBJS):
                x, y = c["S"][oi]
                b = np.degrees(np.arctan2(x, y))
                w.writerow([c["id"], c["k_vis"], c["stratum"], f'{c["nov"]:.3f}',
                            f'{c["pct"]:.0f}', o, int(abs(b) <= a.th_in),
                            f"{x:.3f}", f"{y:.3f}", f"{np.hypot(x, y):.3f}", f"{b:.1f}"])

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Wedge

    ncol = 5
    nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 1.55 * nrow), sharex=True, sharey=True)
    plt.rcParams.update({"axes.grid": True})
    for ax, c in zip(np.atleast_1d(axes).ravel(), chosen):
        ax.add_patch(Wedge((0, 0), a.r_max, 90 - a.th_in, 90 + a.th_in,
                           fc="#1565c0", alpha=.07, zorder=0))
        ax.add_patch(plt.Circle((0, 0), a.r_min, fc="#6a1b9a", alpha=.10, zorder=0))
        for r in (2, 4):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="#ccc", lw=.5,
                                    ls=(0, (3, 3)), zorder=0))
        ax.plot(0, 0, marker="D", ms=5, color="#1a1a1a", zorder=6)
        ax.annotate("", xy=(0, .5), xytext=(0, 0), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", lw=1.2))
        for oi, o in enumerate(OBJS):
            x, y = c["S"][oi]
            ax.scatter(x, y, c=COL[o], marker=MRK[o], s=55, ec="white", lw=.8, zorder=5)
        ax.set_xlim(-4.9, 4.0); ax.set_ylim(-1.2, 3.0); ax.set_aspect("equal")
        ax.set_title(f'{c["id"]}  {c["k_vis"]}vis  nov {c["nov"]:.2f}', fontsize=8, pad=3)
        ax.tick_params(labelsize=6)
    for ax in np.atleast_1d(axes).ravel()[len(chosen):]:
        ax.axis("off")
    h = [Line2D([], [], color=COL[o], marker=MRK[o], ls="", ms=7, label=NICE[o]) for o in OBJS]
    h += [Line2D([], [], color="#1a1a1a", marker="D", ls="", ms=6, label="robot start"),
          Line2D([], [], color="#1565c0", lw=6, alpha=.3, label=f"FOV (|b|<={a.th_in:.0f} deg)"),
          Line2D([], [], color="#6a1b9a", lw=6, alpha=.3, label=f"excluded tube (r<{a.r_min} m)")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(.5, -.02))
    fig.subplots_adjust(hspace=.3, wspace=.06)
    fig.suptitle(f"Proposed test scenes - {len(chosen)} scenes x 3 targets = {3*len(chosen)} takes"
                 f"  (seed {a.seed})", fontsize=12, y=1.0)
    fig.savefig(a.out / "proposed_scenes.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n{len(chosen)} scenes -> {3*len(chosen)} takes")
    print(f"  {csv_path}")
    print(f"  {a.out / 'proposed_scenes.png'}")
    nv = np.array([c["nov"] for c in chosen])
    print(f"  novelty {nv.min():.2f}-{nv.max():.2f} m "
          f"(training NN spacing {nn.min():.2f}-{nn.max():.2f})")


if __name__ == "__main__":
    main()
