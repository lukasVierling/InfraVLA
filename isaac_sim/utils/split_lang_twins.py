"""Family-grouped train/val split for the language-twin datasets (langv2).

Unlike split_dataset_by_seed.py (seed-range, single source dir, one traj per
seed), the language task has N twins per family that share a scene byte-for-byte
and MUST stay on the same side of the split (else a val twin leaks its family's
pixels into train). Val is drawn from COMPLETE families only (all 4 twins
present) so the instruction-swap / counterfactual val metric has both twins to
compare. Trajectories may live across several source dirs (main + top-up); all
are unioned. Output = symlinks (named by traj-dir basename) under
runs/data/splits/<out>/{train,val,test}, matching what the dataloader globs.

Deterministic: complete families are shuffled with a FIXED seed, so re-running
reproduces the identical split.

Run:
  cd <repo root>
  PYTHONPATH=. python isaac_sim/utils/split_lang_twins.py \
      --datasets runs/data_gen/warehouse_2x3_langv2_2026-07-22 \
                 runs/data_gen/warehouse_2x3_langv2b_2026-07-22 \
      --prefix warehouse_2x3_langv2 --val-families 12 --seed 42 \
      --out 2026-07-22_warehouse_2x3_langv2
"""
from __future__ import annotations
import argparse, json, pathlib, random, re
from collections import defaultdict

_REPO = pathlib.Path(__file__).resolve().parents[2]
SPLITS_ROOT = _REPO / "runs" / "data" / "splits"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="source data-gen dirs (recorded trajectory roots)")
    ap.add_argument("--prefix", required=True, help="e.g. warehouse_2x3_langv2")
    ap.add_argument("--val-families", type=int, required=True,
                    help="how many COMPLETE families go to val (x4 twins = val traj)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="subdir under runs/data/splits/")
    args = ap.parse_args()

    # dir name: <prefix>_NNNN_t<A-D>__pose__<ts>. Family = the _NNNN base.
    rex = re.compile(rf"({re.escape(args.prefix)}_\d{{4}})_t([A-D])__pose__")
    # family -> {twin: latest reached traj dir}
    fam: dict[str, dict[str, pathlib.Path]] = defaultdict(dict)
    for ds in args.datasets:
        for d in sorted(pathlib.Path(ds).glob("*/*__pose__*")):
            m = rex.match(d.name)
            if not m or not (d / "result.json").exists():
                continue
            base, twin = m.group(1), m.group(2)
            # keep the LATEST dir if a scene was (re)recorded more than once
            prev = fam[base].get(twin)
            if prev is None or d.name > prev.name:
                fam[base][twin] = d

    complete = sorted(f for f, t in fam.items() if len(t) == 4)
    incomplete = sorted(f for f, t in fam.items() if len(t) < 4)
    if args.val_families > len(complete):
        raise SystemExit(f"need {args.val_families} complete families for val, "
                         f"only {len(complete)} exist")

    rng = random.Random(args.seed)
    shuffled = complete[:]
    rng.shuffle(shuffled)
    val_fams = set(shuffled[:args.val_families])
    # train = every family NOT in val (complete + incomplete both allowed in train)
    train_fams = [f for f in sorted(fam) if f not in val_fams]

    out_dir = SPLITS_ROOT / args.out
    for sub in ("train", "val", "test"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    def link(fams, sub):
        n = 0
        for f in fams:
            for twin, d in fam[f].items():
                dst = out_dir / sub / d.name
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(d.resolve())
                n += 1
        return n

    n_train = link(train_fams, "train")
    n_val = link(sorted(val_fams), "val")
    n_test = link(sorted(val_fams), "test")   # test reuses val (unused in training)

    # no-straddle assertion: no family appears in both train and val
    assert not (set(train_fams) & val_fams), "family straddles train/val!"

    manifest = {
        "prefix": args.prefix, "seed": args.seed,
        "sources": args.datasets,
        "counts": {"train_traj": n_train, "val_traj": n_val,
                   "train_families": len(train_fams), "val_families": len(val_fams),
                   "incomplete_families_in_train": len(incomplete)},
        "val_families": sorted(val_fams),
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[split_lang_twins] {len(fam)} families "
          f"({len(complete)} complete, {len(incomplete)} incomplete)")
    print(f"  train: {n_train} traj / {len(train_fams)} families "
          f"(incl. {len(incomplete)} incomplete 3-twin)")
    print(f"  val:   {n_val} traj / {len(val_fams)} complete families")
    print(f"  test:  {n_test} traj (= val)")
    print(f"  -> {out_dir}")
    print(f"  split_manifest.json written (val family list pinned for reproducibility)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
