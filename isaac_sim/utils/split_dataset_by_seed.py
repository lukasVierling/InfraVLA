"""Split data-gen trajectories into train/val/test by seed range (symlinks).

Run AFTER a data-gen sweep completes. For each seed it picks the LATEST
trajectory dir whose result.json outcome == "reached" (so restart-duplicates
and failed/timeout/stuck runs are skipped), and symlinks it into the right
split. Reports any seeds with no successful trajectory.

  python utils/split_dataset_by_seed.py --date-tag <YYYY-MM-DD> --out <split_name>

Default split: train 0-199, val 200-219, test 220-239.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_GEN = _REPO_ROOT / "runs/data/sweeps/data_gen"
TRAINING_DATA = _REPO_ROOT / "runs/data/splits"

# Default seed -> split mapping. Disjoint, contiguous, 200/20/20.
# Specs are half-open 'START:STOP' strings (Python range() convention), overridable
# on the CLI via --train/--val/--test. e.g. the 500-traj sweep uses
# --train 0:500 --val 500:550 --test 550:600.
DEFAULT_SPLITS = {
    "train": "0:200",
    "val":   "200:220",
    "test":  "220:240",
}


def parse_seed_range(spec: str) -> range:
    """Parse a half-open 'START:STOP' seed range. e.g. '0:200' -> range(0, 200).

    Half-open matches Python range() and the original SPLITS convention:
    train '0:200' is seeds 0..199. STOP must be strictly greater than START.
    """
    try:
        start_s, stop_s = spec.split(":")
        start, stop = int(start_s), int(stop_s)
    except ValueError:
        raise SystemExit(f"bad seed range {spec!r}: expected 'START:STOP', e.g. '0:500'")
    if stop <= start:
        raise SystemExit(f"bad seed range {spec!r}: STOP ({stop}) must be > START ({start})")
    return range(start, stop)


def build_splits(train: str, val: str, test: str) -> dict[str, range]:
    """Build the split->range map from CLI specs, asserting the three are disjoint.

    Overlap would leak a training seed into val/test (or double-link one seed),
    silently corrupting the held-out evaluation — so we hard-fail instead of
    quietly producing a tainted split.
    """
    splits = {
        "train": parse_seed_range(train),
        "val":   parse_seed_range(val),
        "test":  parse_seed_range(test),
    }
    # Pairwise-disjoint check: record which split first claimed each seed; a
    # second claim means two ranges overlap.
    claimed_by: dict[int, str] = {}
    for name, seeds in splits.items():
        for s in seeds:
            if s in claimed_by:
                raise SystemExit(
                    f"seed {s} is in both '{claimed_by[s]}' and '{name}' splits — "
                    f"splits must be disjoint (a seed can't be train AND held-out)."
                )
            claimed_by[s] = name
    return splits


def latest_reached_dir(seed: int, runs_root: Path, prefix: str) -> Path | None:
    """Newest `seed`-dir under runs_root whose rollout reached the goal.

    Recursive glob (**) so a partitioned layout (e.g. warehouse_2x3/part*/) works
    as well as a flat sweep dir. dir name: {prefix}{seed:04d}__pose__YYYY-MM-DD_...
    Name sorts chronologically so the newest 'reached' dir wins (also dedups
    restart-duplicates — several dirs per seed).
    """
    cands = sorted(runs_root.glob(f"**/{prefix}{seed:04d}__pose__*"))
    for d in reversed(cands):                      # newest first (name sorts chronologically)
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            if json.loads(rj.read_text()).get("outcome") == "reached":
                return d
        except (json.JSONDecodeError, OSError):
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output subdir name under runs/data/splits/")
    ap.add_argument("--runs-root", default=str(DATA_GEN),
                    help="root to recursively glob for trajectory dirs. Default: the random_open sweep dir.")
    ap.add_argument("--prefix", default="random_open_seed",
                    help="dir-name prefix before the 4-digit seed, e.g. 'warehouse_2x3_'.")
    ap.add_argument("--date-tag", default="", help="(unused for globbing; kept for back-compat / labels)")
    ap.add_argument("--train", default=DEFAULT_SPLITS["train"],
                    help="train seed range START:STOP (half-open). Default 0:200.")
    ap.add_argument("--val", default=DEFAULT_SPLITS["val"],
                    help="val seed range START:STOP (half-open). Default 200:220.")
    ap.add_argument("--test", default=DEFAULT_SPLITS["test"],
                    help="test seed range START:STOP (half-open). Default 220:240.")
    args = ap.parse_args()

    splits = build_splits(args.train, args.val, args.test)
    runs_root = Path(args.runs_root).resolve()   # absolute -> symlink targets are absolute (relative would dangle)

    out_root = TRAINING_DATA / args.out
    missing: list[tuple[str, int]] = []
    for split, seeds in splits.items():
        sdir = out_root / split
        sdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for seed in seeds:
            d = latest_reached_dir(seed, runs_root, args.prefix)
            if d is None:
                missing.append((split, seed))
                continue
            link = sdir / d.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(d)                     # absolute target (d is absolute)
            n += 1
        print(f"  {split:5s}: {n}/{len(seeds)} linked  -> {sdir}")

    if missing:
        print(f"\n  MISSING/failed ({len(missing)} seeds — re-run these or they'll be absent):")
        for split, seed in missing:
            print(f"    {split} seed{seed:04d}")
        return 1
    print("\n  all seeds present ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
