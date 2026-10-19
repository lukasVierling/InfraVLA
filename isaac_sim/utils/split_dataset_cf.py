"""Split the COUNTERFACTUAL data-gen campaign into train/val/test (symlinks).

Sibling of split_dataset_by_seed.py for the cf campaign, where the seed-range
tool cannot work: run dirs are named warehouse_2x3_0600_cfB__pose__... (a
variant suffix sits between the index and __pose__), scenes come in FAMILIES
(one base => cf0 + blocked variants) that must never straddle a split, and
vacuous variants (blocked twin whose route equals the unblocked route — no
counterfactual signal) must be dropped from TRAIN only.

Driven by the generator's manifest (datagen_2x3_cf_manifest.json): each scene
entry carries name, base, and the vacuous flag. Allocation is by BASE index
range (half-open, like the seed tool), so whole families stay atomic by
construction.

  python isaac_sim/utils/split_dataset_cf.py \
      --out 2026-07-02_warehouse_2x3_cf \
      --runs-root runs/data_gen/warehouse_2x3_cf_2026-07-01 \
      --train-bases 600:790 --val-bases 790:809 --test-bases 809:828 \
      --allow-missing 6

Missing scenes (deterministically-stuck rollouts) are listed loudly; exit is
non-zero if more than --allow-missing are absent.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

# Reuse the seed tool's range parsing + disjointness guard (base indices are
# just as range-shaped as seeds).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_dataset_by_seed import parse_seed_range, build_splits, TRAINING_DATA

_BASE_RE = re.compile(r"_(\d{4})$")   # base index from "warehouse_2x3_0600"


def base_index(scene: dict) -> int:
    m = _BASE_RE.search(scene["base"])
    if not m:
        raise SystemExit(f"cannot parse base index from {scene['base']!r}")
    return int(m.group(1))


def allocate(scenes: list[dict], splits: dict[str, range]) -> dict[str, list[dict]]:
    """Assign every manifest scene to a split by base index; drop vacuous from
    TRAIN only. Family atomicity holds by construction (allocation keys on the
    base index all family members share)."""
    out: dict[str, list[dict]] = {s: [] for s in splits}
    dropped_vacuous: list[str] = []
    unassigned: list[str] = []
    for sc in scenes:
        b = base_index(sc)
        for split, rng in splits.items():
            if b in rng:
                if split == "train" and sc.get("vacuous") is True:
                    dropped_vacuous.append(sc["name"])
                else:
                    out[split].append(sc)
                break
        else:
            unassigned.append(sc["name"])
    if unassigned:
        raise SystemExit(f"{len(unassigned)} scenes fall outside every base range: "
                         f"{unassigned[:5]}... — ranges must cover the manifest.")
    out["_dropped_vacuous"] = dropped_vacuous  # type: ignore[assignment]
    return out


def index_run_dirs(runs_root: Path) -> dict[str, list[Path]]:
    """ONE depth-2 scan of the campaign tree -> {scene_name: [run dirs]}.

    Run dirs sit at depth 2 (cf_pN/<run>, cf_retry/<run>, _disrupted_*/<run>).
    A per-scene recursive glob (`**/name__pose__*`) walks the ~365k-file image
    tree once PER SCENE — minutes over NFS; this single shallow pass is seconds.
    """
    idx: dict[str, list[Path]] = {}
    for part in runs_root.iterdir():
        if not part.is_dir() or part.name == "_launch_logs":
            continue
        for d in part.iterdir():
            # name format: <scene>__pose__YYYY-MM-DD_HH-MM-SS
            if "__pose__" not in d.name or not d.is_dir():
                continue
            scene = d.name.split("__pose__")[0]
            idx.setdefault(scene, []).append(d)
    for dirs in idx.values():
        dirs.sort()          # name sorts chronologically -> newest last
    return idx


def latest_reached_dir_by_name(name: str, index: dict[str, list[Path]]) -> Path | None:
    """Newest run dir for scene `name` whose rollout reached the goal.

    Same newest-reached semantics as split_dataset_by_seed.latest_reached_dir
    (a retry in cf_retry/ beats an older stuck attempt), but reads candidates
    from the prebuilt index instead of a per-name recursive glob.
    """
    for d in reversed(index.get(name, [])):
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
    ap.add_argument("--out", required=True, help="output subdir under runs/data/splits/")
    ap.add_argument("--manifest", default="isaac_sim/configs/experiment/datagen_2x3_cfv4_manifest.json")
    ap.add_argument("--runs-root", default="runs/data_gen/warehouse_2x3_cf_2026-07-01")
    ap.add_argument("--train-bases", default="600:790", help="train BASE range, half-open")
    ap.add_argument("--val-bases",   default="790:809", help="val BASE range, half-open")
    ap.add_argument("--test-bases",  default="809:828", help="test BASE range, half-open")
    ap.add_argument("--allow-missing", type=int, default=0,
                    help="exit 0 if at most this many scenes have no reached run (stuck rollouts)")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    scenes = manifest["scenes"]
    splits = build_splits(args.train_bases, args.val_bases, args.test_bases)
    alloc = allocate(scenes, splits)
    dropped = alloc.pop("_dropped_vacuous")

    runs_root = Path(args.runs_root).resolve()   # absolute -> symlink targets are absolute
    out_root = TRAINING_DATA / args.out
    index = index_run_dirs(runs_root)
    print(f"indexed {sum(len(v) for v in index.values())} run dirs for {len(index)} scenes")

    print(f"manifest: {len(scenes)} scenes; vacuous dropped from train: {len(dropped)}")
    missing: list[tuple[str, str]] = []
    for split, members in alloc.items():
        sdir = out_root / split
        sdir.mkdir(parents=True, exist_ok=True)
        n = 0
        vac_kept = sum(1 for sc in members if sc.get("vacuous") is True)
        for sc in members:
            d = latest_reached_dir_by_name(sc["name"], index)
            if d is None:
                missing.append((split, sc["name"]))
                continue
            link = sdir / d.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(d)
            n += 1
        print(f"  {split:5s}: {n}/{len(members)} linked (vacuous kept: {vac_kept})  -> {sdir}")

    if dropped:
        print(f"  train-dropped vacuous ({len(dropped)}): {' '.join(dropped)}")
    if missing:
        print(f"\n  MISSING — no reached run ({len(missing)}; allow={args.allow_missing}):")
        for split, name in missing:
            print(f"    {split} {name}")
        return 0 if len(missing) <= args.allow_missing else 1
    print("\n  all scenes present ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
