
from __future__ import annotations

import argparse
import json
from pathlib import Path

REACH_ROUTING_M = 1.0
REACH_PRECISION_M = 0.3


def collect_runs(sweep_dir: Path) -> list[dict]:
    """One record per finished run dir (needs both result.json and log.jsonl)."""
    runs = []
    for rd in sorted(sweep_dir.glob("*__pose*")):
        result_path = rd / "result.json"
        log_path = rd / "log.jsonl"
        if not (result_path.is_file() and log_path.is_file()):
            continue                       # still running / torn — skip, count later
        result = json.loads(result_path.read_text())
        closest = None
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)["dist_to_goal"]
            if closest is None or d < closest:
                closest = d
        runs.append({
            "name": rd.name.split("__pose")[0].replace("warehouse_2x3_", ""),
            "outcome": result["outcome"],
            "final_dist": float(result["final_dist"]),
            "closest": float(closest),
        })
    return runs


def report(label: str, runs: list[dict], verbose: bool) -> None:
    n = len(runs)
    if n == 0:
        print(f"{label}: no finished runs")
        return
    n_route = sum(r["closest"] <= REACH_ROUTING_M for r in runs)
    n_prec = sum(r["final_dist"] <= REACH_PRECISION_M for r in runs)
    outcomes = {}
    for r in runs:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print(f"\n== {label} ==  n={n}")
    print(f"  reach@{REACH_ROUTING_M}m (closest approach) : {n_route:3d}  ({100*n_route/n:5.1f}%)")
    print(f"  reach@{REACH_PRECISION_M}m (final pose)      : {n_prec:3d}  ({100*n_prec/n:5.1f}%)")
    print(f"  outcomes: {outcomes}")
    fails = [r for r in runs if r["closest"] > REACH_ROUTING_M]
    if fails:
        print(f"  routing failures (@{REACH_ROUTING_M}m):")
        for r in fails:
            print(f"    {r['name']:<22} {r['outcome']:<8} "
                  f"closest={r['closest']:6.2f}m final={r['final_dist']:6.2f}m")
    if verbose:
        print(f"  {'run':<22} {'outcome':<8} {'closest':>8} {'final':>8}")
        for r in runs:
            print(f"  {r['name']:<22} {r['outcome']:<8} "
                  f"{r['closest']:8.2f} {r['final_dist']:8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sweep_dirs", nargs="+", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print every run's closest/final distances")
    args = ap.parse_args()

    all_runs = []
    for d in args.sweep_dirs:
        runs = collect_runs(d.resolve())
        report(d.name, runs, args.verbose)
        all_runs.extend(runs)
    if len(args.sweep_dirs) > 1:
        report("COMBINED", all_runs, verbose=False)


if __name__ == "__main__":
    main()
