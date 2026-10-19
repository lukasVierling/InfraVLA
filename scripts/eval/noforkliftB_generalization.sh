#!/usr/bin/env bash
# Closed-loop GENERALIZATION eval of the pairing-holdout model (trained with every
# forklift@B family removed — the model never saw a forklift in the left aisle).
#
# Two scene sets, four reported splits:
#   A. heldoutFKB (140 scenes = the 35 forklift@B families x 4 twins)
#        S3 ULTIMATE-OOD      : 35  target IS the forklift in B  (unseen pairing = the goal)
#        S2 OODscene/seenTgt  : 105 forklift@B is IN the scene, but the target is
#                               barrel/box/cone in C/E/F (a pairing the model DID see)
#   B. standard 52-scene test set
#        S1 in-distribution   : 44  no forklift@B anywhere
#        S1b test-OOD         :  8  forklift@B present (also OOD for this model)
# S2 vs S3 is a within-family paired contrast: same scene, seen vs unseen target.
# CAP=1 (a langv2 eval proc is 38-53 GB; 2+ OOM the card).
set -uo pipefail
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python}"
SW=$ROOT/runs/evaluation/sweeps/langv2test_paper
LOG=$ROOT/runs/evaluation/noforkliftB_generalization.log
CKPT="2026-07-27_cctv4_avgpool64_langv2_yawcrux_pre3_noforkliftB_15k_v0/checkpoints/step_13000"
say(){ echo "[nfB $(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== noforkliftB GENERALIZATION EVAL START (192 rollouts, CAP=1) ==="
cd "$ROOT/isaac_sim"
mkdir -p "$SW/nfB_hfkb" "$SW/nfB_std"
for p in 1 2 3 4 5 6 7; do
  say "start lv2t_nfB_hfkb_p$p"
  bash launchers/run_eval_partition.sh "lv2t_nfB_hfkb_p$p" > "$SW/nfB_hfkb/p${p}.log" 2>&1
  say "done  lv2t_nfB_hfkb_p$p ($?)"
done
for p in 1 2 3; do
  say "start lv2t_nfB_std_p$p"
  bash launchers/run_eval_partition.sh "lv2t_nfB_std_p$p" > "$SW/nfB_std/p${p}.log" 2>&1
  say "done  lv2t_nfB_std_p$p ($?)"
done

say "=== overlay check (must be the noforkliftB checkpoint) ==="
got=$(grep -h "load_model.*overlay" "$SW"/nfB_*/p*.log 2>/dev/null | sed 's/.*overlay from //' | sort -u | tr '\n' ' ')
echo "$got" | grep -q "$CKPT" && say "  OK -> $CKPT" || say "  ⚠️ MISMATCH: $got"

say "=== RESULTS ==="
"$PY" - "$SW" <<'PYR' 2>&1 | tee -a "$LOG"
import json, pathlib, sys, collections
SW = pathlib.Path(sys.argv[1])
hm = {m["name"]: m for m in json.loads(pathlib.Path(
    "isaac_sim/configs/experiment/datagen_2x3_langv2_heldoutFKB_manifest.json").read_text())["scenes"]}
tm = {m["name"]: m for m in json.loads(pathlib.Path(
    "isaac_sim/configs/experiment/datagen_2x3_langv2test_manifest.json").read_text())["scenes"]}
def load(root):
    rows = {}
    for sj in pathlib.Path(root).glob("**/summary.jsonl"):
        for ln in sj.read_text().splitlines():
            try:
                r = json.loads(ln); d = r.get("final_dist")
                rows[r["name"]] = (d if isinstance(d, (int, float)) else 9.99, r.get("outcome"))
            except Exception: pass
    return rows
def closest(root):
    out = {}
    for lj in pathlib.Path(root).glob("**/log.jsonl"):
        nm = lj.parent.name.split("__lang__")[0].split("__pose__")[0]
        ds = [json.loads(l).get("dist_to_goal") for l in lj.read_text().splitlines() if l.strip()]
        ds = [x for x in ds if isinstance(x, (int, float))]
        if ds: out[nm] = min(ds)
    return out
H, Hc = load(SW/"nfB_hfkb"), closest(SW/"nfB_hfkb")
T, Tc = load(SW/"nfB_std"),  closest(SW/"nfB_std")
def bucket(rows, cl, keep):
    sel = {k: v for k, v in rows.items() if keep(k)}
    n = len(sel)
    if not n: return None
    f03 = sum(1 for d, _ in sel.values() if d <= 0.3)
    f10 = sum(1 for d, _ in sel.values() if d <= 1.0)
    c = [cl[k] for k in sel if k in cl]
    c10 = sum(1 for v in c if v <= 1.0)
    oc = collections.Counter(o for _, o in sel.values())
    return n, f03/n, f10/n, (c10/len(c) if c else float("nan")), dict(oc)
def ult(k):  # target IS the forklift in B
    m = hm.get(k, {}); return m.get("goal_zone") == "B" and m.get("target_phrase") == "forklift"
splits = [
  ("S1  in-distribution (std test, no fk@B)", bucket(T, Tc, lambda k: tm.get(k,{}).get("zone_objects",{}).get("B") != "forklift")),
  ("S1b std test, fk@B present",              bucket(T, Tc, lambda k: tm.get(k,{}).get("zone_objects",{}).get("B") == "forklift")),
  ("S2  OOD scene / SEEN target",             bucket(H, Hc, lambda k: not ult(k))),
  ("S3  ULTIMATE OOD (go to fk@B)",           bucket(H, Hc, ult)),
  ("--  std test overall (52)",               bucket(T, Tc, lambda k: True)),
  ("--  heldout set overall (140)",           bucket(H, Hc, lambda k: True)),
]
print(f"\n{'split':42s} {'n':>4s} {'FINAL@0.3':>10s} {'FINAL@1m':>9s} {'CLOSEST@1m':>11s}")
print("-"*84)
out = {}
for name, b in splits:
    if not b: print(f"{name:42s}  (no data)"); continue
    n, f03, f10, c10, oc = b
    print(f"{name:42s} {n:>4d} {f03:>10.3f} {f10:>9.3f} {c10:>11.3f}")
    out[name.strip()] = dict(n=n, final_at_0_3=f03, final_at_1=f10, closest_at_1=c10, outcomes=oc)
print("\nopen-loop reference (decision probe): 0.86 correct left-turn on the 35 unseen forklift@B, 1.00 on seen pairings")
(SW/"nfB_REACH_summary.json").write_text(json.dumps(out, indent=1))
print(f"wrote {SW}/nfB_REACH_summary.json")
PYR
say "=== DONE ==="
