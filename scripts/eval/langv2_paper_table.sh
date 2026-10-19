#!/usr/bin/env bash
# Experiment-1 paper table: closed-loop langv2 TEST eval (52 scenes) for arms
# A0 (avgpool64), B1 (cctvfirst), C1 (shared encoder) and E1 (no upsampling);
# A1 (avgpool16) and D3 (frozen backbone) are read from existing sweeps.
#
# 12 jobs (4 arms x 3 shards) run sequentially (CAP=1): one langv2 eval process
# reaches 38-53 GB, so concurrent shards OOM a 95 GB GPU and every rollout
# returns outcome=exception. Each shard's overlay log line is checked against
# the arm's checkpoint, since a stale model reload fails silently.
set -uo pipefail
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python}"
SW=$ROOT/runs/evaluation/sweeps/langv2test_paper
LOG=$ROOT/runs/evaluation/langv2_paper_table.log
CAP=1
say(){ echo "[lv2-table $(date '+%F %T')] $*" | tee -a "$LOG"; }

declare -A CKPT=(
  [A0full]="2026-07-23_cctv4_avgpool64_langv2_yawcrux_pre3_v0/checkpoints/step_15000"
  [B1]="2026-07-26_cctv4_avgpool64_langv2_yawcrux_pre3_cctvfirst_15k_v0/checkpoints/step_13000"
  [C1]="2026-07-27_cctv4_avgpool64_langv2_sharedenc_15k_v0/checkpoints/step_15000"
  [E1]="2026-07-27_cctv4_avgpool64_langv2_noupsample_cfval_15k_v0/checkpoints/step_14000"
)
say "=== START: 4 arms x 3 shards, ${CAP}-slot pool ==="
cd "$ROOT/isaac_sim"
for tag in A0full B1 C1 E1; do
  mkdir -p "$SW/$tag"
  for p in 1 2 3; do
    while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do sleep 20; done
    exp="lv2t_${tag}_p${p}"
    say "start $exp"
    ( bash launchers/run_eval_partition.sh "$exp" > "$SW/$tag/p${p}.log" 2>&1; \
      echo "$exp EXIT=$? $(date '+%T')" >> "$SW/_exits.log" ) &
    sleep 25
  done
done
wait
say "all shards finished: $(tr '\n' ' ' < "$SW/_exits.log" 2>/dev/null)"

# --- overlay verification: did every shard load its OWN checkpoint? ---
say "=== overlay verification ==="
for tag in A0full B1 C1 E1; do
  want="${CKPT[$tag]}"
  got=$(grep -h "load_model.*overlay" "$SW/$tag"/p*.log 2>/dev/null | sed 's/.*overlay from //' | sort -u | tr '\n' ' ')
  if echo "$got" | grep -q "${want}"; then say "  $tag OK -> $want"
  else say "  $tag ⚠️ MISMATCH: wanted $want, got: $got"; fi
done

# --- report ---
say "=== EXPERIMENT 1 TABLE (langv2 test, 52 scenes) ==="
"$PY" - "$SW" <<'PYR' 2>&1 | tee -a "$LOG"
import json, pathlib, sys, collections
SW = pathlib.Path(sys.argv[1])
# arms whose complete sweeps already exist elsewhere
EXTRA = {"A1 avgpool16": "runs/evaluation/sweeps/langv2test_avgpool16_eval",
         "D3 frozen-backbone": "runs/evaluation/sweeps/langv2test_frozen_eval"}
NAMES = {"A0full": "A0 avgpool64 (headline)", "B1": "B1 cctvfirst (reorder)",
         "C1": "C1 shared encoder", "E1": "E1 no upsampling"}
def load(root):
    rows = {}
    for sj in pathlib.Path(root).glob("**/summary.jsonl"):
        for ln in sj.read_text().splitlines():
            try:
                r = json.loads(ln)
                d = r.get("final_dist")
                rows[r["name"]] = (d if isinstance(d, (int, float)) else 9.99, r.get("outcome"))
            except Exception:
                pass
    return rows
arms = [(NAMES[t], SW / t) for t in ("A0full", "B1", "C1", "E1")] + list(EXTRA.items())
print(f"\n{'arm':26s} {'n':>5s} {'reach@0.3':>10s} {'reach@1m':>9s}   outcomes")
print("-" * 82)
for name, root in arms:
    rows = load(root)
    n = len(rows)
    if not n:
        print(f"{name:26s} {'--':>5s}   (no results)"); continue
    r03 = sum(1 for d, _ in rows.values() if d <= 0.3)
    r10 = sum(1 for d, _ in rows.values() if d <= 1.0)
    oc = collections.Counter(o for _, o in rows.values())
    print(f"{name:26s} {f'{n}/52':>5s} {r03/n:>9.3f} {r10/n:>8.3f}   {dict(oc)}")
    (root / "REACH_summary.json").write_text(json.dumps(
        {"arm": name, "n": n, "reach_at_0.3": r03 / n, "reach_at_1.0": r10 / n,
         "reached": r03, "outcomes": dict(oc)}, indent=1))
# stability check: A0 vs an earlier partial sweep (same model, same scenes)
old = load("runs/evaluation/sweeps/langv2test_eval_v4")
new = load(SW / "A0full")
common = sorted(set(old) & set(new))
if common:
    flips = sum(1 for k in common if (old[k][0] <= 0.3) != (new[k][0] <= 0.3))
    import statistics as st
    diffs = [abs(old[k][0] - new[k][0]) for k in common]
    print(f"\nA0 STABILITY (same model+scenes, two independent sweeps, n={len(common)}):")
    print(f"  reach@0.3 verdict flips: {flips}/{len(common)} = {flips/len(common):.1%}"
          f"  | median |Δfinal| {st.median(diffs)*100:.1f} cm")
PYR
say "=== DONE ==="
