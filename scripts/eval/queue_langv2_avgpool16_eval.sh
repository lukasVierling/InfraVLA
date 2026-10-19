#!/usr/bin/env bash
set -uo pipefail
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
RUN=$ROOT/runs/training/runs/2026-07-25_cctv4_avgpool16_langv2_yawcrux_pre3_v0
CKPTS=$RUN/checkpoints
POLICY_TMPL=$ROOT/isaac_sim/configs/policy/omnivla_lang80_7b_step6000.yaml
POLICY_OUT=$ROOT/isaac_sim/configs/policy/omnivla_langv2_avgpool16_best.yaml
EVAL_ROOT=$ROOT/runs/evaluation/sweeps/langv2test_avgpool16_eval
PY="${PYTHON:-python}"
CAP=1  # each eval proc ~38GB; even 2 OOMs the 95GB card during load (2026-07-23)
mkdir -p "$EVAL_ROOT"
say(){ echo "[langv2-eval $(date '+%F %T')] $*"; }

# 1) wait for training to finish: step_15000 checkpoint present AND no live trainer
say "waiting for pre3 training to finish (step_15000 + process exit)..."
i=0
while true; do
  live=$(pgrep -fc "train_omnivla_sim_cctv_v2.py .*avgpool16_langv2_yawcrux_pre3" || true)
  [ -d "$CKPTS/step_15000" ] && [ "${live:-0}" -eq 0 ] && break
  i=$((i+1)); [ $((i % 30)) -eq 0 ] && say "still waiting (${i} min); step_15000=$([ -d "$CKPTS/step_15000" ] && echo yes || echo no) trainer_live=${live:-0}"
  sleep 60
done
say "training done. picking best-val checkpoint..."

# 2) best checkpoint = min loss_main over surviving step_*/val_metrics.json
BEST=$("$PY" - "$CKPTS" << 'PYIN'
import json, sys, pathlib
ck = pathlib.Path(sys.argv[1])
best = None
for d in ck.glob("step_*/val_metrics.json"):
    step = int(d.parent.name.split("_")[1])
    try: lm = json.loads(d.read_text())["loss_main"]
    except Exception: continue
    if best is None or lm < best[1]:
        best = (step, lm)
print(f"{best[0]} {best[1]:.6f}" if best else "NONE 0")
PYIN
)
BEST_STEP=$(echo "$BEST" | awk '{print $1}')
BEST_LOSS=$(echo "$BEST" | awk '{print $2}')
if [ "$BEST_STEP" = "NONE" ]; then say "NO checkpoint with val_metrics — abort"; exit 1; fi
say "best checkpoint: step_${BEST_STEP} (val loss_main=${BEST_LOSS})"

# 3) write the eval policy config from the template, repointed at the best ckpt
sed -E \
  -e "s#checkpoint_dir: \"[^\"]*\"#checkpoint_dir: \"$CKPTS/step_${BEST_STEP}\"#" \
  -e "s#^  step: [0-9]+#  step: ${BEST_STEP}#" \
  -e "s#out_tokens: 64#out_tokens: 16#" \
  "$POLICY_TMPL" > "$POLICY_OUT"
say "wrote policy config $POLICY_OUT"

# 4) generate eval shard experiment yamls (canonical pattern, success_dist 0.3)
PYTHONPATH=$ROOT "$PY" $ROOT/isaac_sim/run_builders/build_langv2_eval.py \
  --manifest $ROOT/isaac_sim/configs/experiment/datagen_2x3_langv2test_manifest.json \
  --policy omnivla_langv2_avgpool16_best --label langv2test_avgpool16 \
  --runs-root "$EVAL_ROOT" --per-shard 8 --success-dist 0.3 \
  --pattern "move toward the {x}" 2>&1 | sed 's/^/  /'

# 5) run the 7 shards through the generic eval launcher (CAP-parallel job pool)
cd "$ROOT/isaac_sim"
SHARDS=$(ls $ROOT/isaac_sim/configs/experiment/langv2test_avgpool16_p*.yaml | sed -E 's#.*/([^/]+)\.yaml#\1#' | sort -V)
say "launching eval shards (cap=$CAP): $SHARDS"
: > "$EVAL_ROOT/_exits.log"
for exp in $SHARDS; do
  while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do sleep 15; done
  say "start $exp"
  ( bash launchers/run_eval_partition.sh "$exp" > "$EVAL_ROOT/${exp}.log" 2>&1; \
    echo "$exp EXIT=$? $(date '+%T')" >> "$EVAL_ROOT/_exits.log" ) &
  sleep 25
done
wait
say "all shards done: $(cat $EVAL_ROOT/_exits.log | tr '\n' ' ')"

# 6) aggregate reach@0.3 overall + by target aisle
"$PY" - "$EVAL_ROOT" "$ROOT/isaac_sim/configs/experiment/datagen_2x3_langv2test_manifest.json" "$BEST_STEP" << 'PYIN'
import json, sys, pathlib, collections
root, manp, step = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
man = {m["name"]: m for m in json.loads(pathlib.Path(manp).read_text())["scenes"]}
rows = []
for sj in root.glob("**/summary.jsonl"):
    for line in sj.read_text().splitlines():
        try: rows.append(json.loads(line))
        except Exception: pass
n = len(rows); reached = sum(1 for r in rows if r.get("outcome") == "reached")
print(f"\n===== langv2 closed-loop TEST eval (best ckpt step_{step}) =====")
print(f"  reach@0.3m: {reached}/{n} = {reached/n:.3f}" if n else "  NO RESULTS")
by_zone = collections.Counter(); by_zone_ok = collections.Counter()
by_outcome = collections.Counter(r.get("outcome") for r in rows)
for r in rows:
    z = man.get(r.get("name",""), {}).get("goal_zone", "?")
    by_zone[z]+=1; by_zone_ok[z]+= (r.get("outcome")=="reached")
print("  outcomes:", dict(by_outcome))
print("  reach@0.3 by target aisle:")
for z in sorted(by_zone):
    print(f"    {z}: {by_zone_ok[z]}/{by_zone[z]} = {by_zone_ok[z]/by_zone[z]:.2f}")
out = root / f"REACH_step{step}.json"
out.write_text(json.dumps({"step": int(step), "n": n, "reached": reached,
    "reach_at_0.3": (reached/n if n else None),
    "by_zone": {z: [by_zone_ok[z], by_zone[z]] for z in by_zone},
    "outcomes": dict(by_outcome)}, indent=2))
print(f"  wrote {out}")
PYIN
say "=== langv2 TEST EVAL COMPLETE ==="
