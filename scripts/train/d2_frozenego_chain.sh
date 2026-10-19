#!/usr/bin/env bash
set -uo pipefail
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python}"
CFG=$ROOT/isaac_sim/train/configs/2026-07-26_cctv4_sepproj_frozenvision_langv2_yawcrux_pre3_15k.yaml
RUN=2026-07-26_cctv4_sepproj_frozenvision_langv2_yawcrux_pre3_15k_v0
CKPTS=$ROOT/runs/training/runs/$RUN/checkpoints
POL=omnivla_langv2_frozenego_best
SW=$ROOT/runs/evaluation/sweeps/langv2test_paper/D2
LOG=$ROOT/runs/training/2026-08-09_d2_frozenego.stdout.log
say(){ echo "[d2 $(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== D2 FROZEN-EGO CHAIN START ==="
# ---- A. train (resilient against transient CUDA faults) ----
for attempt in 1 2 3 4 5 6; do
  [ -d "$CKPTS/step_15000" ] && { say "A: step_15000 present — training DONE"; break; }
  last=$(ls -d "$CKPTS"/step_* 2>/dev/null | sed -E 's#.*/step_0*([0-9]+)$#\1#' | sort -n | tail -1)
  if [ -n "$last" ]; then RES=(--resume-step "$last"); say "A: attempt $attempt resume step_$last"
  else RES=(); say "A: attempt $attempt FRESH"; fi
  ( cd "$ROOT" && PYTHONPATH="$ROOT" "$PY" vla-scripts/train_omnivla_sim_cctv_v2.py \
      --config "$CFG" "${RES[@]}" ) >> "$LOG" 2>&1
  say "A: attempt $attempt exited ($?)"
  [ -d "$CKPTS/step_15000" ] && break
  sleep 20
done
[ -d "$CKPTS/step_15000" ] || { say "ABORT: training never reached 15000"; exit 1; }

# ---- B. best-val checkpoint ----
BEST=$("$PY" - "$CKPTS" <<'PYB'
import json, sys, pathlib
b=None
for d in pathlib.Path(sys.argv[1]).glob("step_*/val_metrics.json"):
    try: lm=json.loads(d.read_text())["loss_main"]
    except Exception: continue
    s=int(d.parent.name.split("_")[1])
    if b is None or lm<b[1]: b=(s,lm)
print(f"{b[0]} {b[1]:.4f}" if b else "NONE 0")
PYB
)
STEP=$(echo "$BEST" | awk '{print $1}'); VL=$(echo "$BEST" | awk '{print $2}')
[ "$STEP" = "NONE" ] && { say "ABORT: no val_metrics"; exit 1; }
say "B: best-val step_$STEP (loss_main=$VL)"

# ---- C. policy config (frozen-ego arch: sepproj on, cctv encoder on) ----
sed -E -e "s#checkpoint_dir: \"[^\"]*\"#checkpoint_dir: \"$CKPTS/step_$STEP\"#" \
       -e "s#^  step: [0-9]+#  step: $STEP#" \
  "$ROOT/isaac_sim/configs/policy/omnivla_langv2_frozen_best.yaml" \
  > "$ROOT/isaac_sim/configs/policy/${POL}.yaml"
say "C: wrote policy ${POL} (cloned from the frozen-backbone policy: sepproj+cctv encoder, repointed)"

# ---- D. shards + eval (CAP=1: a langv2 eval proc is 38-53 GB) ----
mkdir -p "$SW"
PYTHONPATH=$ROOT "$PY" $ROOT/isaac_sim/run_builders/build_langv2_eval.py \
  --manifest $ROOT/isaac_sim/configs/experiment/datagen_2x3_langv2test_manifest.json \
  --policy "$POL" --label lv2t_D2 --runs-root "$SW" \
  --per-shard 18 --success-dist 0.3 --pattern "move toward the {x}" 2>&1 | tail -1 | tee -a "$LOG"
cd "$ROOT/isaac_sim"
for p in 1 2 3; do
  say "D: start lv2t_D2_p$p"
  bash launchers/run_eval_partition.sh "lv2t_D2_p$p" > "$SW/p${p}.log" 2>&1
  say "D: done lv2t_D2_p$p ($?)"
done
say "D: overlay check -> $(grep -h 'load_model.*overlay' $SW/p*.log 2>/dev/null | sed 's/.*overlay from //' | sort -u | tr '\n' ' ')"

# ---- E. report, in the Experiment-1 format ----
say "=== D2 RESULT (langv2 test, 52 scenes) ==="
"$PY" - "$SW" <<'PYR' 2>&1 | tee -a "$LOG"
import json, pathlib, sys, collections
SW=pathlib.Path(sys.argv[1])
rows={}
for sj in SW.glob("**/summary.jsonl"):
    for ln in sj.read_text().splitlines():
        try:
            r=json.loads(ln); d=r.get("final_dist")
            rows[r["name"]]=(d if isinstance(d,(int,float)) else 9.99, r.get("outcome"))
        except Exception: pass
cl={}
for lj in SW.glob("**/log.jsonl"):
    nm=lj.parent.name.split("__lang__")[0].split("__pose__")[0]
    ds=[json.loads(l).get("dist_to_goal") for l in lj.read_text().splitlines() if l.strip()]
    ds=[x for x in ds if isinstance(x,(int,float))]
    if ds: cl[nm]=min(ds)
n=len(rows)
if n:
    f03=sum(1 for d,_ in rows.values() if d<=0.3); f10=sum(1 for d,_ in rows.values() if d<=1.0)
    c10=sum(1 for v in cl.values() if v<=1.0)
    oc=collections.Counter(o for _,o in rows.values())
    print(f"  D2 frozen-ego   n={n}/52   FINAL@0.3={f03/n:.3f}   FINAL@1m={f10/n:.3f}   CLOSEST@1m={c10/max(1,len(cl)):.3f}")
    print(f"  outcomes: {dict(oc)}")
    print(f"  ladder ref -> A0 full-FT 0.712 | D2 (this) | D3 frozen-backbone 0.154")
    (SW/"REACH_summary.json").write_text(json.dumps(
        {"arm":"D2 frozen-ego","n":n,"final_at_0.3":f03/n,"final_at_1.0":f10/n,
         "closest_at_1.0":c10/max(1,len(cl)),"outcomes":dict(oc)},indent=1))
else:
    print("  NO RESULTS")
PYR
say "=== D2 CHAIN COMPLETE ==="
