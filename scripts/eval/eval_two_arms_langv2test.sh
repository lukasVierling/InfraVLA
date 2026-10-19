#!/usr/bin/env bash
set -uo pipefail
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
PY="${PYTHON:-python}"
MAN=$ROOT/isaac_sim/configs/experiment/datagen_2x3_langv2test_manifest.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ echo "[2arm-eval $(date '+%F %T')] $*"; }

eval_one(){  # name  policy_config  label  eval_root
  local name="$1" policy="$2" label="$3" evalroot="$4"
  say "===== EVAL $name (policy=$policy) ====="
  mkdir -p "$evalroot"
  PYTHONPATH=$ROOT "$PY" $ROOT/isaac_sim/run_builders/build_langv2_eval.py \
    --manifest "$MAN" --policy "$policy" --label "$label" \
    --runs-root "$evalroot" --per-shard 13 --success-dist 0.3 \
    --pattern "move toward the {x}" 2>&1 | sed 's/^/  /'
  cd "$ROOT/isaac_sim"
  : > "$evalroot/_exits.log"
  for exp in $(ls $ROOT/isaac_sim/configs/experiment/${label}_p*.yaml | sed -E 's#.*/([^/]+)\.yaml#\1#' | sort -V); do
    say "start shard $exp (gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
    bash launchers/run_eval_partition.sh "$exp" > "$evalroot/${exp}.log" 2>&1   # BLOCKS -> CAP=1
    echo "$exp EXIT=$? $(date '+%T')" >> "$evalroot/_exits.log"
    say "done shard $exp"
    sleep 8   # let the CUDA context fully release before the next model load
  done
  say "$name shards done: $(tr '\n' ' ' < "$evalroot/_exits.log")"
}

AP16_ROOT=$ROOT/runs/evaluation/sweeps/langv2test_avgpool16_eval
FRZ_ROOT=$ROOT/runs/evaluation/sweeps/langv2test_frozen_eval
eval_one avgpool16 omnivla_langv2_avgpool16_best langv2test_avgpool16 "$AP16_ROOT"
eval_one frozen    omnivla_langv2_frozen_best    langv2test_frozen    "$FRZ_ROOT"

say "===== AGGREGATE + HEAD-TO-HEAD ====="
"$PY" - "$MAN" "$AP16_ROOT" "$FRZ_ROOT" <<'PYIN'
import json, sys, pathlib, collections
man = {m["name"]: m for m in json.loads(pathlib.Path(sys.argv[1]).read_text())["scenes"]}
def agg(root):
    rows=[]
    for sj in pathlib.Path(root).glob("**/summary.jsonl"):
        for ln in sj.read_text().splitlines():
            try: rows.append(json.loads(ln))
            except Exception: pass
    n=len(rows)
    if not n: return None
    r03=sum(1 for r in rows if r.get("final_dist",9)<=0.3)
    r10=sum(1 for r in rows if r.get("final_dist",9)<=1.0)
    oc=collections.Counter(r.get("outcome") for r in rows)
    bz, bz03, bz10 = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        z=man.get(r.get("name",""),{}).get("goal_zone","?"); bz[z]+=1
        bz03[z]+= r.get("final_dist",9)<=0.3; bz10[z]+= r.get("final_dist",9)<=1.0
    return dict(n=n,r03=r03,r10=r10,oc=dict(oc),
               byz={z:(bz03[z],bz10[z],bz[z]) for z in sorted(bz)})
arms=[("avgpool64 (baseline)",None),
      ("avgpool16", sys.argv[2]),
      ("frozen-backbone", sys.argv[3])]
print(f"\n{'arm':22s} {'n':>4} {'reach@0.3':>10} {'reach@1m':>9}   outcomes")
print("-"*80)
BASE={"n":37,"r03":26,"r10":37,"oc":{"reached":26,"stuck":8,"exception":2,"timeout":1}}
for name,root in arms:
    a = BASE if root is None else agg(root)
    if a is None: print(f"{name:22s}  (no results yet)"); continue
    print(f"{name:22s} {a['n']:>4} {a['r03']/a['n']:>9.3f} {a['r10']/a['n']:>8.3f}   {a['oc']}")
    if root is not None and a.get("byz"):
        for z,(o3,o10,tot) in a["byz"].items():
            print(f"    aisle {z}: reach@0.3 {o3}/{tot}={o3/tot:.2f}  reach@1m {o10}/{tot}={o10/tot:.2f}")
        out=pathlib.Path(root)/"REACH_summary.json"; out.write_text(json.dumps(a,indent=2))
        print(f"    wrote {out}")
PYIN
say "===== TWO-ARM EVAL COMPLETE ====="
