#!/usr/bin/env bash
# B-CONTROL: does the noforkliftB model fail at aisle B in general, or only when the
# TARGET is the never-seen forklift@B pairing?
#
#   S3  ULTIMATE OOD    forklift sits in B, target = that forklift   -> 0/26 entered B
#   S2  OOD scene       forklift sits in B, target is in C/E/F       (held-out sweep)
#   BC  B-CONTROL       barrel/box/cone in B, target = that object   <- THIS RUN (11)
#
# BC uses unseen TEST scenes, so only the *pairing* is in-distribution. If BC enters B
# reliably, the failure is the unseen pairing; if BC also fails, the model is simply bad
# at aisle B (it lost 35 families of B training data) and the OOD story dissolves.
#
# Preempts the running generalization sweep via the launcher queue gate (CAP=1: a langv2
# eval proc peaks at 38-53 GB, two OOM the card). The in-flight shard is left untouched;
# we wait for it, take the GPU, then drop the gate so the sweep resumes at p7.
set -uo pipefail
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python}"
SW=$ROOT/runs/evaluation/sweeps/langv2test_paper
GATE=$ROOT/runs/evaluation/EVAL_QUEUE_PAUSED
INFLIGHT=${1:?usage: nfB_bcontrol.sh <pid-of-inflight-eval-payload>}
LOG=$ROOT/runs/evaluation/nfB_bcontrol.log
say(){ echo "[bctrl $(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== B-CONTROL START (11 scenes) — waiting for in-flight eval pid $INFLIGHT ==="
# Poll /proc rather than `kill -0`: signalling this driver proved unreliable on this box
# (kill returned 0, target state never changed), so we only ever *observe* it.
while [ -d "/proc/$INFLIGHT" ]; do sleep 20; done
say "in-flight eval finished; GPU free -> $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

mkdir -p "$SW/nfB_bctrl"
cd "$ROOT/isaac_sim"
# EVAL_GATE_BYPASS jumps the queue the paused sweep is waiting behind.
EVAL_GATE_BYPASS=1 bash launchers/run_eval_partition.sh lv2t_nfB_bctrl_p1 \
  > "$SW/nfB_bctrl/p1.log" 2>&1
say "bctrl shard exit=$?"
say "overlay: $(grep -h 'load_model.*overlay' "$SW/nfB_bctrl"/p1.log 2>/dev/null | sed 's/.*overlay from //' | sort -u | tr '\n' ' ')"

say "=== RESULT ==="
"$PY" - "$SW" <<'PYR' 2>&1 | tee -a "$LOG"
import json, os, pathlib, sys
sys.path.insert(0, os.environ["INFRAVLA_ROOT"])
from isaac_sim.scenes._common import WAREHOUSE_2X3_ZONES
SW = pathlib.Path(sys.argv[1])
ZB = WAREHOUSE_2X3_ZONES["B"]                      # canonical west-back aisle rectangle
(bx0, bx1), (by0, by1) = ZB.x_range, ZB.y_range
print(f"aisle-B rect: x[{bx0:.2f},{bx1:.2f}] y[{by0:.2f},{by1:.2f}]")

hm = {m["name"]: m for m in json.loads((pathlib.Path(os.environ["INFRAVLA_ROOT"])/
     "isaac_sim/configs/experiment/datagen_2x3_langv2_heldoutFKB_manifest.json").read_text())["scenes"]}

def rollouts(root):
    """name -> (final_dist, entered_B). entered_B: any logged pose inside the B rect."""
    fin = {}
    for sj in pathlib.Path(root).glob("**/summary.jsonl"):
        for ln in sj.read_text().splitlines():
            try:
                r = json.loads(ln); d = r.get("final_dist")
                fin[r["name"]] = d if isinstance(d, (int, float)) else 9.99
            except Exception: pass
    ent = {}
    for lj in pathlib.Path(root).glob("**/log.jsonl"):
        nm = lj.parent.name.split("__lang__")[0].split("__pose__")[0]
        hit, saw_pose = False, False
        for l in lj.read_text().splitlines():
            if not l.strip(): continue
            try: rec = json.loads(l)
            except Exception: continue
            if "x" not in rec or "y" not in rec: continue   # schema: flat x/y per step
            saw_pose = True
            if bx0 <= rec["x"] <= bx1 and by0 <= rec["y"] <= by1: hit = True; break
        # Never report a silent False: a schema change must fail loudly, not look like
        # "the robot never entered the aisle" (this exact bug bit us on 2026-08-10).
        if not saw_pose:
            raise SystemExit(f"ABORT: no x/y fields in {lj} — entered-B is unmeasurable")
        ent[nm] = hit
    return fin, ent

def row(label, fin, ent, keep):
    sel = [k for k in fin if keep(k)]
    n = len(sel)
    if not n: print(f"{label:34s}  (no data)"); return None
    r03 = sum(1 for k in sel if fin[k] <= 0.3)
    r10 = sum(1 for k in sel if fin[k] <= 1.0)
    e   = [ent[k] for k in sel if k in ent]
    eb  = sum(1 for v in e if v)
    print(f"{label:34s} {n:>4d} {r03/n:>10.3f} {r10/n:>9.3f} {f'{eb}/{len(e)}':>12s}")
    return dict(n=n, reach_0_3=r03/n, reach_1m=r10/n, entered_B=f"{eb}/{len(e)}")

fB, eB = rollouts(SW/"nfB_bctrl")
fH, eH = rollouts(SW/"nfB_hfkb")
ult = lambda k: hm.get(k, {}).get("goal_zone") == "B" and hm.get(k, {}).get("target_phrase") == "forklift"
print(f"\n{'split':34s} {'n':>4s} {'reach@0.3':>10s} {'reach@1m':>9s} {'entered B':>12s}")
print("-"*74)
out = {}
out["BC B-control (seen obj in B)"] = row("BC  seen object in B (target)", fB, eB, lambda k: True)
out["S3 ultimate OOD (forklift@B)"] = row("S3  forklift in B (target)",     fH, eH, ult)
out["S2 OOD scene / seen target"]   = row("S2  forklift in B, target elsewhere", fH, eH, lambda k: not ult(k))
(SW/"nfB_BCONTROL_summary.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {SW}/nfB_BCONTROL_summary.json")
print("read: BC high + S3 low -> the unseen PAIRING is the cause.")
print("      BC also low      -> the model is just bad at aisle B (lost 35 families).")
PYR

rm -f "$GATE"
say "=== gate dropped — generalization sweep resumes at p7 ==="
