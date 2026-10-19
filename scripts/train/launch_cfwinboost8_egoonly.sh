#!/usr/bin/env bash
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
CFG="$ROOT/isaac_sim/train/configs/2026-07-11_baseline_egoonly_2x3cfv4_r16_fg50_cfwinboost8_15k.yaml"
NFS="${CFV4_DATA:-$ROOT/runs/data_gen/warehouse_2x3_cfv4_2026-07-10}"   # spatial-task dataset (with CCTV frames)
DATA="$ROOT/runs/data/warehouse_2x3_cfv4_egoonly_2026-07-10"            # ego-only copy for the baseline
MEM_FREE_THRESH_MIB=20000

say() { echo "[launch_cfwb8_egoonly $(date '+%F %T')] $*"; }

# nounset AFTER conda activate (activate.d hooks read unset vars).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-omnivla}"
set -uo pipefail
cd "$ROOT" || exit 1

# ---- (1) ego-only local copy (no GPU needed — runs while the eval is going) ----
say "copying cfv4 NFS -> localdata WITHOUT cctv_* (~30G expected)"
mkdir -p "$DATA"
rsync -a --info=progress2 --exclude 'cctv_*' "$NFS/" "$DATA/" || { say "ABORT: rsync failed"; exit 1; }
N=$(find "$DATA" -name result.json | wc -l)
say "copy done: $N result.json ($(du -sh "$DATA" | cut -f1))"
[ "$N" -ge 540 ] || { say "ABORT: only $N result.json after copy"; exit 1; }

# ---- (2) split — base ranges MUST match kaitain's cfv4 split exactly ----
say "splitting (train 1500:1750, val 1750:1774, test 1774:1775)"
python isaac_sim/utils/split_dataset_cf.py \
  --out 2026-07-11_warehouse_2x3_cfv4_egoonly \
  --manifest isaac_sim/configs/experiment/datagen_2x3_cfv4_manifest.json \
  --runs-root "$DATA" \
  --train-bases 1500:1750 --val-bases 1750:1774 --test-bases 1774:1775 \
  --allow-missing 25 | tee "$ROOT/runs/training/_cfv4_egoonly_split.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "ABORT: splitter failed"; exit 1; }
# cfv4 manifest has no 'vacuous' key -> splitter must drop NOTHING from train.
grep -q "vacuous dropped from train: 0" "$ROOT/runs/training/_cfv4_egoonly_split.log" || {
  say "ABORT: splitter dropped scenes as vacuous — cfv4 invariant broken"; exit 1; }

# ---- (3) GPU gate: wait out the gridES eval ----
say "gate: waiting for no eval/trainer procs AND GPU used < ${MEM_FREE_THRESH_MIB} MiB..."
i=0
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    n_busy=$(pgrep -fc "eval_policy.py|train_omnivla_sim" || true)
    if [ "${n_busy:-0}" -eq 0 ] && [ "$used" -lt "$MEM_FREE_THRESH_MIB" ]; then
        sleep 30
        used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        n_busy2=$(pgrep -fc "eval_policy.py|train_omnivla_sim" || true)
        if [ "${n_busy2:-0}" -eq 0 ] && [ "$used2" -lt "$MEM_FREE_THRESH_MIB" ]; then
            say "gate open (used=${used2} MiB) — launching."
            break
        fi
    fi
    i=$((i + 1))
    [ $((i % 10)) -eq 0 ] && say "still waiting (used=${used} MiB, busy procs=${n_busy})"
    sleep 60
done

# ---- (4) smoke test (3 steps, no wandb) — catches loader bugs + batch-10 OOM.
# Batch 10x3 is UNTESTED for the ego-only arm; CFG_BS7 (7x4=28, the old
# lineage batch) is the automatic fallback if batch 10 doesn't fit.
CFG_BS7="$ROOT/isaac_sim/train/configs/2026-07-11_baseline_egoonly_2x3cfv4_r16_fg50_cfwinboost8_15k_bs7.yaml"
smoke() { python vla-scripts/train_omnivla_sim_cctv_v2.py --config "$1" --smoke-test 3 --disable-wandb; }

TRY="$CFG"
if smoke "$CFG"; then
    say "smoke OK at batch 10"
else
    say "batch-10 smoke FAILED — falling back to batch 7 (old lineage)"
    TRY="$CFG_BS7"
    smoke "$CFG_BS7" || { say "ABORT: batch-7 smoke ALSO failed — real bug, not batch size"; exit 1; }
fi
rm -rf "$ROOT/runs/training/runs/2026-07-11_baseline_egoonly_2x3cfv4_r16_fg50_cfwinboost8_15k_v0_smoke" \
       "$ROOT/runs/training/runs/2026-07-11_baseline_egoonly_2x3cfv4_r16_fg50_cfwinboost8_15k_bs7_v0_smoke"

# ---- (5) train; if the batch-10 run dies MID-RUN with CUDA OOM (fragmentation,
# val-pass spike — things a 3-step smoke can't see), restart FRESH at batch 7.
# Fresh restart, not resume: a different batch shape mid-run would make the
# curve unusable. The dead batch-10 run dir is left in place for post-mortem
# (different run name -> no collision); delete it manually after.
# Fresh run: do NOT pre-create the run dir (trainer refuses); log to
# runs/training/_<name>.log per house convention.
say "=== training: EGO-ONLY baseline + cfwinboost8 on cfv4, 15k from scratch ($(basename $TRY)) ==="
LOG="$ROOT/runs/training/_train_cfwb8_egoonly_v0.log"
PYTHONUNBUFFERED=1 python vla-scripts/train_omnivla_sim_cctv_v2.py \
    --config "$TRY" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ] && [ "$TRY" = "$CFG" ] && grep -q "CUDA out of memory" "$LOG"; then
    say "batch-10 training died with CUDA OOM (exit $rc) — auto-restarting fresh at batch 7"
    LOG="$ROOT/runs/training/_train_cfwb8_egoonly_bs7_v0.log"
    PYTHONUNBUFFERED=1 python vla-scripts/train_omnivla_sim_cctv_v2.py \
        --config "$CFG_BS7" 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
fi
say "=== training exited: $rc ==="
