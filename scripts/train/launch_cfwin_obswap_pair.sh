#!/usr/bin/env bash
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
PARENT="$ROOT/runs/training/runs/2026-07-11_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg50_cfwinboost8_15k_v0"
MEM_FREE_THRESH_MIB=20000

say() { echo "[obswap_pair $(date '+%F %T')] $*"; }

# nounset AFTER conda activate (activate.d hooks read unset vars).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-omnivla}"
set -uo pipefail
cd "$ROOT" || exit 1

say "gate: waiting for no trainer/eval AND GPU used < ${MEM_FREE_THRESH_MIB} MiB..."
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    busy=$(pgrep -fc "train_omnivla_sim|eval_policy" || true)
    if [ "${busy:-0}" -eq 0 ] && [ "$used" -lt "$MEM_FREE_THRESH_MIB" ]; then
        sleep 30
        used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        busy2=$(pgrep -fc "train_omnivla_sim|eval_policy" || true)
        [ "${busy2:-0}" -eq 0 ] && [ "$used2" -lt "$MEM_FREE_THRESH_MIB" ] && break
    fi
    sleep 60
done
say "gate open."

run_arm() {   # $1 = config stem (without .yaml)
    local stem="$1"
    local cfg="$ROOT/isaac_sim/train/configs/${stem}.yaml"
    local rundir="$ROOT/runs/training/runs/${stem}_v0"
    say "=== ARM ${stem} ==="
    mkdir -p "$rundir/checkpoints" "$rundir/logs"
    # cp -a, NOT symlink — checkpoint pruning must never reach the parent.
    if [ ! -d "$rundir/checkpoints/step_15000" ]; then
        say "staging parent step_15000..."
        cp -a "$PARENT/checkpoints/step_15000" "$rundir/checkpoints/step_15000" || return 1
    fi
    say "--- val-only baseline (pre-ft cf-val) ---"
    PYTHONUNBUFFERED=1 python vla-scripts/train_omnivla_sim_cctv_v2.py \
        --config "$cfg" --resume-step 15000 --val-only --disable-wandb \
        2>&1 | tee "$rundir/logs/valonly_baseline_15000.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { say "BASELINE VAL FAILED — skipping arm."; return 1; }
    say "--- fine-tune 15000 -> 19000 ---"
    PYTHONUNBUFFERED=1 python vla-scripts/train_omnivla_sim_cctv_v2.py \
        --config "$cfg" --resume-step 15000 \
        2>&1 | tee -a "$rundir/logs/train.log"
    local rc=${PIPESTATUS[0]}
    say "--- arm exited: $rc ---"
    return "$rc"
}

run_arm "2026-07-12_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg100_cfwin_ft19k" \
    || say "WARNING: control arm failed"
run_arm "2026-07-12_cctv4_sharedenc_2x3cfv4_egogoal_avgpool64_r16_fg100_cfwin_obswap_ft19k" \
    || say "WARNING: swap arm failed"
say "=== BOTH ARMS DONE ==="
