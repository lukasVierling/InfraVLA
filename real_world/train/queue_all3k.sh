#!/bin/bash
# Trains both real-robot arms of the paper to 3000 steps, then merges each for
# serving. Trainings run sequentially (one job fills the GPU); each merge runs on
# CPU in the background so it overlaps the next training.
set +u
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-omnivla}" || exit 1
set -u
cd "$ROOT" || exit 1

R="$ROOT/runs/training/real_world"
S="$ROOT/runs/serving/real_world"
mkdir -p "$R"
LOG=$R/_all3k.log
: > "$LOG"

say() { echo "[$(date -Is)] $*" >> "$LOG"; }

merge_bg() {   # run_dir_name  out_name  training_config
  ( python real_world/train/merge_lora_for_serving.py --v2 \
      --base "$ROOT/omnivla-original" \
      --ckpt "$R/$1/checkpoints/step_03000" --step 3000 --device cpu \
      --out "$S/$2" --config "$3" >> "$LOG" 2>&1
    say "MERGE $2 rc=$?" ) &
}

say "InfraVLA stage 1: train 0 -> 2000, then resume 2000 -> 3000"
python real_world/train/train_omnivla_real_cctv.py \
  --config real_world/train/configs/spot_cctv_cfboost.yaml >> "$LOG" 2>&1
python real_world/train/train_omnivla_real_cctv.py \
  --config real_world/train/configs/spot_cctv_cfboost_3k.yaml \
  --resume-step 2000 >> "$LOG" 2>&1
rc=$?; say "cfboost rc=$rc"
[ $rc -eq 0 ] && merge_bg 2026-08-21_spot_cctv_cfboost_v0 2026-08-21_spot_cctv_cfboost_v0_step3000_merged \
  real_world/train/configs/spot_cctv_cfboost_3k.yaml

say "baseline (no CCTV): train 0 -> 3000"
python real_world/train/train_omnivla_real_cctv.py \
  --config real_world/train/configs/spot_egoonly_langsim_full.yaml >> "$LOG" 2>&1
rc=$?; say "egoonly rc=$rc"
[ $rc -eq 0 ] && merge_bg 2026-08-21_spot_egoonly_langsim_full_v0 2026-08-21_spot_egoonly_langsim_full_v0_step3000_merged \
  real_world/train/configs/spot_egoonly_langsim_full.yaml

wait
say "ALL DONE"
