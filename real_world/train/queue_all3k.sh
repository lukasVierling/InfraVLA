#!/bin/bash
# All three full-data arms to 3000 steps (~20 epochs on the 75-run split),
# then merge each for serving. Trainings are sequential (one job fills the GPU);
# each merge runs on CPU in the background so it overlaps the next training.
set +u
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-omnivla}" || exit 1
set -u
cd "$ROOT" || exit 1

R="$ROOT/runs/training/real_world"
S="$ROOT/runs/serving/real_world"
LOG=$R/_all3k_2026_08_24.log
: > "$LOG"

say() { echo "[$(date -Is)] $*" >> "$LOG"; }

merge_bg() {   # run_dir_name  out_name
  ( python real_world/train/merge_lora_for_serving.py \
      --base "$ROOT/omnivla-original" \
      --ckpt "$R/$1/checkpoints/step_03000" --step 3000 --device cpu \
      --out "$S/$2" >> "$LOG" 2>&1
    say "MERGE $2 rc=$?" ) &
}

say "STAGE 1/3: resume langsim_full_v0  2000 -> 3000"
python real_world/train/train_omnivla_real_cctv.py \
  --config real_world/train/configs/spot_cctv_langsim_full_3k.yaml \
  --resume-step 2000 >> "$LOG" 2>&1
rc=$?; say "langsim_full_v0 rc=$rc"
[ $rc -eq 0 ] && merge_bg 2026-08-21_spot_cctv_langsim_full_v0 2026-08-21_spot_cctv_langsim_full_v0_step3000_merged

say "STAGE 2/3: resume cfboost_v0  2000 -> 3000"
python real_world/train/train_omnivla_real_cctv.py \
  --config real_world/train/configs/spot_cctv_cfboost_3k.yaml \
  --resume-step 2000 >> "$LOG" 2>&1
rc=$?; say "cfboost_v0 rc=$rc"
[ $rc -eq 0 ] && merge_bg 2026-08-21_spot_cctv_cfboost_v0 2026-08-21_spot_cctv_cfboost_v0_step3000_merged

say "STAGE 3/3: train egoonly_langsim_full_v0  0 -> 3000"
python real_world/train/train_omnivla_real_cctv.py \
  --config real_world/train/configs/spot_egoonly_langsim_full.yaml >> "$LOG" 2>&1
rc=$?; say "egoonly_full_v0 rc=$rc"
[ $rc -eq 0 ] && merge_bg 2026-08-21_spot_egoonly_langsim_full_v0 2026-08-21_spot_egoonly_langsim_full_v0_step3000_merged

wait
say "ALL DONE"
