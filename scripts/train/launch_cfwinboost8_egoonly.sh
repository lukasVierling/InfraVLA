#!/usr/bin/env bash
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
CFG="$ROOT/isaac_sim/train/configs/2026-07-11_baseline_egoonly_2x3cfv4_r16_fg50_cfwinboost8_15k.yaml"
NFS="${CFV4_DATA:-$ROOT/runs/data_gen/warehouse_2x3_cfv4_2026-07-10}"   # spatial-task dataset (with CCTV frames)
DATA="$ROOT/runs/data/warehouse_2x3_cfv4_egoonly_2026-07-10"            # ego-only copy for the baseline

say() { echo "[launch_cfwb8_egoonly $(date '+%F %T')] $*"; }

# nounset AFTER conda activate (activate.d hooks read unset vars).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-omnivla}"
set -uo pipefail
cd "$ROOT" || exit 1

# ---- (1) ego-only copy of the dataset (CCTV frames dropped) ----
say "copying cfv4 dataset without cctv_* (~30G)"
mkdir -p "$DATA"
rsync -a --info=progress2 --exclude 'cctv_*' "$NFS/" "$DATA/" || { say "ABORT: rsync failed"; exit 1; }
N=$(find "$DATA" -name result.json | wc -l)
say "copy done: $N result.json ($(du -sh "$DATA" | cut -f1))"
[ "$N" -ge 540 ] || { say "ABORT: only $N result.json after copy"; exit 1; }

# ---- (2) split — same base ranges as the cfv4 split of the CCTV runs ----
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

# ---- (3) smoke test (3 steps, no wandb) ----
python vla-scripts/train_omnivla_sim_cctv_v2.py --config "$CFG" --smoke-test 3 --disable-wandb || {
    say "ABORT: smoke test failed"; exit 1; }
rm -rf "$ROOT/runs/training/runs/2026-07-11_baseline_egoonly_2x3cfv4_r16_fg50_cfwinboost8_15k_v0_smoke"

# ---- (4) train ----
say "=== training: ego-only baseline on cfv4, 15k steps ==="
LOG="$ROOT/runs/training/_train_cfwb8_egoonly_v0.log"
PYTHONUNBUFFERED=1 python vla-scripts/train_omnivla_sim_cctv_v2.py \
    --config "$CFG" 2>&1 | tee "$LOG"
say "=== training exited: ${PIPESTATUS[0]} ==="
