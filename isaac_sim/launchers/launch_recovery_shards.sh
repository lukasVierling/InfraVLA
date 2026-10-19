#!/usr/bin/env bash
set -o pipefail
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
OUT="${OUT:?set OUT=<absolute output_root under runs/data/recovery/>}"
N_SHARDS="${N_SHARDS:-4}"
QUOTA="${QUOTA:-7}"
MAX_ROLLOUTS="${MAX_ROLLOUTS:-60}"
EXTRA="${EXTRA:-}"   # extra hydra overrides (e.g. replay/measure knobs)
POLICY="${POLICY:-omnivla_finetuned_cctv4_2x3cfv4_avgpool64_cfwinboost8_step15000}"
mkdir -p "$OUT"
for s in $(seq 0 $((N_SHARDS - 1))); do
  tmux kill-session -t "recovery_s${s}" 2>/dev/null
  tmux new-session -d -s "recovery_s${s}" "bash -c '\
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${CONDA_ENV:-omnivla}" && \
source "${ISAACSIM_PATH:-/isaacsim}/setup_conda_env.sh" && cd "$ROOT/isaac_sim"/recovery && \
python recovery.py policy=${POLICY} fine_grained.enabled=true \
  recovery.output_root=${OUT} recovery.shard=${s} recovery.n_shards=${N_SHARDS} \
  recovery.quota=${QUOTA} recovery.max_rollouts=${MAX_ROLLOUTS} ${EXTRA} \
  2>&1 | tee ${OUT}/shard${s}.log'"
  echo "[recovery-shards $(date '+%T')] launched shard ${s}/${N_SHARDS} (tmux recovery_s${s})"
  # Stagger: keeps the 4-shard checkpoint-load VRAM spikes from stacking.
  [ "$s" -lt $((N_SHARDS - 1)) ] && sleep 30
done
echo "[recovery-shards] all launched. Logs: ${OUT}/shard*.log"
