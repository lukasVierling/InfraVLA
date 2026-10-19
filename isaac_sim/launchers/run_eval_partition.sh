#!/usr/bin/env bash
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
EXP="${1:?usage: run_eval_partition.sh <experiment_name>}"

# CAP=1 queue gate — see the Notes block at the end of this file.
GATE="$ROOT/runs/evaluation/EVAL_QUEUE_PAUSED"
if [ -z "${EVAL_GATE_BYPASS:-}" ]; then
  while [ -f "$GATE" ]; do
    echo "[launch] $(date +%F\ %T)  queue gate held — waiting to start ${EXP}"
    sleep 30
  done
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-omnivla}"
source "${ISAACSIM_PATH:-/isaacsim}/setup_conda_env.sh"
cd "$ROOT/isaac_sim"
# Keep each eval self-contained: write the launch log into the run's OWN output dir
# (recorder.runs_root from the experiment cfg) under runs/, next to its results/images
# — never in the code tree. Fallback also stays under runs/.
RUNS_ROOT=$(sed -nE 's/^[[:space:]]*runs_root:[[:space:]]*"?([^"]+)"?[[:space:]]*$/\1/p' "configs/experiment/${EXP}.yaml" | head -1)
RUNS_ROOT="${RUNS_ROOT//\$\{oc.env:INFRAVLA_ROOT\}/$ROOT}"   # resolve the config placeholder
LOGDIR="${RUNS_ROOT:-$ROOT/runs/evaluation/_unresolved}/logs"
mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/${EXP}.log"
echo "[launch] $(date '+%F %T')  experiment=${EXP}  python=$(which python)  log=${LOG}"
python eval_policy.py +experiment="${EXP}" 2>&1 | tee "${LOG}"
echo "[launch] $(date '+%F %T')  DONE experiment=${EXP}  eval_exit=${PIPESTATUS[0]}"
