#!/usr/bin/env bash
set -u
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
cd "$(dirname "$0")/.."   # launchers/ -> isaac_sim/ (where data_gen.py + configs/ live)
EXP="${1:?usage: run_datagen_partition.sh <experiment_name>}"
# Launch log goes into the dataset's OWN dir (recorder.runs_root) under runs/, next
# to the generated trajectories — not in the code tree. Fallback also stays under runs/.
RUNS_ROOT=$(sed -nE 's/^[[:space:]]*runs_root:[[:space:]]*"?([^"]+)"?[[:space:]]*$/\1/p' "configs/experiment/${EXP}.yaml" | head -1)
RUNS_ROOT="${RUNS_ROOT//\$\{oc.env:INFRAVLA_ROOT\}/$ROOT}"   # resolve the config placeholder
LOGDIR="${RUNS_ROOT:-$ROOT/runs/data_gen/_unresolved}/logs"
mkdir -p "${LOGDIR}"
export PYTHONEXE="/lib64/ld-linux-x86-64.so.2 ${ISAACSIM_PATH:-/isaacsim}/kit/python/bin/python3"
LOG="${LOGDIR}/${EXP}.log"
echo "[launch] $(date '+%F %T')  experiment=${EXP}  log=${LOG}"
${ISAACSIM_PATH:-/isaacsim}/python.sh data_gen.py +experiment="${EXP}" 2>&1 | tee "${LOG}"
# PIPESTATUS[0] is data_gen's exit; 1 is expected from a benign Kit shutdown crash.
echo "[launch] $(date '+%F %T')  DONE experiment=${EXP}  data_gen_exit=${PIPESTATUS[0]} (1 = benign shutdown crash)"
