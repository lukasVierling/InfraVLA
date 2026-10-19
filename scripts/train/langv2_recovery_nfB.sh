#!/usr/bin/env bash
# Recovery data generation for the LANGUAGE task, mined from the noforkliftB model.
#
# Question: nfB wedges on the rack corner entering aisle B (8/10 stuck on seen pairings,
# 0/30 entry on the unseen forklift@B pairing). Does oracle recovery supervision fix the
# in-distribution wedging, and does any of it transfer to the OOD pairing?
#
# Recipe = the spatial b20 campaign (scripts/_archive/train/recovery_b20_chain.sh): n_spawns=1,
# budget_factor=2.0, fine_grained follower on. Two knobs MUST differ from spatial because
# they are a training-side contract, not a preference -- the cf window definition has to
# match the arm's cf_window_balance (nfB config lines 103-106): criterion=yaw (not
# distance) and window 3/0 (not 16/8).
#
# Mining source is nfB's TRAIN split only. Verified 2026-08-10 to have zero scene- and
# zero family-level overlap with: nfB val, the held-out forklift@B set (the S2/S3 test),
# and the parent test/val splits. Recovery data becomes TRAINING data, so this must hold.
#
# Three-way sharding: the live recovery worker measured 23.6 GiB VRAM, so three
# workers plus the persistent Isaac service use about 75 GiB on the 95.6 GiB
# card. Four workers previously OOMed and remain forbidden.
set -uo pipefail
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
GATE=$ROOT/runs/evaluation/EVAL_QUEUE_PAUSED
OUT=$ROOT/runs/data/recovery/2026-08-10_nfB13k_langv2_recovery_t05_N50
SPLIT=$ROOT/runs/data/splits/langv2_no_forkliftB/train
LOG=$ROOT/runs/data/recovery/2026-08-10_nfB13k_langv2_recovery_t05_N50.log
N_SHARDS=3
QUOTA_PER_SHARD=17
MAX_ROLLOUTS_PER_SHARD=67
STAGGER_SECONDS=60
say(){ mkdir -p "$(dirname "$LOG")"; echo "[lv2rec $(date '+%F %T')] $*" | tee -a "$LOG"; }

run_worker(){
  shard=$1
  quota=$2
  cd "$ROOT/isaac_sim"
  # Conda and Isaac's activation hooks legitimately probe unset variables.
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-omnivla}"
  source ${ISAACSIM_PATH:-/isaacsim}/setup_conda_env.sh
  set -u
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec python recovery.py \
    policy=omnivla_langv2_noforkliftB_best \
    fine_grained.enabled=true \
    recovery.split_dir="$SPLIT" \
    recovery.output_root="$OUT" \
    recovery.criterion=yaw \
    recovery.yaw_eps_deg=1.0 \
    recovery.window_pre_steps=3 \
    recovery.window_post_steps=0 \
    recovery.rollout_dev_thresh_m=0.5 \
    recovery.budget_factor=2.0 \
    recovery.n_spawns=1 \
    recovery.quota="$quota" \
    recovery.max_rollouts="$MAX_ROLLOUTS_PER_SHARD" \
    recovery.n_shards="$N_SHARDS" \
    recovery.shard="$shard" \
    recovery.seed=0
}

if [ "${1:-}" = "--worker" ]; then
  run_worker "$2" "$3"
  exit $?
fi

lower_gate(){
  if [ -e "$GATE" ]; then
    gate_history=$ROOT/runs/evaluation/_gate_history
    mkdir -p "$gate_history"
    mv "$GATE" "$gate_history/EVAL_QUEUE_PAUSED_$(date '+%Y%m%d_%H%M%S')_$$"
  fi
}

active_worker_count(){
  active=0
  for shard in $(seq 0 $((N_SHARDS - 1))); do
    tmux has-session -t "lv2recov_s${shard}" 2>/dev/null && active=$((active + 1))
  done
  echo "$active"
}

exit_cleanup(){
  active=$(active_worker_count)
  if [ "$active" -eq 0 ]; then
    lower_gate
  else
    say "supervisor exiting with $active workers alive; keeping evaluation gate raised"
  fi
}
trap exit_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUT"
say "=== LANGV2 RECOVERY MINING START (3 shards x 17 = 51, dev 0.5, n_spawns 1, budget 2.0) ==="

# Hold the eval queue so the standard-52 shards do not grab the card.
touch "$GATE"; say "queue gate raised - eval waits until all recovery shards exit"

say "waiting for the GPU (no eval_policy running, <15 GB used)"
while :; do
  busy=$(pgrep -fc "eval_policy.py|train_omnivla_sim_cctv_v2.py" || true)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${busy:-1}" -eq 0 ] && [ "${used:-99999}" -lt 15000 ] && break
  sleep 60
done
say "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)) - launching workers"

for shard in $(seq 0 $((N_SHARDS - 1))); do
  session="lv2recov_s${shard}"
  if tmux has-session -t "$session" 2>/dev/null; then
    say "ERROR: tmux $session already exists"
    exit 2
  fi
  tmux new-session -d -s "$session" -c "$ROOT" \
    "bash scripts/train/langv2_recovery_nfB.sh --worker $shard $QUOTA_PER_SHARD 2>&1 | tee -a $OUT/shard${shard}.log"
  say "launched shard $shard/$N_SHARDS in tmux $session (quota $QUOTA_PER_SHARD)"
  if [ "$shard" -lt $((N_SHARDS - 1)) ]; then
    sleep "$STAGGER_SECONDS"
  fi
done

while [ "$(active_worker_count)" -gt 0 ]; do
  sleep 60
done

missing_manifests=0
for shard in $(seq 0 $((N_SHARDS - 1))); do
  [ -f "$OUT/recovery_manifest_shard${shard}.json" ] || missing_manifests=$((missing_manifests + 1))
done
collected=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'rec_*' | wc -l)
say "workers exited: collected=$collected, missing_manifests=$missing_manifests"

lower_gate
trap - EXIT
say "queue gate dropped - eval sweep resumes"
say "=== DONE ==="
[ "$missing_manifests" -eq 0 ]
