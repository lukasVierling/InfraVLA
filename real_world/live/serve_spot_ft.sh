#!/usr/bin/env bash
set -euo pipefail
ROOT="${INFRAVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; export INFRAVLA_ROOT="$ROOT"
ENV_NAME="${ENV_NAME:-omnivla}"
if [ "$(basename "${CONDA_PREFIX:-}")" != "$ENV_NAME" ]; then
  set +u +e
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then . "$c/etc/profile.d/conda.sh"; break; fi
  done
  conda activate "$ENV_NAME"
  rc=$?
  set -u -e
  [ $rc -eq 0 ] || { echo "could not activate conda env '$ENV_NAME'" >&2; exit 1; }
fi
command -v python >/dev/null || { echo "no python after activating $ENV_NAME" >&2; exit 1; }

CKPT="${CKPT:-$ROOT/checkpoints/spot_current}"
# Left unset on purpose: both are read from the checkpoint's serving.json when it
# has one, and only fall back to these defaults when it does not. Setting either
# explicitly is allowed but is checked against the contract and refused on a
# mismatch -- see the CONTRACT block below.
SPACING_SET="${SPACING+yes}"
CCTV_SET="${CCTV+yes}"
SPACING="${SPACING:-}"
CCTV="${CCTV:-}"
BIND="${BIND:-tcp://127.0.0.1:5556}"
NEED_MIB="${NEED_MIB:-18000}"

[ -e "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }
REAL=$(readlink -f "$CKPT")

# derive the step from the files present; refuse if ambiguous or absent
mapfile -t HEADS < <(ls "$REAL"/pose_projector--*_checkpoint.pt 2>/dev/null || true)
[ "${#HEADS[@]}" -eq 1 ] || { echo "expected exactly one pose_projector--*.pt in $REAL, found ${#HEADS[@]}" >&2; exit 1; }
STEP=$(basename "${HEADS[0]}" | sed -E 's/pose_projector--([0-9]+)_checkpoint\.pt/\1/')
[ -f "$REAL/action_head--${STEP}_checkpoint.pt" ] || { echo "action_head--${STEP}_checkpoint.pt missing" >&2; exit 1; }

# a raw training checkpoint has lora_adapter/ and no safetensors -> reject it
if [ -d "$REAL/lora_adapter" ] && ! ls "$REAL"/model-*.safetensors >/dev/null 2>&1; then
  echo "ERROR: $REAL looks like a RAW training checkpoint (lora_adapter/, no merged weights)." >&2
  echo "       define_model ignores LoRA adapters. Merge it first:" >&2
  echo "       python real_world/train/merge_lora_for_serving.py --base $ROOT/omnivla-original \\" >&2
  echo "              --ckpt $REAL --step $STEP --out ${REAL}_merged" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# CONTRACT CHECK. num_images_in_input / cctv_cam_names / spacing live only in the
# training config, and wrong values fail silently (e.g. serving a CCTV checkpoint
# without --cctv yields plausible-looking chunks from a misaligned action slice).
# merge_lora_for_serving writes serving.json beside the merged weights; this
# refuses any launch that contradicts it.
# ---------------------------------------------------------------------------
SIDE="$REAL/serving.json"
if [ -f "$SIDE" ]; then
  C_SPACING=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['spacing_m'])" "$SIDE")
  C_CAMS=$(python3 -c "import json,sys;print(','.join(json.load(open(sys.argv[1]))['cctv_cam_names']))" "$SIDE")
  C_IMGS=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['num_images_in_input'])" "$SIDE")
  C_NAME=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['run_name'])" "$SIDE")
  if [ "$SPACING_SET" = yes ] && [ "$SPACING" != "$C_SPACING" ]; then
    echo "ERROR: SPACING=$SPACING contradicts the checkpoint contract ($C_SPACING)." >&2
    echo "       $SIDE says this model was trained at $C_SPACING m/unit." >&2; exit 1
  fi
  if [ "$CCTV_SET" = yes ] && [ "$CCTV" != "$C_CAMS" ]; then
    echo "ERROR: CCTV='$CCTV' contradicts the checkpoint contract ('$C_CAMS')." >&2
    echo "       Serving a $C_IMGS-image model with the wrong camera list mis-slices" >&2
    echo "       the action hidden state and fails without an error." >&2; exit 1
  fi
  SPACING="$C_SPACING"; CCTV="$C_CAMS"
  echo "contract   : $C_NAME  ($C_IMGS image slots, cams='${C_CAMS:-none}', spacing $C_SPACING)"
else
  [ -n "$SPACING" ] || { echo "ERROR: no serving.json in $REAL and SPACING unset." >&2
    echo "       Re-merge with --config <training yaml>, or pass SPACING=<metric_waypoint_spacing>." >&2; exit 1; }
  echo "WARNING: no serving.json in $REAL -- no contract to check." >&2
  echo "         spacing=$SPACING cctv='${CCTV:-none}' are UNVERIFIED operator input." >&2
fi

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE" -lt "$NEED_MIB" ]; then
  echo "ERROR: only ${FREE} MiB free on the GPU, need ~${NEED_MIB}." >&2
  echo "       Currently holding it:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 1
fi

echo "checkpoint : $REAL"
echo "step       : $STEP"
echo "spacing    : $SPACING m   (policy_server default is 0.305 — overridden)"
echo "bind       : $BIND"
echo "gpu free   : ${FREE} MiB"
echo

exec python "$ROOT/real_world/live/policy_server.py" \
    --vla-path "$REAL" --step "$STEP" --spacing "$SPACING" --bind "$BIND" \
    ${CCTV:+--cctv "$CCTV"}
