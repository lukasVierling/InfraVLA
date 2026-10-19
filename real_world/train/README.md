# `real_world/train/` — fine-tuning on Spot demonstrations

This folder trains the real-robot policies of the paper. The task mirrors the simulated semantic
task: Spot walks to one of three objects (a plant, a fire extinguisher, a traffic cone) that only two
fixed cameras see. The data are 75 teleoperated demonstrations, 25 object arrangements with one
demonstration per object, about 10 minutes in total.

Training reuses the simulation trainer unchanged. `train_omnivla_real_cctv.py` loads
`vla-scripts/train_omnivla_sim_cctv_v2.py` as a module and swaps only its dataset class for
`SpotRealCCTVDataset`.

## Configs

| Paper row | Config | Notes |
|---|---|---|
| Fine-tuned OmniVLA (no CCTV) | `spot_egoonly_langsim_full.yaml` | On-board view only, 3k steps. Same trainer with an empty camera list. |
| InfraVLA (stage 1) | `spot_cctv_cfboost.yaml`, then `spot_cctv_cfboost_3k.yaml` | Two CCTV views, counterfactual-snippet upsampling. The 3k config resumes the first at step 2000. |

The configs expect the recorded data under `runs/data/real_world/` and write runs to
`runs/training/real_world/`. The recorded data are not part of this repository.

## Run

```bash
export INFRAVLA_ROOT=$(pwd)
python real_world/train/train_omnivla_real_cctv.py --config real_world/train/configs/spot_cctv_cfboost.yaml
python real_world/train/train_omnivla_real_cctv.py --config real_world/train/configs/spot_cctv_cfboost_3k.yaml --resume-step 2000
```

Add `--smoke-test 3 --disable-wandb` for a quick check. `queue_all3k.sh` trains both arms (InfraVLA
and the ego-only baseline) to 3k steps and merges each one for serving.

## Merge before serving

`inference/run_omnivla.py` loads the backbone and the two heads but never applies a LoRA adapter.
Serving a raw training checkpoint therefore runs the base backbone with fine-tuned heads, without an
error. Merge first:

```bash
python real_world/train/merge_lora_for_serving.py --v2 \
    --base omnivla-original --ckpt runs/training/real_world/<run>/checkpoints/step_03000 \
    --step 3000 --out checkpoints/<run>_step3000_merged \
    --config real_world/train/configs/spot_cctv_cfboost_3k.yaml
```

With `--config`, the merge also writes `serving.json` with the waypoint spacing, camera list and
number of image slots the model was trained with. `real_world/live/serve_spot_ft.sh` reads it and refuses a launch that contradicts it.

## Sample format

`SpotRealDataset` and `SpotRealCCTVDataset` return the same fields as the simulation dataset.

- `actions` (8, 4): `[dx/s, dy/s, cos dψ, sin dψ]` in the body frame at 3 Hz, where `s` is
  `metric_waypoint_spacing`. The paper's configs use 0.19 m, the maximum recorded speed of
  0.578 m/s divided by 3 Hz.
- `goal_pose` (4,): a hindsight goal ahead along the trajectory.
- Language samples use the prompt format of OmniVLA: `"What action should the robot take to move
  toward the plant?"`.
- `obj_pose_norm` is all zeros, so keep `loss.obj_weight: 0.0`. A nonzero weight pulls the end of the
  predicted chunk toward the origin.
