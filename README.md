# InfraVLA: Extending Vision-Language-Action Navigation with Infrastructure Cameras

![InfraVLA overview](assets/overview.png)

Warehouses, offices and hospitals often already have cameras installed. They see parts of the
building the robot cannot see from where it stands, but navigation policies do not use them.

**InfraVLA** adapts the pretrained navigation VLA [OmniVLA](https://omnivla-nav.github.io) to these
static infrastructure (CCTV) views. A CCTV encoder turns each view into input tokens, and the policy
is trained in two stages: first on demonstrations with upsampled counterfactual data, then on
recovery data mined from its own rollouts.

## Results

Success rates from closed-loop rollouts. The baseline is OmniVLA fine-tuned on the same data,
without CCTV input.

| Task | Baseline | InfraVLA |
|---|--:|--:|
| Semantic (sim): reach the object named in the instruction | 34.0 % | **100.0 %** |
| Spatial (sim): reroute around blocked aisles | 73.6 % | **100.0 %** |
| Semantic, held-out object placement | – | 88.2 % |
| Spatial, held-out blocking patterns | – | 88.9 % |
| Real Spot robot, < 10 min of demonstrations | 29.2 % | **83.3 %** |

## Method

![InfraVLA architecture](assets/architecture.png)

- **Architecture.** OmniVLA encodes the on-board image and the goal (language, image or 2D pose) and
  predicts 8 future waypoints with a LLaMA 2 7B backbone. InfraVLA adds a CCTV encoder whose 256
  tokens per view are average-pooled to 64. The model is fine-tuned with LoRA.
- **Stage 1: counterfactual upsampling.** Pairs of demonstrations start from the same spawn but differ
  in the instruction or the scene. The window where their paths split is upsampled to about half of
  the training samples, so the policy has to use the CCTV views to decide.
- **Stage 2: recovery data.** The stage-1 policy is rolled out; when it derails, an A\* supervisor
  steers it back, and these corrections are added to the training data (HG-DAgger style).

## Repository layout

The repository is a fork of [OmniVLA](https://github.com/NHirose/OmniVLA).

| Path | Contents |
|---|---|
| `prismatic/extern/hf/modeling_prismatic_v2.py` | The InfraVLA model (CCTV encoder, pooling, token insertion). |
| `prismatic/vla/datasets/` | Datasets and counterfactual upsampling. |
| `vla-scripts/train_omnivla_sim_cctv_v2.py` | Training script. |
| `isaac_sim/` | Simulation: data generation, evaluation, recovery mining, configs. |
| `real_world/` | Spot robot: fine-tuning and deployment. |
| `scripts/` | Launch scripts for the reported runs. |

Data, checkpoints and outputs are written to `runs/` (git-ignored).

## Installation

Tested on Linux with an NVIDIA RTX PRO 6000 (96 GB).

```bash
conda create -n omnivla python=3.10 -y
conda activate omnivla
pip install numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -e ".[infravla]"          # ".[infravla,real]" for the real robot
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# pretrained OmniVLA checkpoint
git clone https://huggingface.co/NHirose/omnivla-original

export INFRAVLA_ROOT=$(pwd)
```

- Keep the pinned [OpenVLA-OFT fork of `transformers`](https://github.com/moojink/transformers-openvla-oft.git);
  stock `transformers` silently gives wrong predictions.
- Simulation needs [Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/). Set
  `ISAACSIM_PATH` if it is not installed at `/isaacsim`.
- Set your W&B entity in the training config (`output.wandb_entity`) or pass `--disable-wandb`.

## Usage

Run all commands from the repository root.

**1. Generate demonstrations** in Isaac Sim:

```bash
python isaac_sim/run_builders/build_2x3_lang.py --config isaac_sim/configs/datagen_2x3_langv2.yaml --tag mydata --date $(date +%F)   # semantic task
python isaac_sim/run_builders/build_2x3.py --tag mydata --date $(date +%F)                                                       # spatial task
bash isaac_sim/launchers/run_datagen_partition.sh data_gen_2x3_<tag>_p1
```

**2. Split into train and validation:**

```bash
python isaac_sim/utils/split_lang_twins.py --datasets runs/data_gen/<dataset> \
    --prefix warehouse_2x3_langv2 --val-families 12 --seed 42 --out <split_name>              # semantic task
python isaac_sim/utils/split_dataset_cf.py --manifest isaac_sim/configs/experiment/datagen_2x3_cfv4_manifest.json \
    --runs-root runs/data_gen/<dataset> --train-bases 1500:1750 --val-bases 1750:1774 --test-bases 1774:1775 \
    --out <split_name>                                                                          # spatial task
```

**3. Train stage 1:**

```bash
python vla-scripts/train_omnivla_sim_cctv_v2.py --config isaac_sim/train/configs/2026-07-23_cctv4_avgpool64_langv2_yawcrux_pre3.yaml
```

Training configs for every experiment are in `isaac_sim/train/configs/`.

**4. Mine recovery data** with the stage-1 policy (one process per shard):

```bash
cd isaac_sim
python recovery/recovery.py policy=omnivla_langv2_pre3_best fine_grained.enabled=true \
    recovery.split_dir=$INFRAVLA_ROOT/runs/data/splits/2026-07-22_warehouse_2x3_langv2/train \
    recovery.output_root=$INFRAVLA_ROOT/runs/data/recovery/2026-08-11_A015k_langv2_recovery_t05_N48 \
    recovery.criterion=yaw recovery.yaw_eps_deg=1.0 recovery.window_pre_steps=3 recovery.window_post_steps=0 \
    recovery.rollout_dev_thresh_m=0.5 recovery.budget_factor=2.0 recovery.n_spawns=1 \
    recovery.quota=48 recovery.n_shards=3 recovery.shard=0 recovery.seed=0
```

**5. Train stage 2** by resuming the stage-1 checkpoint inside the new run directory:

```bash
PARENT=runs/training/runs/2026-07-23_cctv4_avgpool64_langv2_yawcrux_pre3_v0/checkpoints/step_15000
RUN=runs/training/runs/2026-08-11_langv2_A0_dmrecext2_N48_ft20k_v0
mkdir -p $RUN/checkpoints/step_15000 && cp -a $PARENT/. $RUN/checkpoints/step_15000/
python vla-scripts/train_omnivla_sim_cctv_v2.py --config isaac_sim/train/configs/2026-08-11_langv2_A0_dmrecext2_N48_ft20k.yaml --resume-step 15000
```

**6. Evaluate in closed loop.** Evaluation configs are in `isaac_sim/configs/experiment/`:

```bash
bash isaac_sim/launchers/run_eval_partition.sh lv2t_A0full_p1
```

Results are written to `runs/evaluation/`, one line per rollout in `summary.jsonl`.

For the real robot, see [real_world/train/README.md](real_world/train/README.md) and
[real_world/live/README.md](real_world/live/README.md).

## Citation

```bibtex
@misc{vierling2026infravla,
  title         = {{InfraVLA}: Extending Vision-Language-Action Navigation with Infrastructure Cameras},
  author        = {Vierling, Lukas and Ramtoula, Benjamin and Robinson, Luke and Clark, Ronald and De Martini, Daniele},
  year          = {2026},
  eprint        = {2609.33647},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.33647}
}
```

## Acknowledgements

This code builds on [OmniVLA](https://github.com/NHirose/OmniVLA) and
[OpenVLA-OFT](https://openvla-oft.github.io/). We thank the authors for releasing their code.

## License

MIT, see [LICENSE](LICENSE).
