
from __future__ import annotations

import argparse
import json
import os
import random as py_random
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Type

# Make repo root importable so `experiments.robot.openvla_utils`,
# `isaac_sim.*`, and `prismatic.*` all resolve when invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import copy
import torch.nn as nn
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, WeightedRandomSampler

# Suppress some noisy threads — paper trainer also does this.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic_v2 import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.vision_encoder import build_cctv_downsampler, resolve_cctv_downsample, resolve_image_slot_perm
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_CCTV
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK, POSE_DIM
from prismatic.vla.datasets.cctv_isaac_sim_dataset import IsaacSimDataset
from prismatic.vla.datasets.dataset_utils import (
    apply_cf_window_filter, cf_window_sample_mask,
)


# funcitn to load the yaml into a conifg
def _expand_env(node):
    """Expand ${VAR} in every string of a loaded config (paths use ${INFRAVLA_ROOT})."""
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    return os.path.expandvars(node) if isinstance(node, str) else node


def load_yaml_config(path: Path) -> dict:
    """Load YAML config and return as plain dict, with ${INFRAVLA_ROOT} expanded."""
    os.environ.setdefault("INFRAVLA_ROOT", str(Path(__file__).resolve().parents[1]))
    with open(path, "r") as f:
        return _expand_env(yaml.safe_load(f))

# track which git version we used for running the code
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"

#
def register_omnivla_classes() -> None:
    """Register OmniVLA's custom HF classes for from_pretrained() to find them.

    Mirrors paper's train_omnivla.py:800-803. Idempotent — calling twice
    raises ValueError, which we catch.
    """
    #tells the auto factories the mapping from "openvla" to the openvla config, the from config the processor and image processor
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
        #find out what exactly are those things, image processor, prismatic processor and  vision2seq
    except ValueError:
        pass  # already registered

#function to load the processor and the vla model
def load_processor_and_vla(
    vla_path: Path,
    num_images_in_input: int,
    use_cctv_encoder: bool = False,
    num_ego_images: int = 2,
    separate_cctv_projector: bool = False,
    device: str = "cuda",
):
    """Load PrismaticProcessor + OpenVLAForActionPrediction_MMNv1.

    Mirrors paper's train_omnivla.py:806-847.

    use_cctv_encoder: add a parallel CCTV vision stream (frozen copy of the
        ego encoders) that gets its OWN LoRA weights — wrap_lora targets all
        Linear layers, so the copy is picked up automatically. Must happen
        before wrap_lora / get_peft_model.
    """
    #call above defined function so the following auto class knows the configs and what to call
    register_omnivla_classes()
    # v2: load the v2 model class DIRECTLY (not via AutoModel) so we always get the
    # modeling_prismatic_v2 code path regardless of the checkpoint's auto_map, and we
    # deliberately SKIP update_auto_map() so the shared base checkpoint's config.json
    # is never mutated to point at v2.
    #processor to process the input (images and tokenization)
    processor = AutoProcessor.from_pretrained(str(vla_path), trust_remote_code=True)
    #vla backbone (v2 class), bf16 + low_cpu_mem_usage to stream shards to gpu
    vla = OpenVLAForActionPrediction_MMNv1.from_pretrained(
        str(vla_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    #TODO find out what the usual input here is and what the meaning behidn that is
    vla.vision_backbone.set_num_images_in_input(num_images_in_input)
    # parallel CCTV stream (frozen copy of ego encoders, own LoRA via wrap_lora)
    if use_cctv_encoder:
        vla.vision_backbone.add_cctv_encoder(num_ego_images=num_ego_images)
        # Optional DISTINCT vision->LLM projector for CCTV blocks: warm-started as a
        # copy of the (pretrained) ego projector, created BEFORE wrap_lora so it gets
        # its OWN LoRA — exactly like the cctv encoder above. Wired into the encoder
        # seam at install time (install_pixel_shuffle_encoder). At init it's identical
        # to the shared projector, so it only diverges via its separate adapter.
        if separate_cctv_projector:
            vla.cctv_projector = copy.deepcopy(vla.projector)
    #move the vla to the gpu
    vla.to(dtype=torch.bfloat16, device=device)
    return processor, vla

#function to wrap lora 
def wrap_lora(vla, lora_rank: int, lora_alpha: int, lora_dropout: float,
              freeze_vision: bool = False, freeze_llm: bool = False):
    """LoRA-wrap every Linear layer in the model.

    Mirrors paper's train_omnivla.py:850-866. Target modules = ALL Linear
    layers (paper's choice — broadly trainable but kept small by rank=32).
    `lora_alpha` is now config-sourced (was hardcoded `min(lora_rank, 16)`).

    freeze_vision (model.freeze_vision, default False): attach NO adapters to
    the EGO vision encoder or the EGO projector, so they stay at their released
    OmniVLA state (LoRA already freezes base weights — no adapter == frozen).
    The CCTV stream is deliberately NOT covered: `cctv_featurizer` /
    `cctv_fused_featurizer` / `cctv_projector` still get adapters and learn.

    freeze_llm (model.freeze_llm, default False): additionally attach NO
    adapters to the Llama backbone (`language_model.`, incl. its lm_head) —
    the "freeze level 2" arm: the only LoRA'd modules left are the CCTV
    featurizers + cctv_projector, so the trainable set is exactly {CCTV
    stream (LoRA), action_head, pose_projector} = new-sensor in, action-space
    adaptation out, pretrained navigation trunk untouched. The frozen LLM can
    still READ the CCTV tokens — the trainable cctv_projector learns to place
    their content in directions the frozen attention already responds to.
    """
    # Trailing dots are LOAD-BEARING: they keep the CCTV copies out of this set
    # ("vision_backbone.cctv_featurizer." does not start with
    # "vision_backbone.featurizer.", and "cctv_projector." not with "projector.").
    EGO_VISION = (
        "vision_backbone.featurizer.",
        "vision_backbone.fused_featurizer.",
        "projector.",
    )
    LLM = ("language_model.",)
    #get the modules we want to inject lora into ie all linear layers in the model -> i assume not in the processor -> seems to only to atual image processing and then tokenziation not any embeddings so probably no learning component
    target_modules = [
        name for name, module in vla.named_modules()
        if isinstance(module, torch.nn.Linear)
        and not (freeze_vision and name.startswith(EGO_VISION))
        and not (freeze_llm and name.startswith(LLM))
    ]
    if freeze_vision or freeze_llm:
        n_linear = sum(1 for _n, m in vla.named_modules()
                       if isinstance(m, torch.nn.Linear))
        frozen_bits = [b for b, on in (("EGO encoder+projector", freeze_vision),
                                       ("LLM backbone", freeze_llm)) if on]
        print(f"  freeze: {' + '.join(frozen_bits)} frozen "
              f"({n_linear - len(target_modules)} Linears excluded from LoRA); "
              f"{len(target_modules)} Linears still adapted")
        if not target_modules:
            raise ValueError("freeze flags excluded EVERY Linear from LoRA — "
                             "nothing left to adapt; check the config")
    #lora config: rank/alpha/dropout all from YAML, target all linear layers, gaussian init (paper's choice)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    #get the peft model (huggingface function)
    vla = get_peft_model(vla, lora_config)
    #get the trainable parameters all fine
    vla.print_trainable_parameters()
    return vla

#function to load a checkpoint I asuume if training failed etc.
def load_module_checkpoint(
    module_name: str,
    path: Path,
    step: int,
    device: str = "cpu",
) -> dict:
    """Load a {pose_projector,action_head} checkpoint.

    Filename convention: `{module}--{step}_checkpoint.pt`.

    Paper has a quirk: `pose_projector` files are actually saved as
    `proprio_projector` historically (train_omnivla.py:188-189). Mirror
    that fallback so paper's released checkpoints work.
    """
    primary = path / f"{module_name}--{step}_checkpoint.pt"
    if not primary.exists() and module_name == "pose_projector":
        primary = path / f"proprio_projector--{step}_checkpoint.pt"
    print(f"  Loading checkpoint: {primary}")
    #just basic torch load
    sd = torch.load(primary, weights_only=True, map_location=device)
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

#init function
def init_module(
    cls: Type[nn.Module],
    name: str,
    init_kwargs: dict,
    pretrained_dir: Path | None,
    pretrained_step: int | None,
    device: str,
    to_bf16: bool = False,
) -> nn.Module:
    """Instantiate a module, optionally loading pretrained weights.

    `pretrained_dir + pretrained_step` together specify the source. Set both
    to None to skip loading (random init).
    """
    #TODO where do init kwargs come form? Config? we are loading the model here why call it m? kind of vague 
    m = cls(**init_kwargs)
    #get number of parames that require grad and print
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  {name}: {n_params:,} trainable params")

    if pretrained_dir is not None and pretrained_step is not None:
        #for pretrained we load from a checkpoint
        sd = load_module_checkpoint(name, Path(pretrained_dir), pretrained_step, device="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing:
            print(f"    WARN: {len(missing)} missing keys: {missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"    WARN: {len(unexpected)} unexpected keys: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    if to_bf16:
        m = m.to(torch.bfloat16)
    m = m.to(device)
    return m

def pad_and_mask_batch(batch: dict, pad_token_id: int, model_max_length: int) -> dict:
    """Pad input_ids/labels and compute attention masks.

    Paper's `merge_dict_batches` does this inline; we extract it into a
    standalone step since we don't have multiple datasets to merge.
    """
    # padding the input ids looks fine TODO will we pad based on what modlaity is chosen?
    input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(batch["labels"], batch_first=True, padding_value=IGNORE_INDEX)
    #cut off at max length
    batch["input_ids"] = input_ids[:, :model_max_length]
    batch["labels"] = labels[:, :model_max_length]
    #ne will return a boolean mas where True for the non overlaps (so in this case where there is no pad_token)
    batch["attention_mask"] = batch["input_ids"].ne(pad_token_id)
    # do the same as above but we are using ignore index isntead of pad token id, where was ignore idx defined? probably at data generation time 
    batch["attention_mask_label"] = batch["labels"].ne(IGNORE_INDEX)
    # modality_id stored as list of ints by collator → tensor for indexing.
    batch["modality_id"] = torch.as_tensor(batch["modality_id"], dtype=torch.long)
    #setting the goal modality so I assume the earlier mask didn't mask out input based on modality 
    batch["goal_mask_select"] = batch["modality_id"]
    return batch

#important forward function
def forward_and_loss(
    batch: dict,
    vla,
    pose_projector,
    action_head,
    num_patches: int,
    device: str,
    train_mode: bool = True,
    *,
    smoothness_weight: float,   # YAML loss.smoothness_weight (paper: 0.1)
    obj_weight: float,          # YAML loss.obj_weight (paper: 0.1; inert until language)
    abs_pose_projector=None,    # optional absolute-pose encoder (extra token; see main())
) -> tuple[torch.Tensor, dict]:
    """One forward pass; returns (loss, metrics_dict).

    Loss composition (mirrors paper at train_omnivla.py:404, MINUS MBRA):
        L = 1.0 * MSE(action_chunk, predicted)
          + 0.1 * MSE(obj_pose, predicted_final[:2])   ← only on lang samples
          + 0.1 * MSE(predicted[:, :-1], predicted[:, 1:])   ← smoothness

    The obj term is GUARDED — for batches with no lang samples (always the
    case in iter 1), MSE on empty tensor would NaN. We return 0.0 instead.
    """
    #get the shape of the batch but only first entry so we get the batch size 
    bsize = batch["pixel_values"].shape[0]
    #ge the modality id 
    modality_id = batch["modality_id"].to(device)
    #ok not calculating grads if we are not in train mode TODO why did we activate the train mode for inference in the experiment script? maybe it is just very low quality also because of random clipping that ignored the dimensions
    forward_ctx = torch.no_grad() if not train_mode else _nullcontext()
    with forward_ctx:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                attention_mask_label=batch["attention_mask_label"].to(device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                modality_id=modality_id.to(torch.bfloat16),
                block_present=batch["block_present"].to(device),   # CCTV joint-training mask -> _build_multimodal_attention_cctv
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device),
                proprio_projector=pose_projector,
                abs_proprio=(batch["abs_pose"].to(torch.bfloat16).to(device)
                             if abs_pose_projector is not None else None),
                abs_proprio_projector=abs_pose_projector,
                use_film=False,
            )
    #get the labels but why are we only starting from 1: for the second dimensions?
    #TODO why starting at idx 1 but then later on predicting the first token as well? 
    ground_truth_token_ids = batch["labels"][:, 1:].to(device)
    #get the current action mask where is this function defined? in the original omnivla script so can rely on this but better understand the different masks we have in our code later TODO
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    # next acion mask, find out how this works
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    # get the last hidden state
    last_hidden_states = output.hidden_states[-1]                       # (B, seq, D)
    # get the last hidden state but for text only modality
    text_hidden_states = last_hidden_states[:, num_patches:-1]          # exclude vision patches
    # TODO what is going on here?
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(bsize, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    # use the action head to predict next action train mode will do with or without grad 
    if train_mode:
        predicted_actions = action_head.predict_action(
            actions_hidden_states, modality_id.to(torch.bfloat16),
        )
    else:
        with torch.no_grad():
            predicted_actions = action_head.predict_action(
                actions_hidden_states, modality_id.to(torch.bfloat16),
            )
    #get the reference actions ie the labels 
    action_ref = batch["actions"].to(device).to(torch.bfloat16)
    #TODO what is the obj pose norm
    obj_pose_norm = batch["obj_pose_norm"].to(device).to(torch.bfloat16)
    #clac the loss between action ref and predicted actions -> why not L1? like in the paper?
    loss_main = nn.MSELoss()(action_ref, predicted_actions)
    #calc the loss we use 0.1 multiplier for the MSELoss, why are we using MSE not L1? Is this the smootheness loss and not the main loss we are using?
    # iI see the loss is over the actions itself for smootheness
    loss_smooth = smoothness_weight * nn.MSELoss()(predicted_actions[:, :-1], predicted_actions[:, 1:])
    # if we use a modality with language 
    lan_bool = (modality_id == 7) | (modality_id == 8)
    #TODO why are we not predicting yaw, does langauge not predict this?
    if lan_bool.any():
        #we calc the loss again but only for the language samples and only for xy dim why TODO?
        loss_obj = obj_weight * nn.MSELoss()(
            obj_pose_norm[lan_bool],
            predicted_actions[:, -1, 0:2][lan_bool],
        )
    else:
        # no language samples → obj loss on empty tensor would NaN → use 0.0 instead.
        loss_obj = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
    #add all theolosses together
    loss = loss_main + loss_smooth + loss_obj
    #get the metrics we want to evaluate on
    metrics = {
        "loss": loss.detach().float().item(),
        "loss_main": loss_main.detach().float().item(),
        "loss_smooth": loss_smooth.detach().float().item(),
        "loss_obj": loss_obj.detach().float().item(),
    }  
    #now we get the sample wise L1 loss? why TODO
    with torch.no_grad():
        per_sample_l1 = (
            (predicted_actions.float() - action_ref.float()).abs().mean(dim=(1, 2))
        )                                                          # (B,)
        for mod in (4, 5, 6, 7, 8):
            mask = (modality_id == mod)
            if mask.any():
                metrics[f"l1_modality_{mod}"] = per_sample_l1[mask].mean().item()
        per_axis = (predicted_actions.float() - action_ref.float()).abs().mean(dim=(0, 1))   # (4,)
        for i, name in enumerate(("dx", "dy", "cos_yaw", "sin_yaw")):
            metrics[f"l1_axis_{name}"] = per_axis[i].item()

        # --- per perception-dropout condition L1 (block_present cols: 0=ego_obs, 2=cctv_obs) ---
        bp = batch["block_present"].to(per_sample_l1.device)
        ego_p = bp[:, 0] > 0.5
        cctv_p = bp[:, 2] > 0.5
        for cname, cmask in (("both", ego_p & cctv_p), ("egoonly", ego_p & ~cctv_p), ("cctvonly", ~ego_p & cctv_p)):
            if cmask.any():
                metrics[f"l1_cond_{cname}"] = per_sample_l1[cmask].mean().item()
                metrics[f"frac_cond_{cname}"] = cmask.float().mean().item()

    return loss, metrics
#run validation script 
def run_validation(
    val_loader: DataLoader,
    vla,
    pose_projector,
    action_head,
    num_patches: int,
    device: str,
    pad_token_id: int,
    model_max_length: int,
    *,
    smoothness_weight: float,
    obj_weight: float,
    max_samples: int | None = None,   # YAML val.max_samples; None = full pass
    abs_pose_projector=None,
) -> dict:
    """Val pass over up to `max_samples` samples (None = full set). Returns mean metrics."""
    vla.eval()
    action_head.eval()
    pose_projector.eval()

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n_batches = 0
    n_samples = 0
    for batch in val_loader:
        #TODO what does this function do?
        batch = pad_and_mask_batch(batch, pad_token_id, model_max_length)
        _, m = forward_and_loss(
            batch, vla, pose_projector, action_head,
            num_patches=num_patches, device=device, train_mode=False,
            smoothness_weight=smoothness_weight, obj_weight=obj_weight,
            abs_pose_projector=abs_pose_projector,
        )
        #TODO what happens here?
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
        n_batches += 1
        # Cap val cost once we've seen `max_samples` (None = no cap).
        n_samples += len(batch["modality_id"])
        if max_samples is not None and n_samples >= max_samples:
            break
    means = {k: sums[k] / counts[k] for k in sums}
    means["n_batches"] = n_batches
    #activat etrain mode for vla and the action head and pose projector
    vla.train()
    action_head.train()
    #TODO what is a pose projector
    pose_projector.train()
    return means

def verify_dataset_sanity(dataset, n_check: int = 100) -> None:
    """Quick checks on the dataset.

    Asserts that surprise me to find FALSE would mean a normalisation /
    indexing / modality-dropout bug — better to catch at startup than to
    discover from a degraded val loss in hour 3.
    """
    print("Verifying dataset sanity...")
    rng = py_random.Random(0)
    modality_counts: dict[int, int] = {}
    max_abs_xy = 0.0
    max_abs_sincos = 0.0
    for _ in range(n_check):
        idx = rng.randint(0, len(dataset) - 1)
        item = dataset[idx]
        actions = item["actions"]
        #make sure the shape is correct
        assert actions.shape == (NUM_ACTIONS_CHUNK, ACTION_DIM), \
            f"actions shape {actions.shape} != expected ({NUM_ACTIONS_CHUNK}, {ACTION_DIM})"
        #get the absolute values for xy and for sincos (yaw)
        max_abs_xy = max(max_abs_xy, float(np.abs(actions[:, :2]).max()))
        max_abs_sincos = max(max_abs_sincos, float(np.abs(actions[:, 2:]).max()))
        modality_counts[int(item["modality_id"])] = modality_counts.get(int(item["modality_id"]), 0) + 1
    #TODO where does the 32. come from
    assert max_abs_xy <= 32.0, f"action xy magnitude {max_abs_xy} suggests normalization bug"
    assert max_abs_sincos <= 1.01, f"|cos,sin| {max_abs_sincos} > 1 — math error"
    # Modality coverage. Originally hardcoded "4 AND 5 present" (written for the
    # goal-image runs). Generalise to the dataset's CONFIGURED modalities: every
    # configured modality EXCEPT 6 (image-only, intentionally dropped for long
    # goals) must appear over n_check. For no_goal_image runs modality_list=[4],
    # so only pose (4) is required; for goal-image runs configured={4,5,6} this
    # still requires 4 AND 5 (original behaviour). Also catch any rogue modality
    # outside the configured set (would signal an indexing / dropout bug).
    configured = set(int(m) for m in getattr(dataset, "modality_list", (4, 5)))
    observed = set(modality_counts)
    assert observed <= configured, \
        f"rogue modality {observed - configured} outside configured {configured}: {modality_counts}"
    required = configured - {6}
    assert required <= observed, \
        f"configured modalities {required - observed} missing from sample of {n_check}: {modality_counts}"
    print(f"  ✓ action shapes consistent; max |xy|={max_abs_xy:.2f}, max |cos,sin|={max_abs_sincos:.4f}")
    print(f"  ✓ modality distribution over {n_check}: {modality_counts}")


def verify_vram_headroom(
    vla, pose_projector, action_head, val_loader,
    num_patches: int, device: str, pad_token_id: int, model_max_length: int,
    *, smoothness_weight: float, obj_weight: float, abs_pose_projector=None,
) -> None:
    """Run one forward+backward, print peak VRAM."""
    #function to verify that there is enough vram for training
    print("Verifying VRAM headroom (1 forward+backward)...")
    torch.cuda.reset_peak_memory_stats(device)
    vla.train()
    pose_projector.train()
    action_head.train()
    batch = pad_and_mask_batch(next(iter(val_loader)), pad_token_id, model_max_length)
    loss, _ = forward_and_loss(
        batch, vla, pose_projector, action_head,
        num_patches=num_patches, device=device, train_mode=True,
        smoothness_weight=smoothness_weight, obj_weight=obj_weight,
        abs_pose_projector=abs_pose_projector,
    )
    loss.backward()
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"  ✓ peak VRAM: {peak_gb:.1f} GB / {total_gb:.1f} GB total ({100*peak_gb/total_gb:.0f}%)")
    if peak_gb > 0.9 * total_gb:
        print("  ⚠ WARNING: VRAM > 90% — consider reducing batch_size in the YAML")
    for p in vla.parameters():
        if p.grad is not None:
            p.grad = None
    for p in pose_projector.parameters():
        if p.grad is not None:
            p.grad = None
    for p in action_head.parameters():
        if p.grad is not None:
            p.grad = None


def save_checkpoint(
    run_dir: Path,
    step: int,
    vla,
    pose_projector,
    action_head,
    cctv_pixel_shuffle,
    optimizer,
    scheduler,
    rng_state: dict,
    val_metrics: dict | None,
    cctv_cam_embed=None,
    abs_pose_projector=None,
) -> Path:
    """Save full state for step.

    Layout:
        run_dir/processor/                # written ONCE at startup, not per-checkpoint
        run_dir/checkpoints/step_NNNNN/
            lora_adapter/                 # PEFT format
            pose_projector--{step}_checkpoint.pt
            action_head--{step}_checkpoint.pt
            training_state.pt             # optimizer + scheduler + RNG + step
            val_metrics.json
    """
    #Function to save a checkpoint 
    ckpt_dir = run_dir / "checkpoints" / f"step_{step:05d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    #save the lora adapter but apparently save the whole val?
    (ckpt_dir / "lora_adapter").mkdir(exist_ok=True)
    vla.save_pretrained(ckpt_dir / "lora_adapter")
    #save the pose projector? TODO still have to find out what that is 
    torch.save(
        pose_projector.state_dict(),
        ckpt_dir / f"pose_projector--{step}_checkpoint.pt",
    )
    #also save the action head -> pose projector action head and vla saved indepndently
    torch.save(
        action_head.state_dict(),
        ckpt_dir / f"action_head--{step}_checkpoint.pt",
    )
    # v2: the standalone CCTV downsampler module (peer of pose/action; NOT in the
    # LoRA adapter) — save it so a resume/eval doesn't silently lose its weights.
    # Param-free downsamplers (avgpool) have nothing to save, so skip them.
    if cctv_pixel_shuffle is not None and any(p.numel() for p in cctv_pixel_shuffle.parameters()):
        torch.save(
            cctv_pixel_shuffle.state_dict(),
            ckpt_dir / f"cctv_pixel_shuffle--{step}_checkpoint.pt",
        )
    # v2: per-camera identity embedding (raw Parameter peer) — save so resume/eval keeps it.
    if cctv_cam_embed is not None:
        torch.save(cctv_cam_embed.detach().cpu(), ckpt_dir / f"cctv_cam_embed--{step}_checkpoint.pt")
    # v2: absolute-pose encoder (peer module, fresh-trained) — save so resume/eval keep it.
    if abs_pose_projector is not None:
        torch.save(abs_pose_projector.state_dict(),
                   ckpt_dir / f"abs_pose_projector--{step}_checkpoint.pt")
    #also save optimizer state and scheduler to continue crahsed training where it left
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "rng": rng_state,
        },
        ckpt_dir / "training_state.pt",
    )

    if val_metrics is not None:
        (ckpt_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    print(f"  ✓ saved checkpoint @ step {step} → {ckpt_dir}")
    return ckpt_dir


def prune_old_checkpoints(
    run_dir: Path, keep_top_k: int, keep_latest: bool, keep_every_n_steps: int = 0,
) -> None:
    """Keep top-K by val L1 + the most recent + milestones. Delete others.

    Reads each step_*/val_metrics.json's `loss_main` (the main MSE).
    `keep_every_n_steps` > 0 also protects every checkpoint whose step is a
    multiple of it (e.g. 5000 → keep 5k, 10k, 15k, ...), so we can eval
    earlier models to study overfitting even if their val L1 wasn't top-K.
    """
    #keep topk val checkpoints and the most recent one to save memory
    ckpts_root = run_dir / "checkpoints"
    if not ckpts_root.exists():
        return
    candidates = []
    #get performance of each checkpoint
    for d in ckpts_root.iterdir():
        if not d.is_dir():
            continue
        m_path = d / "val_metrics.json"
        if not m_path.exists():
            continue
        m = json.loads(m_path.read_text())
        candidates.append((float(m.get("loss_main", float("inf"))), int(d.name.split("_")[-1]), d))
    if len(candidates) <= keep_top_k + (1 if keep_latest else 0):
        return
    #sort cadidates by loss 
    by_loss = sorted(candidates, key=lambda x: x[0])
    #get the ones to keep
    keep = set(d for _, _, d in by_loss[:keep_top_k])
    #add the latest checkpoint to the keep set
    if keep_latest:
        latest = max(candidates, key=lambda x: x[1])[2]
        keep.add(latest)
    #protect milestone checkpoints (every keep_every_n_steps) for overfitting analysis
    if keep_every_n_steps:
        for _, step, d in candidates:
            if step % keep_every_n_steps == 0:
                keep.add(d)
    #prune the ones we don't want
    for _, step, d in candidates:
        if d not in keep:
            print(f"  pruning checkpoint step {step} ({d})")
            shutil.rmtree(d)


def load_eval_modules(
    ckpt_dir: Path,
    vla,
    pose_projector,
    action_head,
    cctv_pixel_shuffle=None,
    cctv_cam_embed=None,
    abs_pose_projector=None,
) -> None:
    """Load a checkpoint's MODEL weights only (LoRA adapter + peer modules) — no
    optimizer/scheduler/RNG. Shared by load_full_state (resume) and the --val-only
    checkpoint sweep, which needs weights only and so can run on eval-file-only
    checkpoints that lack training_state.pt.
    """
    adapter_dir = ckpt_dir / "lora_adapter"
    #load the adapter
    if adapter_dir.is_dir():
        vla.load_adapter(str(adapter_dir), adapter_name="default", is_trainable=True)
        print(f"  ✓ reloaded LoRA adapter from {adapter_dir}")
    else:
        raise FileNotFoundError(
            f"missing LoRA adapter dir at {adapter_dir} — resume from this "
            f"checkpoint would silently use pretrained LoRA weights"
        )
    #load the action head and pose projector
    pp_files = list(ckpt_dir.glob("pose_projector--*_checkpoint.pt"))
    ah_files = list(ckpt_dir.glob("action_head--*_checkpoint.pt"))
    if not pp_files or not ah_files:
        raise FileNotFoundError(f"missing pose_projector or action_head checkpoint in {ckpt_dir}")
    #load state dict for both
    pose_projector.load_state_dict(torch.load(pp_files[0], weights_only=True, map_location="cpu"), strict=False)
    action_head.load_state_dict(torch.load(ah_files[0], weights_only=True, map_location="cpu"), strict=False)
    # v2: restore the CCTV downsampler module if this run used a PARAMETERIZED one
    # (pixelshuffle). Param-free downsamplers (avgpool) reconstruct from config — nothing to load.
    if cctv_pixel_shuffle is not None and any(p.numel() for p in cctv_pixel_shuffle.parameters()):
        ps_files = list(ckpt_dir.glob("cctv_pixel_shuffle--*_checkpoint.pt"))
        if not ps_files:
            raise FileNotFoundError(
                f"parameterized CCTV downsampler run but no cctv_pixel_shuffle checkpoint in {ckpt_dir}"
            )
        cctv_pixel_shuffle.load_state_dict(
            torch.load(ps_files[0], weights_only=True, map_location="cpu"), strict=False
        )
        print(f"  ✓ reloaded CCTV downsampler from {ps_files[0].name}")
    # v2: restore the per-camera identity embedding Parameter if this run used it.
    if cctv_cam_embed is not None:
        ce_files = list(ckpt_dir.glob("cctv_cam_embed--*_checkpoint.pt"))
        if not ce_files:
            raise FileNotFoundError(f"cctv_cam_embed run but no cctv_cam_embed checkpoint in {ckpt_dir}")
        with torch.no_grad():
            cctv_cam_embed.copy_(
                torch.load(ce_files[0], weights_only=True, map_location="cpu").to(cctv_cam_embed.device))
        print(f"  ✓ reloaded per-camera embedding from {ce_files[0].name}")
    # v2: restore the absolute-pose encoder if this run used it.
    if abs_pose_projector is not None:
        ap_files = list(ckpt_dir.glob("abs_pose_projector--*_checkpoint.pt"))
        if not ap_files:
            raise FileNotFoundError(f"abs_pose run but no abs_pose_projector checkpoint in {ckpt_dir}")
        abs_pose_projector.load_state_dict(
            torch.load(ap_files[0], weights_only=True, map_location="cpu"))
        print(f"  ✓ reloaded abs-pose encoder from {ap_files[0].name}")


def load_full_state(
    ckpt_dir: Path,
    vla,
    pose_projector,
    action_head,
    optimizer,
    scheduler,
    cctv_pixel_shuffle=None,
    cctv_cam_embed=None,
    abs_pose_projector=None,
) -> int:
    """Restore full training state. Returns the step number.

    Reloads:
      - LoRA adapter + pose_projector + action_head + CCTV peers (load_eval_modules).
      - Optimizer + scheduler state (so AdamW momentum isn't reset).
      - RNG state (Python/numpy/torch/cuda) so dataloader shuffle continues.
    """
    #funtion to load the checkpoint of a model with opitmizer etc.
    load_eval_modules(
        ckpt_dir, vla, pose_projector, action_head,
        cctv_pixel_shuffle=cctv_pixel_shuffle, cctv_cam_embed=cctv_cam_embed,
        abs_pose_projector=abs_pose_projector,
    )
    #load the training state (optimizer, scheudler  etc.)
    state = torch.load(ckpt_dir / "training_state.pt", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    #also get the random seed state
    rng = state["rng"]
    py_random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state(rng["cuda"])
    return int(state["step"])


def get_rng_state() -> dict:
    #function to get the state for the random number generators of libraries
    return {
        "python": py_random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }

class _DummyWandb:
    """Stub that quacks like wandb so trainer code doesn't need conditionals."""
    class _Run:
        id = "disabled"
    def init(self, **kwargs):
        return self._Run()
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


def main() -> None:
    #main entry point
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="path to YAML config")
    ap.add_argument("--smoke-test", type=int, default=0, help="run N steps + 1 val + exit")
    ap.add_argument("--resume-step", type=int, default=None,
                    help="resume from runs/training/runs/{run_name}/checkpoints/step_NNNNN/")
    ap.add_argument("--disable-wandb", action="store_true",
                    help="skip wandb init; useful for smoke tests before wandb is set up")
    ap.add_argument("--val-only", action="store_true",
                    help="load the --resume-step checkpoint (weights only), run ONE val pass on "
                         "the (deterministic) val set, print 'VALSWEEP step=.. loss_main=..', exit. "
                         "For fair checkpoint comparison on a FROZEN val set.")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)

    #TODO wandb should be implemented no need for that anymore
    if cfg["logging"]["wandb_entity"] == "TODO" and not args.disable_wandb:
        raise SystemExit(
            "wandb_entity is still 'TODO' in the YAML. Set it before launching, "
            "or pass --disable-wandb."
        )
    #set random seed stuff
    seed = int(cfg["train"].get("seed", 42))
    py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #chck whether cuda is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #get the folder to store everyhting
    runs_root = Path(cfg["output"]["runs_root"])
    run_name = cfg["run"]["name"]
    if args.smoke_test:
        run_name = run_name + "_smoke"
    run_dir = runs_root / run_name
    if args.resume_step is None and run_dir.exists():
        raise SystemExit(
            f"Run dir already exists: {run_dir}\n"
            f"Bump run.name (e.g. _v0 → _v1) or pass --resume-step to continue.\n"
            f"For smoke tests, just `rm -rf {run_dir}` to retry."
        )
    #make dirs for saving things 
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)

    shutil.copy(args.config, run_dir / "config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #looks good can be reused for test runs 
    if args.disable_wandb:
        wandb = _DummyWandb()
        wandb_run = wandb.init()
    else:
        import wandb
        #login for wandb, should add stuff to config do we still need login then?
        wandb_run = wandb.init(
            entity=cfg["logging"]["wandb_entity"],
            project=cfg["logging"]["wandb_project"],
            name=run_name,
            tags=cfg["logging"].get("wandb_tags", []),
            config=cfg,
            notes=cfg["run"].get("notes", ""),
            resume="allow" if args.resume_step is not None else "never",
        )
    #dump the data of the run
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "git_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wandb_run_id": wandb_run.id,
        "seed": seed,
        "data_train": cfg["data"]["train_dir"],
        "data_val":   cfg["data"]["val_dir"],
        "data_test":  cfg["data"]["test_dir"],
        "resume_step": args.resume_step,
        "smoke_test": args.smoke_test,
    }, indent=2))
    #load the model and the processor
    print(f"\n=== Loading model ===")
    print(f"  vla_path: {cfg['model']['vla_path']}")
    processor, vla = load_processor_and_vla(
        Path(cfg["model"]["vla_path"]),
        num_images_in_input=cfg["model"]["num_images_in_input"],
        use_cctv_encoder=cfg["model"].get("use_cctv_encoder", False),
        num_ego_images=cfg["model"].get("num_ego_images", 2),
        separate_cctv_projector=cfg["model"].get("separate_cctv_projector", False),
        device=device,
    )
    #print whta vla we are using this hsould return OpenVLA hopefully?
    print(f"  vla class: {type(vla).__name__}")
    
    if args.resume_step is None:
        processor.save_pretrained(run_dir / "processor")
    #get the llm dim
    llm_dim = vla.llm_dim
    vision_backbone = vla.vision_backbone
    # get the num pathces which is pateches times image =one for the goal pose token 
    num_patches = (
        vision_backbone.get_num_patches() * vision_backbone.get_num_images_in_input()
        + 1   # +1 for the goal_pose token projected by ProprioProjector
    )
    #if lora then wrap in lora
    if cfg["model"]["use_lora"]:
        vla = wrap_lora(
            vla,
            lora_rank=cfg["model"]["lora_rank"],
            lora_alpha=cfg["model"]["lora_alpha"],
            lora_dropout=cfg["model"]["lora_dropout"],
            freeze_vision=cfg["model"].get("freeze_vision", False),
            freeze_llm=cfg["model"].get("freeze_llm", False),
        )
    #get the step of pretraining form where we finetune
    pretrained_step = cfg["model"].get("pretrained_step")
    #initialize the pose projector
    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        {"llm_dim": llm_dim, "proprio_dim": POSE_DIM},
        pretrained_dir=cfg["model"]["vla_path"],
        pretrained_step=pretrained_step,
        device=device,
    )
    #init the action head 
    action_head = init_module(
        L1RegressionActionHead_idcat,
        "action_head",
        {"input_dim": llm_dim, "hidden_dim": llm_dim, "action_dim": ACTION_DIM},
        pretrained_dir=cfg["model"]["vla_path"],
        pretrained_step=pretrained_step,
        device=device,
        to_bf16=True,
    )
    # v2: optional CCTV token-downsampler (pixelshuffle=learned, avgpool=param-free).
    # Standalone PEER module (like pose_projector/action_head) -> NOT LoRA-wrapped.
    # Installed into the v2 vision-encoder seam so the forward picks it up automatically.
    # `cctv_pixel_shuffle` is the install slot (kept named for ckpt/file compat); it now
    # holds whatever build_cctv_downsampler returns.
    ds_method, ds_out = resolve_cctv_downsample(cfg["model"].get)
    downsample_ego = bool(cfg["model"].get("downsample_ego", False))   # pool ego views too -> uniform out_tokens
    cctv_pixel_shuffle = None
    cctv_cam_embed = None   # optional per-camera identity embedding (Parameter, peer-saved)
    if ds_method != "none":
        grid_side = int(round(vision_backbone.get_num_patches() ** 0.5))  # sqrt(256) = 16
        cctv_pixel_shuffle = build_cctv_downsampler(
            ds_method, ds_out, vision_dim=vision_backbone.embed_dim, grid_side=grid_side,
        ).to(dtype=torch.bfloat16, device=device)
        # Optional CCTV extras (default off): per-camera identity embedding (raw
        # Parameter, zero-init -> no-op at start; full-trained peer like the downsampler)
        # and a distinct CCTV projector (set on vla in load_processor_and_vla BEFORE
        # wrap_lora -> own LoRA). Both wired into the encoder seam here.
        n_cctv_i = cfg["model"]["num_images_in_input"] - cfg["model"]["num_ego_images"]
        if cfg["model"].get("cctv_cam_embed", False):
            cctv_cam_embed = nn.Parameter(
                torch.zeros(n_cctv_i, llm_dim, dtype=torch.bfloat16, device=device))
        cctv_projector_ref = getattr(vla, "cctv_projector", None)  # set iff separate_cctv_projector
        vla.install_pixel_shuffle_encoder(
            cctv_pixel_shuffle, cfg["model"]["num_ego_images"], downsample_ego,
            cctv_projector=cctv_projector_ref, cctv_cam_embed=cctv_cam_embed,
        )
        if cctv_projector_ref is not None:
            print("  cctv SEPARATE projector: enabled (own LoRA, warm-started from ego projector)")
        if cctv_cam_embed is not None:
            print(f"  cctv per-camera embedding: {cctv_cam_embed.numel():,} params "
                  f"({n_cctv_i} x {llm_dim}, zero-init, full-trained peer)")
        n_ps = sum(p.numel() for p in cctv_pixel_shuffle.parameters())
        print(f"  cctv downsampler '{ds_method}': {n_ps:,} trainable params (NOT LoRA); "
              f"CCTV 256->{cctv_pixel_shuffle.out_tokens} tokens/view")
        # Vision span shrinks (ego stays 256, cctv -> out_tokens), so the offset that
        # forward_and_loss / verify_vram_headroom use to skip the BOS+vision tokens MUST
        # be recomputed — otherwise the action hidden-state slice runs off the (now
        # shorter) sequence end and yields an empty tensor. Mirrors the non-shuffle
        # formula get_num_patches()*n_img + 1, but with the reduced per-view counts.
        n_ego_ps = cfg["model"]["num_ego_images"]
        n_cctv_ps = cfg["model"]["num_images_in_input"] - n_ego_ps
        ego_tokens = cctv_pixel_shuffle.out_tokens if downsample_ego else vision_backbone.get_num_patches()
        num_patches = (
            ego_tokens * n_ego_ps                                 # ego views: out_tokens if downsample_ego else 256
            + cctv_pixel_shuffle.out_tokens * n_cctv_ps           # cctv views: out_tokens each
            + 1                                                   # +1 = BOS offset
        )
        print(f"  num_patches recomputed for downsample: {num_patches} "
              f"(non-downsampled would be {vision_backbone.get_num_patches() * cfg['model']['num_images_in_input'] + 1})")

    # --- Optional absolute-pose encoder (default off) ---
    # A SECOND ProprioProjector (fresh weights, same fc1->GELU->fc2 structure) that
    # maps [x, y, cos(yaw), sin(yaw), goal_x, goal_y] (world frame, xy normalised by
    # data.abs_pose_bounds) to ONE extra token appended after the pose token.
    # fc2 is ZERO-INITIALISED so the token is exactly 0 at step 0 (near-no-op start;
    # fc1 still gets gradients through fc2 as it trains). Full-trained peer module
    # (like pose_projector but with NO pretrained checkpoint) — saved/loaded as
    # abs_pose_projector--{step}_checkpoint.pt.
    abs_pose_projector = None
    if cfg["model"].get("abs_pose_encoder", False):
        abs_pose_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=6)
        nn.init.zeros_(abs_pose_projector.fc2.weight)
        nn.init.zeros_(abs_pose_projector.fc2.bias)
        abs_pose_projector = abs_pose_projector.to(dtype=torch.bfloat16, device=device)
        num_patches += 1   # the extra abs-pose token shifts the action hidden-state slice
        n_abs = sum(p.numel() for p in abs_pose_projector.parameters())
        print(f"  abs-pose encoder: {n_abs:,} params (fresh, fc2 zero-init), num_patches -> {num_patches}")

    # --- Optional config-driven image-slot order (e.g. ego-last: CCTV first so the
    # onboard view sits closest to the action readout). Resolved here, where
    # num_ego_images + cctv_cam_names are known; the SAME resolver runs at eval
    # (policy.py) so deploy == train. None/absent => identity => legacy order. ---
    slot_perm = resolve_image_slot_perm(
        cfg["model"].get("image_slot_order", None),
        cfg["model"]["num_ego_images"],
        tuple(cfg["data"].get("cctv_cam_names", ("cctv",))),
    )
    vla.set_image_block_order(slot_perm)
    if slot_perm is not None:
        print(f"  image_slot_order: {cfg['model']['image_slot_order']} -> block perm {slot_perm}")

    #build the dataset
    print(f"\n=== Building datasets ===")
    #first get the action tokenizer probably defined via the vla we are using
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    #get the dataset args that all datsets share
    common_dataset_kwargs = dict(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        context_size=cfg["data"]["context_size"],
        num_actions_chunk=cfg["data"]["num_actions_chunk"],
        modality_list=tuple(cfg["data"]["modality_list"]),
        modality_weights=(tuple(cfg["data"]["modality_weights"])
                          if cfg["data"].get("modality_weights") else None),
        lang_manifest=cfg["data"].get("lang_manifest"),
        lang_pattern=cfg["data"].get("lang_pattern"),
        first_step_only=cfg["data"].get("first_step_only", False),
        keep_steps_max=cfg["data"].get("keep_steps_max", None),
        image_only_max_distance_steps=cfg["data"]["image_only_max_distance_steps"],
        # Goal-sampling horizon + stop-token flag — previously dataset-default
        # hardcodes (1, 20, True); now YAML-sourced.
        min_goal_distance_steps=cfg["data"]["min_goal_distance_steps"],
        max_goal_distance_steps=cfg["data"]["max_goal_distance_steps"],
        predict_stop_token=cfg["data"]["predict_stop_token"],
        metric_waypoint_spacing=cfg["data"]["metric_waypoint_spacing"],
        waypoint_spacing=cfg["data"]["waypoint_spacing"],
        image_size=tuple(cfg["data"]["image_size"]),
        random_hflip=cfg["data"]["random_hflip"],
        goals_per_obs=cfg["data"]["goals_per_obs"],
        emit_cctv=cfg["data"]["emit_cctv"],
        # CCTV joint-training: feed a CCTV goal image (4 image slots) iff true.
        use_cctv_goal=cfg["data"]["use_cctv_goal"],
        # 4-CCTV mode: names of the fixed array cams to stack through the shared
        # encoder; no_goal_image drops the goal image entirely (goal = pose only).
        # .get() keeps the legacy single-"cctv" path working for old configs.
        cctv_cam_names=tuple(cfg["data"].get("cctv_cam_names", ("cctv",))),
        no_goal_image=cfg["data"].get("no_goal_image", False),
        # Per-camera CCTV dropout (each cctv view independently masked; ego always
        # present). Train-time only — val forces 0.0 below for a stable curve.
        cctv_cam_dropout=cfg["data"].get("cctv_cam_dropout", 0.0),
        # P(goal pinned to trajectory END) — de-leaked long-horizon supervision
        # for the cf campaign. Applies to train AND val (it is the task
        # distribution, not an augmentation — val must measure what train
        # optimizes). .get(0.0) keeps every pre-cf config byte-identical.
        final_goal_prob=cfg["data"].get("final_goal_prob", 0.0),
        # Counterfactual blockade-signal ablation: replace real CCTV frames with
        # solid black (aisle sealed) / white (aisle clear) per-aisle images.
        # Shared by train + val (it defines the input distribution, not an
        # augmentation). .get(False) keeps every existing config unchanged.
        cctv_blockade_signal=cfg["data"].get("cctv_blockade_signal", False),
        # Per-camera responsibility crop of the CCTV frames (default off).
        # Shared by train + val — it defines the input distribution, not an
        # augmentation. .get(None) keeps every existing config unchanged.
        cctv_crop=cfg["data"].get("cctv_crop"),
        cctv_crop_ref_size=cfg["data"].get("cctv_crop_ref_size", 720),
        # Absolute-pose token (default off): dataset emits "abs_pose" iff the
        # model-side encoder is enabled; bounds = warehouse extents from YAML.
        # Shared by train + val (input distribution, not an augmentation).
        abs_pose=cfg["model"].get("abs_pose_encoder", False),
        abs_pose_bounds=cfg["data"].get("abs_pose_bounds"),
        seed=seed,
    )
    # Counterfactual decision-window mode (data.counterfactual_windows, opt-in):
    # cut the sample index down (apply_cf_window_filter, dataset_utils) to the
    # frames around each cf pair's divergence point — the moments where the GT
    # action depends on the CCTV-visible seal, not on pose+goal. Applies to
    # train AND val (it defines the task distribution, not an augmentation —
    # the val loss becomes a counterfactual val loss). All three knobs are
    # required keys: window semantics belong in the YAML, not in code defaults.
    # The filter MUST run right after each construction, before the swap
    # builders / sampler-weight code below read samples_index (same ordering
    # the retired CounterfactualDataset subclass guaranteed via __init__).
    cfw = cfg["data"].get("counterfactual_windows") or {}
    if cfw.get("enabled", False):
        print(f"[counterfactual_windows] ENABLED: pre={cfw['pre_steps']} "
              f"post={cfw['post_steps']} dev_thresh={cfw['dev_thresh_m']}m")
    #create the train dataset (train-time perception dropout)
    train_kwargs = dict(
        traj_root=Path(cfg["data"]["train_dir"]),
        max_trajectories=cfg["data"].get("max_train_trajectories"),
        perception_dropout=tuple(cfg["data"]["perception_dropout"]),
        ego_blackout=cfg["data"].get("ego_blackout", False),
        **common_dataset_kwargs,
    )
    # DatasetManager path (data.dataset_manager, opt-in; ported from the edge
    # trainer 2026-07-16): normal + cf-window view + recovery trajectories as
    # weighted entries of ONE ConcatDataset — emphasis via dataset-mass shares
    # (WeightedRandomSampler below), not index filters or per-sample boosts.
    # Mutually exclusive with the single-dataset balancing/swap knobs: those
    # poke attributes (samples_index, chunk_theta_deg, trajectories) that a
    # ConcatDataset does not have.
    dm_cfg = cfg["data"].get("dataset_manager") or {}
    use_manager = dm_cfg.get("enabled", False)
    # mgr_sample_w: per-sample masses when the manager path is active (None on
    # the plain path). sanity_dataset: a single IsaacSimDataset for
    # verify_dataset_sanity, which needs attributes ConcatDataset lacks —
    # always the "normal" entry (all entries share the sample format).
    mgr_sample_w = None
    if use_manager:
        for excl in ("counterfactual_windows", "cf_window_balance",
                     "curvature_balance", "cf_onboard_swap",
                     "cf_onboard_swap_extra"):
            if (cfg["data"].get(excl) or {}).get("enabled", False):
                raise ValueError(
                    f"data.dataset_manager is mutually exclusive with "
                    f"data.{excl} (manager expresses emphasis as weighted "
                    f"entries; the single-dataset knobs don't compose with a "
                    f"ConcatDataset).")
        from prismatic.vla.datasets.counterfactual_dataset_2 import DatasetManager
        mgr = DatasetManager(dataset_cls=IsaacSimDataset)
        mgr.add_dataset("normal", weight=float(dm_cfg["normal_weight"]),
                        dataset_kwargs=train_kwargs)
        cf_sub = dm_cfg.get("counterfactual") or {}
        if cf_sub.get("enabled", False):
            mgr.create_counterfactual(
                "normal", name="cf", weight=float(cf_sub["weight"]),
                pre_steps=cf_sub["pre_steps"], post_steps=cf_sub["post_steps"],
                dev_thresh_m=cf_sub["dev_thresh_m"],
                criterion=cf_sub.get("criterion", "distance"),
                yaw_eps_deg=float(cf_sub.get("yaw_eps_deg", 1.0)))
        rec_sub = dm_cfg.get("recovery") or {}
        if rec_sub.get("enabled", False):
            mgr.register_recovery(
                "recovery", traj_root=Path(rec_sub["traj_root"]),
                weight=float(rec_sub["weight"]), parent="normal",
                anchor_after_handoff=rec_sub.get("anchor_after_handoff"),
                anchor_rejoin=rec_sub.get("anchor_rejoin"))
        train_dataset, mgr_sample_w = mgr.build()
        sanity_dataset = mgr.dataset("normal")
    else:
        train_dataset = IsaacSimDataset(**train_kwargs)
        if cfw.get("enabled", False):
            apply_cf_window_filter(
                train_dataset, pre_steps=cfw["pre_steps"],
                post_steps=cfw["post_steps"], dev_thresh_m=cfw["dev_thresh_m"])
        sanity_dataset = train_dataset
    # Counterfactual onboard-bundle swap (data.cf_onboard_swap, opt-in;
    # 2026-07-12): inside decision windows, serve the cf SIBLING's ego obs +
    # ego goal + goal_pose with the sample's own CCTV + labels, so the onboard
    # drift stops predicting the branch and the loss must be paid through the
    # CCTV stream. Pair-gated: only where the seal is NOT onboard-visible
    # (aligned-frame ego diff below ego_diff_thresh_px). TRAIN only — val
    # stays untouched so cf-val keeps measuring the honest input.
    # Diagonal-invisible EXTRA-SAMPLE swap (Lukas 2026-07-12): honest samples
    # untouched; eligible frames (differing south seal diagonal to the exit
    # aisle, [t*-pre, t*+post]) each appended once as a forced-swap duplicate
    # (sibling ego bundle + own CCTV/labels). Requires final_goal_prob=1.0.
    cose = cfg["data"].get("cf_onboard_swap_extra") or {}
    if cose.get("enabled", False):
        from prismatic.vla.datasets.dataset_utils import build_onboard_swap_extra
        extra_diag = build_onboard_swap_extra(
            train_dataset,
            pre_steps=cose["pre_steps"],
            post_steps=cose["post_steps"],
            dev_thresh_m=cose["dev_thresh_m"],
            max_offset_m=cose["max_offset_m"],
            split_x_boundary=cose["split_x_boundary"],
            split_y_boundary=cose["split_y_boundary"],
        )
        if extra_diag["n_extra"] == 0:
            raise RuntimeError("cf_onboard_swap_extra: no eligible frames — "
                               "check the diagonal gate knobs")
    # val_swap side channel (Lukas 2026-07-12): a THIRD val loader serving
    # ONLY forced-swap extras built from the VAL split's diagonal-invisible
    # pairs — the over-training curve of "predicts the branch when the onboard
    # bundle is crossed and only CCTV + label co-vary", on held-out scenes.
    # Logged as val_swap/*; reporting only, never checkpoint selection.
    val_swap_dataset = None
    if cose.get("enabled", False) and cose.get("val_report", True):
        # No window filter here: the honest index is emptied below, so only
        # the extras (gated by the cose range) are ever served. (The retired
        # subclass version also implicitly required counterfactual_windows to
        # be enabled for its constructor kwargs — that coupling is gone.)
        val_swap_dataset = IsaacSimDataset(
            traj_root=Path(cfg["data"]["val_dir"]),
            max_trajectories=cfg["data"].get("max_val_trajectories", 20),
            perception_dropout=tuple(cfg["data"]["val_perception_dropout"]),
            ego_blackout=cfg["data"].get("ego_blackout", False),
            **{**common_dataset_kwargs, "random_hflip": False, "seed": seed + 1,
               "cctv_cam_dropout": 0.0,
               "deterministic_sampling": cfg["data"].get("fixed_val_sampling", True)},
        )
        build_onboard_swap_extra(
            val_swap_dataset,
            pre_steps=cose["pre_steps"], post_steps=cose["post_steps"],
            dev_thresh_m=cose["dev_thresh_m"], max_offset_m=cose["max_offset_m"],
            split_x_boundary=cose["split_x_boundary"],
            split_y_boundary=cose["split_y_boundary"],
        )
        # Serve ONLY the forced-swap extras: with the honest index emptied,
        # every idx dispatches to onboard_swap_extra in __getitem__.
        val_swap_dataset.samples_index = []
        if len(val_swap_dataset) == 0:
            print("[val_swap] no eligible val pairs — channel disabled")
            val_swap_dataset = None
        else:
            print(f"[val_swap] {len(val_swap_dataset)} forced-swap val samples")
    cos = cfg["data"].get("cf_onboard_swap") or {}
    if cos.get("enabled", False):
        from prismatic.vla.datasets.dataset_utils import build_onboard_swap_map
        swap_diag = build_onboard_swap_map(
            train_dataset,
            pre_steps=cos["pre_steps"],
            post_steps=cos["post_steps"],
            dev_thresh_m=cos["dev_thresh_m"],
            max_offset_m=cos["max_offset_m"],
            p=cos["p"],
            junction_filter=cos["junction_filter"],
            split_y_boundary=cos.get("split_y_boundary"),
            ego_diff_thresh_px=cos.get("ego_diff_thresh_px"),
            # real-robot spawn-frame mode (2026-08-21): swap only at t <
            # first_frames, gated on estimated graft endpoint error.
            first_frames=cos.get("first_frames"),
            graft_err_cap_m=cos.get("graft_err_cap_m"),
        )
        if swap_diag["swappable_samples"] == 0:
            raise RuntimeError(
                "cf_onboard_swap: no swappable samples — every pair failed the "
                "ego-visibility gate or the offset cap; check the knobs")
    #create the val dataset — perception fixed (val_perception_dropout) so the
    #val-loss curve is stable/comparable; per-mode comparison comes from eval.
    val_dataset = IsaacSimDataset(
        traj_root=Path(cfg["data"]["val_dir"]),
        max_trajectories=cfg["data"].get("max_val_trajectories", 20),
        perception_dropout=tuple(cfg["data"]["val_perception_dropout"]),
        ego_blackout=cfg["data"].get("ego_blackout", False),
        # val: no per-camera dropout — all cctv views present for a stable,
        # comparable val curve (matches the all-present eval protocol).
        # fixed_val_sampling (default ON, 2026-07-09): pin every per-sample
        # draw (goal k / final-goal coin, modality) to the sample's identity so
        # the val set is byte-identical each pass — best-val selection compares
        # weights, not sampling luck. NOTE: val losses are NOT comparable with
        # runs trained before this date (different goal/modality draws). Set
        # data.fixed_val_sampling: false to reproduce the legacy behaviour.
        **{**common_dataset_kwargs, "random_hflip": False, "seed": seed + 1,
           "cctv_cam_dropout": 0.0,
           "deterministic_sampling": cfg["data"].get("fixed_val_sampling", True)},
    )
    if cfw.get("enabled", False):
        apply_cf_window_filter(
            val_dataset, pre_steps=cfw["pre_steps"],
            post_steps=cfw["post_steps"], dev_thresh_m=cfw["dev_thresh_m"])
    # Optional counterfactual-val side channel (data.cf_val_report): build a
    # SECOND val dataset = the decision-window subset of the val split, and run
    # it alongside the main val each val_freq (logged as val_cf/*). The main
    # val stays the FULL set so the curve + best-checkpoint selection remain
    # comparable to the parent arms; cf-val is reporting only. Same seed (+1)
    # and deterministic per-(traj,t) sampling as the main val, so the cf-val
    # numbers are directly comparable across runs (incl. the cfwin_ft arm,
    # whose main val IS this subset). All keys required — YAML owns the knobs.
    cvr = cfg["data"].get("cf_val_report") or {}
    cf_val_dataset = None
    if cvr.get("enabled", False):
        if cfw.get("enabled", False):
            raise ValueError(
                "data.cf_val_report is redundant when data.counterfactual_windows "
                "is enabled — the main val already IS the cf-window subset.")
        cf_val_dataset = IsaacSimDataset(
            traj_root=Path(cfg["data"]["val_dir"]),
            max_trajectories=cfg["data"].get("max_val_trajectories", 20),
            perception_dropout=tuple(cfg["data"]["val_perception_dropout"]),
            ego_blackout=cfg["data"].get("ego_blackout", False),
            # mirror the main val's overrides exactly (stable, deterministic)
            **{**common_dataset_kwargs, "random_hflip": False, "seed": seed + 1,
               "cctv_cam_dropout": 0.0,
               "deterministic_sampling": cfg["data"].get("fixed_val_sampling", True)},
        )
        apply_cf_window_filter(
            cf_val_dataset, pre_steps=cvr["pre_steps"],
            post_steps=cvr["post_steps"], dev_thresh_m=cvr["dev_thresh_m"],
            tag="cf_val_report",
            criterion=cvr.get("criterion", "distance"),
            yaw_eps_deg=float(cvr.get("yaw_eps_deg", 1.0)))
    #get the collator (CCTV variant: 3- or 4-image channel concat + block_present)
    collator = PaddedCollatorForActionPrediction_Nav_CCTV(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
        num_img=cfg["model"]["num_images_in_input"],
    )
    # Curvature-balanced sampling (data.curvature_balance in the YAML, opt-in).
    # RESAMPLES rare turn-heavy chunks (WeightedRandomSampler, replacement=True)
    # instead of loss-upweighting them: same emphasis in expectation, but the
    # emphasis is spread across batches rather than spiking single gradients
    # (Buda et al. 2018). |theta| classes via np.digitize over class_edges_deg:
    # edges [10,20,30] -> classes <10 / 10-20 / 20-30 / >=30, one weight each.
    # Motivation: cfv2fwd is 8.3:1 straight:curve; 55% of loss mass is straight
    # cruise (see isaac_sim/outputs/train_curvature/ 2026-07-07).
    cb = cfg["data"].get("curvature_balance") or {}
    # sample_w accumulates MULTIPLICATIVELY across the (independent) balancing
    # axes below — curvature and cf-window boosts compose; None = no resampling.
    # Seeded with the manager's per-entry masses when that path is active (the
    # balancing knobs below are then excluded, so no double-weighting).
    sample_w = mgr_sample_w
    if cb.get("enabled", False):
        theta = train_dataset.chunk_theta_deg()
        edges = np.asarray(cb["class_edges_deg"], dtype=float)
        class_w = np.asarray(cb["class_weights"], dtype=float)
        assert len(class_w) == len(edges) + 1, \
            f"curvature_balance: need len(class_weights) == len(class_edges_deg)+1"
        cls = np.digitize(theta, edges)          # exact edge value -> upper class
        sample_w = class_w[cls]
        counts = np.bincount(cls, minlength=len(class_w))
        mass = counts * class_w / float((counts * class_w).sum())
        print(f"[curvature_balance] edges={edges.tolist()} weights={class_w.tolist()}")
        print(f"[curvature_balance] class counts={counts.tolist()} "
              f"sampling mass={[round(float(m), 3) for m in mass]}")
    # Counterfactual-window UPWEIGHTING (data.cf_window_balance, opt-in): keep
    # the full dataset (cruise supervision stays in) but boost the resampling
    # weight of samples inside a decision window — the single-stage alternative
    # to the counterfactual_windows fine-tune. Gradient mass on the crux =
    # f*boost / (f*boost + (1-f)) with f = window fraction (~0.114 on cfv3), so
    # boost 8 ~= 50/50 crux:cruise. TRAIN only — val stays the full set for
    # comparability with the parent arms (cf-val comes from --val-only with
    # counterfactual_windows enabled). All keys required: YAML owns the knobs.
    cwb = cfg["data"].get("cf_window_balance") or {}
    if cwb.get("enabled", False):
        if cfw.get("enabled", False):
            raise ValueError(
                "data.cf_window_balance and data.counterfactual_windows are "
                "mutually exclusive: boosting inside an already-window-filtered "
                "dataset is a uniform no-op. Enable exactly one.")
        cf_mask, cf_mining = cf_window_sample_mask(
            train_dataset,
            pre_steps=cwb["pre_steps"],
            post_steps=cwb["post_steps"],
            dev_thresh_m=cwb["dev_thresh_m"],
            criterion=cwb.get("criterion", "distance"),
            yaw_eps_deg=float(cwb.get("yaw_eps_deg", 1.0)),
        )
        boost = float(cwb["boost"])
        if boost <= 0.0:
            raise ValueError(f"cf_window_balance.boost must be > 0, got {boost}")
        n_in = int(cf_mask.sum())
        if n_in == 0:
            raise ValueError(
                "cf_window_balance: no samples fell inside a decision window — "
                "is this really a counterfactual (_cf) split?")
        w = np.where(cf_mask, boost, 1.0)
        crux_mass = n_in * boost / (n_in * boost + (len(cf_mask) - n_in))
        n_bad = sum(1 for e in cf_mining["pair_log"] if e["status"] != "ok")
        print(f"[cf_window_balance] {n_in}/{len(cf_mask)} window samples "
              f"({100.0 * n_in / len(cf_mask):.1f}%), boost={boost} -> "
              f"crux sampling mass {crux_mass:.3f} | "
              f"{cf_mining['n_families']} families, {n_bad} non-ok pairs, "
              f"pre={cwb['pre_steps']} post={cwb['post_steps']} "
              f"criterion={cwb.get('criterion', 'distance')} "
              f"thresh={cwb['dev_thresh_m']}m/eps={cwb.get('yaw_eps_deg', 1.0)}deg")
        for entry in cf_mining["pair_log"]:
            if entry["status"] != "ok":
                print(f"[cf_window_balance]   WARNING {entry['status']}: "
                      f"{entry['base']} cf{entry['a']} vs cf{entry['b']}")
        sample_w = w if sample_w is None else sample_w * w
    train_sampler = None
    if sample_w is not None:
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),           # one "epoch" = dataset-size draws
            replacement=True,
        )
    #get the train data loader (sampler and shuffle are mutually exclusive)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["train"].get("num_workers", 4) > 0,
    )
    #get the val datset
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg["train"].get("num_workers", 4) > 0,
    )
    # cf-val side channel loader (None when data.cf_val_report is off).
    cf_val_loader = None
    if cf_val_dataset is not None:
        cf_val_loader = DataLoader(
            cf_val_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg["train"].get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
            persistent_workers=cfg["train"].get("num_workers", 4) > 0,
        )
    # val_swap side channel loader (None unless cf_onboard_swap_extra + val_report).
    val_swap_loader = None
    if val_swap_dataset is not None:
        val_swap_loader = DataLoader(
            val_swap_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=0,      # 36 samples — not worth worker processes
            pin_memory=True,
            drop_last=False,
        )

    # Loss-term weights (YAML `loss` section). Read once here and threaded into
    # every forward_and_loss call so the coefficients are never hardcoded.
    smoothness_weight = cfg["loss"]["smoothness_weight"]
    obj_weight = cfg["loss"]["obj_weight"]

    # --- --val-only: fixed-val checkpoint sweep -------------------------------
    # Load ONE checkpoint's weights (no optimizer/training_state), run a single
    # val pass on the (deterministic, frozen) val set, print a parseable line,
    # exit. Skips the VRAM verify / optimizer / resume / training loop below.
    # A wrapper loops this over checkpoints for an apples-to-apples comparison.
    if args.val_only:
        assert args.resume_step is not None, "--val-only needs --resume-step to pick the checkpoint"
        ckpt_dir = run_dir / "checkpoints" / f"step_{args.resume_step:05d}"
        print(f"\n=== --val-only: loading step {args.resume_step} weights ===")
        load_eval_modules(
            ckpt_dir, vla, pose_projector, action_head,
            cctv_pixel_shuffle=cctv_pixel_shuffle, cctv_cam_embed=cctv_cam_embed,
            abs_pose_projector=abs_pose_projector,
        )
        m = run_validation(
            val_loader, vla, pose_projector, action_head,
            num_patches=num_patches, device=device,
            pad_token_id=processor.tokenizer.pad_token_id,
            model_max_length=processor.tokenizer.model_max_length,
            smoothness_weight=smoothness_weight, obj_weight=obj_weight,
            max_samples=cfg["val"].get("max_samples"),
            abs_pose_projector=abs_pose_projector,
        )
        print(f"VALSWEEP step={args.resume_step} loss_main={m['loss_main']:.6f} "
              f"n_batches={m.get('n_batches', '?')}")
        # cf-val side channel in the sweep too (2026-08-21): lets a run that was
        # trained WITHOUT data.cf_val_report be scored on the same decision-window
        # subset retrospectively, through the identical run_validation path — the
        # only way to place a baseline arm against arms that logged cf-val live.
        if cf_val_loader is not None:
            mc = run_validation(
                cf_val_loader, vla, pose_projector, action_head,
                num_patches=num_patches, device=device,
                pad_token_id=processor.tokenizer.pad_token_id,
                model_max_length=processor.tokenizer.model_max_length,
                smoothness_weight=smoothness_weight, obj_weight=obj_weight,
                max_samples=cfg["val"].get("max_samples"),
                abs_pose_projector=abs_pose_projector,
            )
            print(f"VALSWEEP step={args.resume_step} cf_loss_main={mc['loss_main']:.6f} "
                  f"n_batches={mc.get('n_batches', '?')}")
        return

    print(f"\n=== Startup verification ===")
    #verify the dataset sanity function define earlier does checks with max etc. 
    # manager path: ConcatDataset lacks the attributes sanity checks poke —
    # verify the "normal" entry (all entries share the sample format).
    verify_dataset_sanity(sanity_dataset, n_check=100)
    verify_vram_headroom(
        vla, pose_projector, action_head, val_loader,
        num_patches=num_patches, device=device,
        pad_token_id=processor.tokenizer.pad_token_id,
        model_max_length=processor.tokenizer.model_max_length,
        smoothness_weight=smoothness_weight, obj_weight=obj_weight,
        abs_pose_projector=abs_pose_projector,
    )
    # get trainable params 
    trainable_params = (
        [p for p in vla.parameters() if p.requires_grad]
        + [p for p in pose_projector.parameters() if p.requires_grad]
        + [p for p in action_head.parameters() if p.requires_grad]
    )
    # v2: the standalone pixel-shuffle module trains fully (not in `vla`, so wrap_lora
    # never saw it and it isn't in vla.parameters()).
    if cctv_pixel_shuffle is not None:
        trainable_params += [p for p in cctv_pixel_shuffle.parameters() if p.requires_grad]
    # v2: per-camera identity embedding — raw Parameter (not in `vla`), full-trained peer.
    if cctv_cam_embed is not None:
        trainable_params += [cctv_cam_embed]
    # v2: absolute-pose encoder — standalone peer module, full-trained from scratch.
    if abs_pose_projector is not None:
        trainable_params += [p for p in abs_pose_projector.parameters() if p.requires_grad]
    #print stuff
    print(f"\n=== Optimizer ===")
    print(f"  total trainable params: {sum(p.numel() for p in trainable_params):,}")
    #get the optimizer why are we havgint these default values here, kind of weird because we won't know when stuff breaks
    optimizer = AdamW(
        trainable_params,
        lr=cfg["train"]["learning_rate"],
        betas=tuple(cfg["train"].get("betas", (0.9, 0.999))),
        eps=cfg["train"].get("eps", 1e-8),
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    #get a scheduler -> find out what shceudler this is actually
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg["train"]["num_steps_before_decay"]],
        gamma=cfg["train"]["lr_decay_gamma"],
    )
    #where to store the chckpoints 
    start_step = 0
    if args.resume_step is not None:
        resume_dir = run_dir / "checkpoints" / f"step_{args.resume_step:05d}"
        print(f"\n=== Resuming from step {args.resume_step} ===")
        start_step = load_full_state(
            resume_dir, vla, pose_projector, action_head, optimizer, scheduler,
            cctv_pixel_shuffle=cctv_pixel_shuffle,
            cctv_cam_embed=cctv_cam_embed,
            abs_pose_projector=abs_pose_projector,
        )

    print(f"\n=== Starting training (step {start_step} → {cfg['train']['max_steps']}) ===")
    max_steps = cfg["train"]["max_steps"]
    grad_accum = cfg["train"]["grad_accumulation_steps"]
    log_freq = cfg["logging"]["log_freq"]
    val_freq = cfg["val"]["val_freq"]
    save_freq = cfg["checkpoint"]["save_freq"]
    keep_top_k = cfg["checkpoint"]["keep_top_k"]
    keep_latest = cfg["checkpoint"]["keep_latest"]
    keep_every_n_steps = cfg["checkpoint"].get("keep_every_n_steps", 0)

    train_loss_buffer = deque(maxlen=log_freq)
    step = start_step
    accum_loss = 0.0
    t_loop_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    vla.train()
    pose_projector.train()
    action_head.train()
    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            raw_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            raw_batch = next(train_iter)

        batch = pad_and_mask_batch(
            raw_batch, processor.tokenizer.pad_token_id, processor.tokenizer.model_max_length,
        )
        loss, metrics = forward_and_loss(
            batch, vla, pose_projector, action_head,
            num_patches=num_patches, device=device, train_mode=True,
            smoothness_weight=smoothness_weight, obj_weight=obj_weight,
            abs_pose_projector=abs_pose_projector,
        )
        (loss / grad_accum).backward()
        accum_loss += metrics["loss"]

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss_buffer.append(metrics["loss"])
        step += 1
        
        if step % log_freq == 0:
            elapsed = time.time() - t_loop_start
            steps_per_s = log_freq / max(elapsed, 1e-6)
            t_loop_start = time.time()
            avg_loss = sum(train_loss_buffer) / len(train_loss_buffer)
            wandb.log({
                "train/loss": avg_loss,
                "train/loss_main": metrics["loss_main"],
                "train/loss_smooth": metrics["loss_smooth"],
                "train/loss_obj": metrics["loss_obj"],
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/steps_per_s": steps_per_s,
                **{f"train/{k}": v for k, v in metrics.items() if k.startswith(("l1_modality_", "l1_axis_", "l1_cond_", "frac_cond_"))},
            }, step=step)
            print(f"  step {step:>6} | loss={avg_loss:.4f} | {steps_per_s:.2f} step/s")

        if step % val_freq == 0 or step == max_steps:
            print(f"\n  >>> Validation at step {step}")
            val_metrics = run_validation(
                val_loader, vla, pose_projector, action_head,
                num_patches=num_patches, device=device,
                pad_token_id=processor.tokenizer.pad_token_id,
                model_max_length=processor.tokenizer.model_max_length,
                smoothness_weight=smoothness_weight, obj_weight=obj_weight,
                max_samples=cfg["val"]["max_samples"],
                abs_pose_projector=abs_pose_projector,
            )
            print(f"      val loss_main={val_metrics['loss_main']:.4f}")
            wandb.log({f"val/{k}": v for k, v in val_metrics.items() if k != "n_batches"}, step=step)

            # cf-val side channel: same pass on the decision-window subset.
            # Reporting only — checkpoint selection below stays on the FULL val.
            if cf_val_loader is not None:
                cf_val_metrics = run_validation(
                    cf_val_loader, vla, pose_projector, action_head,
                    num_patches=num_patches, device=device,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    model_max_length=processor.tokenizer.model_max_length,
                    smoothness_weight=smoothness_weight, obj_weight=obj_weight,
                    max_samples=cfg["val"]["max_samples"],
                    abs_pose_projector=abs_pose_projector,
                )
                print(f"      cf-val loss_main={cf_val_metrics['loss_main']:.4f}")
                wandb.log({f"val_cf/{k}": v for k, v in cf_val_metrics.items()
                           if k != "n_batches"}, step=step)

            # val_swap side channel: loss on FORCED-SWAP val samples (crossed
            # onboard bundle, own CCTV + labels) — the CCTV-reliance curve.
            if val_swap_loader is not None:
                val_swap_metrics = run_validation(
                    val_swap_loader, vla, pose_projector, action_head,
                    num_patches=num_patches, device=device,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    model_max_length=processor.tokenizer.model_max_length,
                    smoothness_weight=smoothness_weight, obj_weight=obj_weight,
                    max_samples=None,
                    abs_pose_projector=abs_pose_projector,
                )
                print(f"      val_swap loss_main={val_swap_metrics['loss_main']:.4f}")
                wandb.log({f"val_swap/{k}": v for k, v in val_swap_metrics.items()
                           if k != "n_batches"}, step=step)

            if step % save_freq == 0 or step == max_steps:
                save_checkpoint(
                    run_dir, step, vla, pose_projector, action_head,
                    cctv_pixel_shuffle,
                    optimizer, scheduler,
                    rng_state=get_rng_state(),
                    val_metrics=val_metrics,
                    cctv_cam_embed=cctv_cam_embed,
                    abs_pose_projector=abs_pose_projector,
                )
                prune_old_checkpoints(
                    run_dir, keep_top_k=keep_top_k, keep_latest=keep_latest,
                    keep_every_n_steps=keep_every_n_steps,
                )

        if args.smoke_test and step >= args.smoke_test:
            print(f"\n>>> Smoke test: forcing a final val pass + checkpoint save")
            val_metrics = run_validation(
                val_loader, vla, pose_projector, action_head,
                num_patches=num_patches, device=device,
                pad_token_id=processor.tokenizer.pad_token_id,
                model_max_length=processor.tokenizer.model_max_length,
                smoothness_weight=smoothness_weight, obj_weight=obj_weight,
                max_samples=cfg["val"]["max_samples"],
                abs_pose_projector=abs_pose_projector,
            )
            print(f"    val loss_main={val_metrics['loss_main']:.4f}")
            save_checkpoint(
                run_dir, step, vla, pose_projector, action_head,
                cctv_pixel_shuffle,
                optimizer, scheduler,
                rng_state=get_rng_state(),
                val_metrics=val_metrics,
                cctv_cam_embed=cctv_cam_embed,
                abs_pose_projector=abs_pose_projector,
            )
            print(f"\n=== Smoke test complete: {step} train steps + 1 val pass + 1 checkpoint ===")
            break

    print(f"\n=== Training complete @ step {step} ===")
    wandb.finish()


if __name__ == "__main__":
    main()