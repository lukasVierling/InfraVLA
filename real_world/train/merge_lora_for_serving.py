"""Bake a LoRA fine-tune into a self-contained serving checkpoint.

WHY THIS EXISTS
    inference/run_omnivla.py::define_model loads the VLA with
    `AutoModelForVision2Seq.from_pretrained(vla_path)` and then loads the
    pose_projector / action_head. It NEVER applies a LoRA adapter — the
    `lora_adapter/` directory inside omnivla-original is decorative as far as
    inference is concerned.

    Point the policy server at a raw training checkpoint and you get base
    backbone weights driving fine-tuned heads. Nothing errors; the numbers are
    just wrong. Merging removes the whole failure mode instead of adding a
    load-adapter flag to shared inference code.

    python real_world/train/merge_lora_for_serving.py --v2 \
        --base omnivla-original \
        --ckpt runs/training/real_world/<run>/checkpoints/step_03000 \
        --step 3000 \
        --out  checkpoints/<run>_step3000_merged \
        --config real_world/train/configs/<run config>.yaml

Runs on CPU by default (~15 GB bf16) so it does not fight a busy GPU.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# prismatic lives in the repo, not site-packages, on some machines
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
# v2 class: for checkpoints trained through train_omnivla_sim_cctv_v2 (swappable
# vision-encoder seam). The LoRA keys are name-compatible either way, but merging
# through the class the adapter was actually built around removes any doubt.
from prismatic.extern.hf.modeling_prismatic_v2 import OpenVLAForActionPrediction_MMNv1 as OpenVLA_V2
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# files that describe the model but are not weights; save_pretrained does not
# re-emit the custom code, and from_pretrained(trust_remote_code) needs it.
AUX = ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py",
       "preprocessor_config.json", "processor_config.json", "tokenizer.json",
       "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
       "generation_config.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True, help="dir with lora_adapter/ + head .pt files")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--v2", action="store_true",
                    help="load the v2 model class (CCTV-downsample checkpoints)")
    ap.add_argument("--config", default=None,
                    help="training YAML of the run; writes serving.json (spacing, cameras, "
                         "image slots) that real_world/live/serve_spot_ft.sh checks")
    args = ap.parse_args()

    base, ckpt, out = Path(args.base), Path(args.ckpt), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    print(f"loading base {base} ...")
    if args.v2:
        vla = OpenVLA_V2.from_pretrained(
            base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)

    adapter = ckpt / "lora_adapter"
    if not adapter.is_dir():
        raise SystemExit(f"no lora_adapter/ under {ckpt}")
    print(f"merging adapter {adapter} ...")
    before = vla.language_model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    vla = PeftModel.from_pretrained(vla, str(adapter))
    vla = vla.merge_and_unload()
    after = vla.language_model.model.layers[0].self_attn.q_proj.weight.detach()
    delta = (after.float() - before.float()).abs().max().item()
    print(f"  max |Δ| on layer0 q_proj after merge: {delta:.6g}")
    if delta == 0.0:
        raise SystemExit("merge changed NOTHING — the adapter did not apply. Refusing to "
                         "write a checkpoint that would silently serve base weights.")

    print(f"saving merged model -> {out}")
    vla.save_pretrained(out, safe_serialization=True)

    for name in AUX:
        src = base / name
        if src.exists():
            shutil.copy2(src, out / name)
    for head in (f"pose_projector--{args.step}_checkpoint.pt",
                 f"action_head--{args.step}_checkpoint.pt"):
        shutil.copy2(ckpt / head, out / head)
        print(f"  + {head}")

    # from_pretrained(trust_remote_code=True) resolves the custom classes through
    # config.auto_map; save_pretrained drops it when the class was registered
    # rather than loaded remotely.
    import json
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    base_cfg = json.loads((base / "config.json").read_text())
    if "auto_map" in base_cfg:
        cfg["auto_map"] = base_cfg["auto_map"]
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print("  + restored config.auto_map")
    spacing = None
    if args.config:
        import yaml
        tcfg = yaml.safe_load(Path(args.config).read_text())
        serving = {
            "run_name": tcfg["run"]["name"],
            "spacing_m": float(tcfg["data"]["metric_waypoint_spacing"]),
            "cctv_cam_names": list(tcfg["data"].get("cctv_cam_names") or []),
            "num_images_in_input": int(tcfg["model"]["num_images_in_input"]),
        }
        (out / "serving.json").write_text(json.dumps(serving, indent=2))
        spacing = serving["spacing_m"]
        print(f"  + serving.json {serving}")
    print("\ndone. serve with:")
    print(f"  CKPT={out} real_world/live/serve_spot_ft.sh")
    if spacing is None:
        print("  (no --config given: pass SPACING=<metric_waypoint_spacing of the run> explicitly)")


if __name__ == "__main__":
    main()
