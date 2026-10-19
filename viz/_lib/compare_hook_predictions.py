
from __future__ import annotations

import argparse, json, sys
import pathlib
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "viz/_lib")); sys.path.insert(0, str(_REPO / "vla-scripts"))

import train_omnivla_sim_cctv_v2 as T2
from cf_counterfactual_probe import build_model, predict_chunk
from prismatic.vla.datasets.cctv_isaac_sim_dataset import IsaacSimDataset
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_CCTV

OUT_DIR = _REPO / "isaac_sim" / "outputs" / "hook_predictions"


def build_split(cfg: dict, processor, split_dir: Path, seed: int) -> IsaacSimDataset:
    """Exactly the trainer's dataset for this config, on `split_dir`."""
    d = cfg["data"]
    return IsaacSimDataset(
        action_tokenizer=ActionTokenizer(processor.tokenizer),
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        traj_root=split_dir,
        max_trajectories=None,
        context_size=d["context_size"],
        num_actions_chunk=d["num_actions_chunk"],
        modality_list=tuple(d["modality_list"]),
        modality_weights=d.get("modality_weights"),
        lang_manifest=d["lang_manifest"],
        lang_pattern=d["lang_pattern"],
        first_step_only=d.get("first_step_only", False),
        image_only_max_distance_steps=d["image_only_max_distance_steps"],
        min_goal_distance_steps=d["min_goal_distance_steps"],
        max_goal_distance_steps=d["max_goal_distance_steps"],
        metric_waypoint_spacing=d["metric_waypoint_spacing"],
        waypoint_spacing=d["waypoint_spacing"],
        image_size=tuple(d["image_size"]),
        random_hflip=False,
        goals_per_obs=d["goals_per_obs"],
        emit_cctv=d["emit_cctv"],
        use_cctv_goal=d.get("use_cctv_goal", False),
        cctv_cam_names=tuple(d.get("cctv_cam_names", ("cctv",))),
        no_goal_image=d.get("no_goal_image", False),
        perception_dropout=tuple(d["val_perception_dropout"]),
        ego_blackout=d.get("ego_blackout", False),
        final_goal_prob=d.get("final_goal_prob", 0.0),
        predict_stop_token=d["predict_stop_token"],
        seed=seed,
    )


def run_split(ds, cfg, model_bits, tag: str) -> list[dict]:
    processor, vla, pp, ah, num_patches, app = model_bits
    collator = PaddedCollatorForActionPrediction_Nav_CCTV(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id,
        padding_side="right", num_img=cfg["model"]["num_images_in_input"])
    pad_id, max_len = processor.tokenizer.pad_token_id, processor.tokenizer.model_max_length

    rows = []
    for i in range(len(ds)):
        s = ds[i]
        batch = T2.pad_and_mask_batch(collator([s]), pad_id, max_len)
        with torch.no_grad():
            pred = predict_chunk(batch, vla, pp, ah, num_patches, "cuda",
                                 abs_pose_projector=app)
        gt = np.asarray(s["actions"], dtype=np.float32)
        name = ds.samples[i].traj_dir.name if hasattr(ds, "samples") else str(i)
        rows.append({
            "split": tag, "idx": i, "traj": name,
            "gt_lat_w8": float(gt[-1, 1]),
            "pred_lat_w8": float(pred[-1, 1]),
            "correct": bool(np.sign(gt[-1, 1]) == np.sign(pred[-1, 1])),
            "l1": float(np.abs(pred - gt).mean()),
            "gt": gt.tolist(), "pred": pred.tolist(),
        })
        print(f"  [{tag} {i:3d}] gt_lat={gt[-1,1]:+6.2f} pred_lat={pred[-1,1]:+6.2f} "
              f"{'OK ' if rows[-1]['correct'] else 'MISS'} L1={rows[-1]['l1']:.3f} "
              f"{name[:44]}", flush=True)
    return rows


def baselines(train_rows: list[dict], rows: list[dict], n_wp: int) -> dict:
    """L1 of the two trivial predictors on `rows`, for scale reference."""
    straight = np.zeros((n_wp, 4), dtype=np.float32)
    for k in range(n_wp):
        straight[k] = (k + 1.0, 0.0, 1.0, 0.0)
    mean_const = np.mean(np.array([r["gt"] for r in train_rows], dtype=np.float32), axis=0)
    out = {}
    for nme, ref in (("straight", straight), ("mean_const", mean_const)):
        l1 = [float(np.abs(ref - np.array(r["gt"], dtype=np.float32)).mean()) for r in rows]
        out[nme] = float(np.mean(l1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", type=int, required=True)
    args = ap.parse_args()

    cfg = T2.load_yaml_config(Path(args.config))
    T2.register_omnivla_classes()
    print("=== building model ===", flush=True)
    model_bits = build_model(cfg, Path(args.ckpt), args.step, "cuda")
    processor = model_bits[0]
    seed = int(cfg["train"].get("seed", 42))

    n_wp = int(cfg["data"]["num_actions_chunk"])
    ds_tr = build_split(cfg, processor, Path(cfg["data"]["train_dir"]), seed)
    ds_va = build_split(cfg, processor, Path(cfg["data"]["val_dir"]), seed)

    print("=== TRAIN ===", flush=True)
    tr = run_split(ds_tr, cfg, model_bits, "train")
    print("=== VAL ===", flush=True)
    va = run_split(ds_va, cfg, model_bits, "val")

    summary = {}
    for tag, rows in (("train", tr), ("val", va)):
        acc = float(np.mean([r["correct"] for r in rows]))
        summary[tag] = {
            "n": len(rows),
            "direction_accuracy": acc,
            "mean_l1": float(np.mean([r["l1"] for r in rows])),
            "mean_abs_pred_lat_w8": float(np.mean([abs(r["pred_lat_w8"]) for r in rows])),
            "mean_abs_gt_lat_w8": float(np.mean([abs(r["gt_lat_w8"]) for r in rows])),
            "baseline_l1": baselines(tr, rows, n_wp),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"hook_preds_step{args.step}.json").write_text(
        json.dumps({"summary": summary, "rows": tr + va}, indent=2))

    # ---- panels: GT vs predicted chunk, robot frame (x forward, y left) ----
    for tag, rows in (("train", tr), ("val", va)):
        n = min(len(rows), 24)
        ncol = 6; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.5 * nrow), squeeze=False)
        for j in range(nrow * ncol):
            ax = axes[j // ncol][j % ncol]
            if j >= n:
                ax.axis("off"); continue
            r = rows[j]
            g = np.array(r["gt"]); p = np.array(r["pred"])
            ax.plot(g[:, 1], g[:, 0], "-o", ms=3, color="tab:green", label="GT")
            ax.plot(p[:, 1], p[:, 0], "-o", ms=3, color="tab:red", label="pred")
            ax.axvline(0, color="k", lw=0.5, alpha=0.4)
            ax.set_title(("OK " if r["correct"] else "MISS ") + f"L1={r['l1']:.2f}",
                         fontsize=8, color=("green" if r["correct"] else "red"))
            ax.set_aspect("equal"); ax.invert_xaxis()   # +y left -> left on screen
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.legend(fontsize=6)
        s = summary[tag]
        fig.suptitle(f"{tag}: direction acc {s['direction_accuracy']:.2f} "
                     f"(chance 0.50, n={s['n']})  |  L1 {s['mean_l1']:.3f} "
                     f"(straight {s['baseline_l1']['straight']:.3f}, "
                     f"mean-const {s['baseline_l1']['mean_const']:.3f})", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"hook_{tag}_step{args.step}.png", dpi=125)

    print("\n=== SUMMARY ===")
    for tag in ("train", "val"):
        s = summary[tag]
        print(f"  {tag:5s} n={s['n']:3d}  dir-acc={s['direction_accuracy']:.3f} (chance 0.500)  "
              f"L1={s['mean_l1']:.3f}  [straight {s['baseline_l1']['straight']:.3f}, "
              f"mean-const {s['baseline_l1']['mean_const']:.3f}]  "
              f"|pred lat w8|={s['mean_abs_pred_lat_w8']:.2f} vs |gt|={s['mean_abs_gt_lat_w8']:.2f}")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
