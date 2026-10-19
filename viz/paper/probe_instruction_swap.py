
from __future__ import annotations

import argparse, json, re, sys
from collections import defaultdict
from itertools import combinations
import pathlib
from pathlib import Path

import numpy as np
import torch

_REPO = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "viz/_lib")); sys.path.insert(0, str(_REPO / "vla-scripts"))

import train_omnivla_sim_cctv_v2 as T2
from cf_counterfactual_probe import build_model, predict_chunk
from compare_hook_predictions import build_split
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_CCTV

OUT_DIR = _REPO / "isaac_sim" / "outputs" / "instruction_swap"
# Default = langv1 (2-twin tA/tB). langv2 passes --fam-prefix warehouse_2x3_langv2
# --twins ABCD (4 twins, one target per aisle); the metrics below are unchanged.
_FAM = re.compile(r"^(warehouse_2x3_lang_\d{4})_t([AB])__pose__")


def fam_regex(prefix: str, twins: str) -> re.Pattern:
    return re.compile(rf"^({re.escape(prefix)}_\d{{4}})_t([{twins}])__pose__")


def family_pairs(ds, fam_re: re.Pattern | None = None,
                 max_families: int | None = None) -> list[tuple[int, int]]:
    """Sample-index pairs for families with >=2 twins present.

    2 twins -> 1 pair (the langv1 case, identical to the original behaviour);
    4 twins -> all 6 unordered pairs. run() then evaluates each pair in BOTH
    directions, so every twin is probed with every sibling's instruction.
    """
    fam_re = fam_re or _FAM
    by_fam = defaultdict(dict)
    for i, (ti, t) in enumerate(ds.samples_index):
        name = ds.trajectories[ti].traj_dir.name
        m = fam_re.match(name)
        if m:
            by_fam[m.group(1)][m.group(2)] = i
    bases = sorted(by_fam)
    if max_families is not None:
        bases = bases[:max_families]          # cap cost on the big train split
    out: list[tuple[int, int]] = []
    for base in bases:
        twins_present = sorted(by_fam[base])  # e.g. ['A','B','C','D']
        for a, b in combinations(twins_present, 2):
            out.append((by_fam[base][a], by_fam[base][b]))
    return out


def scene_of(ds, i: int) -> str:
    ti, _ = ds.samples_index[i]
    return ds.trajectories[ti].traj_dir.name.split("__pose__")[0]


def predict_with_phrase(ds, i: int, phrase: str | None, collator, model_bits, cfg):
    """Predict sample i, optionally overriding its instruction phrase.

    The dataset looks the phrase up as `self._phrase_by_scene[scene]` when it
    builds the prompt, so swapping that entry and re-fetching the sample changes
    the instruction and NOTHING else.
    """
    processor, vla, pp, ah, num_patches, app = model_bits
    scene = scene_of(ds, i)
    saved = ds._phrase_by_scene[scene]
    if phrase is not None:
        ds._phrase_by_scene[scene] = phrase
    try:
        s = ds[i]
    finally:
        ds._phrase_by_scene[scene] = saved
    batch = T2.pad_and_mask_batch(
        collator([s]), processor.tokenizer.pad_token_id,
        processor.tokenizer.model_max_length)
    with torch.no_grad():
        pred = predict_chunk(batch, vla, pp, ah, num_patches, "cuda",
                             abs_pose_projector=app)
    return pred, batch, np.asarray(s["actions"], dtype=np.float32)


def run(ds, tag: str, model_bits, cfg, fam_re=None, max_families=None) -> dict:
    processor = model_bits[0]
    collator = PaddedCollatorForActionPrediction_Nav_CCTV(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id,
        padding_side="right", num_img=cfg["model"]["num_images_in_input"])
    pairs = family_pairs(ds, fam_re, max_families)
    print(f"=== {tag}: {len(pairs)} twin PAIRS ===", flush=True)

    rows = []
    for (iA, iB) in pairs:
        for i, j in ((iA, iB), (iB, iA)):          # each twin, swapped with the other
            scene_i, scene_j = scene_of(ds, i), scene_of(ds, j)
            phrase_i = ds._phrase_by_scene[scene_i]
            phrase_j = ds._phrase_by_scene[scene_j]

            p_own, b_own, gt_i = predict_with_phrase(ds, i, None, collator, model_bits, cfg)
            p_swp, b_swp, _    = predict_with_phrase(ds, i, phrase_j, collator, model_bits, cfg)
            _,     _,    gt_j  = predict_with_phrase(ds, j, None, collator, model_bits, cfg)

            # The intervention must actually reach the model (lesson of 2026-07-21).
            # Phrases of different word-piece length give different seq lengths,
            # which is itself proof the prompt changed; -1 encodes that case.
            if b_own["input_ids"].shape == b_swp["input_ids"].shape:
                n_tok_diff = int((b_own["input_ids"] != b_swp["input_ids"]).sum())
            else:
                n_tok_diff = -1
            assert phrase_i == phrase_j or n_tok_diff != 0, (
                f"prompt tokens identical after swapping '{phrase_i}' -> '{phrase_j}'")

            own_lat, swp_lat = float(p_own[-1, 1]), float(p_swp[-1, 1])
            rows.append({
                "split": tag, "scene": scene_i, "twin": scene_j,
                "phrase_own": phrase_i, "phrase_swapped": phrase_j,
                "n_prompt_tokens_changed": n_tok_diff,
                "gt_lat_own": float(gt_i[-1, 1]),
                "gt_lat_twin": float(gt_j[-1, 1]),
                "pred_lat_own": own_lat,
                "pred_lat_swapped": swp_lat,
                "own_correct": bool(np.sign(own_lat) == np.sign(gt_i[-1, 1])),
                "flipped": bool(np.sign(own_lat) != np.sign(swp_lat)),
                "swap_correct": bool(np.sign(swp_lat) == np.sign(gt_j[-1, 1])),
                "twins_gt_opposite": bool(np.sign(gt_i[-1, 1]) != np.sign(gt_j[-1, 1])),
                "pred_delta": float(np.abs(p_own - p_swp).mean()),
            })
            r = rows[-1]
            print(f"  {scene_i[-9:]} '{phrase_i}' -> '{phrase_j}' | "
                  f"gt {r['gt_lat_own']:+.2f}/twin {r['gt_lat_twin']:+.2f} | "
                  f"pred own {own_lat:+.2f} swap {swp_lat:+.2f} | "
                  f"{'FLIP' if r['flipped'] else 'same'} "
                  f"{'swapOK' if r['swap_correct'] else 'swapMISS'} "
                  f"(dtok={n_tok_diff}, dpred={r['pred_delta']:.3f})", flush=True)
    return rows


def summarize(rows: list[dict], tag: str) -> dict:
    # Only pairs whose GT directions actually differ can show a meaningful flip.
    opp = [r for r in rows if r["twins_gt_opposite"]]
    s = {
        "n": len(rows),
        "n_twins_gt_opposite": len(opp),
        "own_accuracy": float(np.mean([r["own_correct"] for r in rows])) if rows else float("nan"),
        "flip_rate_all": float(np.mean([r["flipped"] for r in rows])) if rows else float("nan"),
        "flip_rate_opposite_gt": float(np.mean([r["flipped"] for r in opp])) if opp else float("nan"),
        "swap_accuracy_opposite_gt": float(np.mean([r["swap_correct"] for r in opp])) if opp else float("nan"),
        "mean_pred_delta": float(np.mean([r["pred_delta"] for r in rows])) if rows else float("nan"),
    }
    print(f"\n  [{tag}] n={s['n']} (GT-opposite twins: {s['n_twins_gt_opposite']})")
    print(f"    own accuracy          : {s['own_accuracy']:.3f}")
    print(f"    flip rate (opp-GT)    : {s['flip_rate_opposite_gt']:.3f}   <- 0 => language ignored")
    print(f"    swap accuracy (opp-GT): {s['swap_accuracy_opposite_gt']:.3f}   (chance 0.50)")
    print(f"    mean |pred change|    : {s['mean_pred_delta']:.4f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--fam-prefix", default="warehouse_2x3_lang",
                    help="scene family prefix; langv2 -> warehouse_2x3_langv2")
    ap.add_argument("--twins", default="AB", help="twin letters, e.g. AB or ABCD")
    ap.add_argument("--max-families", type=int, default=None,
                    help="cap families per split (train is large)")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--force-lang", action="store_true",
                    help="force modality 7 (language) for every probed sample. Needed on\n                         lang80-style configs: on a non-language sample the prompt is\n                         'No language instruction', so swapping the phrase is a no-op and\n                         the prompt-changed assertion (correctly) fires.")
    args = ap.parse_args()

    cfg = T2.load_yaml_config(Path(args.config))
    T2.register_omnivla_classes()
    print("=== building model ===", flush=True)
    model_bits = build_model(cfg, Path(args.ckpt), args.step, "cuda")
    processor = model_bits[0]
    seed = int(cfg["train"].get("seed", 42))

    fam_re = fam_regex(args.fam_prefix, args.twins)
    want = [t.strip() for t in args.splits.split(",") if t.strip()]
    out = {}
    for tag, key in (("train", "train_dir"), ("val", "val_dir")):
        if tag not in want:
            continue
        ds = build_split(cfg, processor, Path(cfg["data"][key]), seed)
        if args.force_lang:
            # Deterministic per-(traj,frame) reseeding means ds[i] is reproducible,
            # so restricting the modality set is a clean intervention: every probed
            # sample now carries a real instruction.
            ds.modality_list = [7]
            ds.modality_weights = [1.0]
            print(f"  [{tag}] forced modality 7 (language-only) for probing")
        rows = run(ds, tag, model_bits, cfg, fam_re, args.max_families)
        out[tag] = {"summary": summarize(rows, tag), "rows": rows}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"instruction_swap_step{args.step}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
