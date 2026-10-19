
from __future__ import annotations
import argparse, json, pathlib, re, sys
from collections import defaultdict

import numpy as np
import torch

_REPO = next(p for p in pathlib.Path(__file__).resolve().parents
                if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "viz/_lib")); sys.path.insert(0, str(_REPO / "vla-scripts"))
import cf_counterfactual_probe as P
import train_omnivla_sim_cctv_v2 as T2
from compare_hook_predictions import build_split
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_CCTV

OUT = _REPO / "isaac_sim" / "outputs" / "synonym_probe"
ZONE_CLASS = {"B": "left", "E": "right", "C": "straight", "F": "straight"}
# kind -> [canonical, synonym1, synonym2].  Canonical MUST match the training phrase.
SYNONYMS = {
    "forklift":     ["forklift",     "lift truck",  "fork truck"],
    "barrel":       ["barrel",       "drum",        "cask"],
    "box":          ["box",          "carton",      "package"],
    "traffic_cone": ["traffic cone", "safety cone", "pylon"],
}


def classify(lat, tau):
    return "left" if lat > tau else ("right" if lat < -tau else "straight")


def predict_lat(ds, idx, collator, mb):
    s = ds[idx]
    b = T2.pad_and_mask_batch(collator([s]), mb[0].tokenizer.pad_token_id,
                              mb[0].tokenizer.model_max_length)
    with torch.no_grad():
        pr = P.predict_chunk(b, *mb[1:5], "cuda", abs_pose_projector=mb[5])
    return float(pr[-1, 1]), float(np.asarray(s["actions"], dtype=np.float32)[-1, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--fam-prefix", default="warehouse_2x3_langv2")
    ap.add_argument("--manifest", default=str(
        _REPO / "isaac_sim/configs/experiment/datagen_2x3_langv2_all_manifest.json"))
    args = ap.parse_args()

    man = {m["name"]: m for m in json.loads(pathlib.Path(args.manifest).read_text())["scenes"]}
    rex = re.compile(rf"^({re.escape(args.fam_prefix)}_\d{{4}})_t([A-D])__pose__")
    cfg = T2.load_yaml_config(pathlib.Path(args.config)); T2.register_omnivla_classes()
    print("=== building model ===", flush=True)
    mb = P.build_model(cfg, pathlib.Path(args.ckpt), args.step, "cuda")
    collator = PaddedCollatorForActionPrediction_Nav_CCTV(
        mb[0].tokenizer.model_max_length, mb[0].tokenizer.pad_token_id,
        padding_side="right", num_img=cfg["model"]["num_images_in_input"])
    ds = build_split(cfg, mb[0], pathlib.Path(cfg["data"]["val_dir"]),
                     int(cfg["train"].get("seed", 42)))
    ds.modality_list = [7]; ds.modality_weights = [1.0]

    # decision-frame (min t) sample index per trajectory, + scene/kind/zone
    first_idx = {}
    for i, (ti, t) in enumerate(ds.samples_index):
        if ti not in first_idx or t < ds.samples_index[first_idx[ti]][1]:
            first_idx[ti] = i
    scenes = []  # (idx, scene, kind, true_class)
    for ti, i in first_idx.items():
        name = ds.trajectories[ti].traj_dir.name
        if not rex.match(name):
            continue
        scene = name.split("__pose__")[0]
        mm = man.get(scene, {}); z = mm.get("goal_zone")
        if z in ZONE_CLASS:
            scenes.append((i, scene, mm["target_kind"], ZONE_CLASS[z]))

    # tau from GT of the canonical run (measured, validated below)
    saved = dict(ds._phrase_by_scene)
    gts = {}
    for i, scene, kind, tc in scenes:
        ds._phrase_by_scene[scene] = SYNONYMS[kind][0]        # canonical
    for i, scene, kind, tc in scenes:
        _, gt = predict_lat(ds, i, collator, mb); gts[scene] = gt
    st = [abs(gts[s]) for _, s, _, tc in scenes if tc == "straight"]
    tn = [abs(gts[s]) for _, s, _, tc in scenes if tc != "straight"]
    tau = float((np.percentile(st, 90) + np.percentile(tn, 10)) / 2)
    gt_ok = np.mean([classify(gts[s], tau) == tc for _, s, _, tc in scenes])
    print(f"[tau={tau:.3f}] GT-rule reproduces zone label {gt_ok:.3f}\n")

    # per (kind, variant_index) accuracy on that kind's scenes
    results = defaultdict(dict)   # kind -> {variant_word: (correct, n)}
    for kind in SYNONYMS:
        kscenes = [(i, s, tc) for i, s, k, tc in scenes if k == kind]
        for vi, word in enumerate(SYNONYMS[kind]):
            ok = 0
            for i, s, tc in kscenes:
                ds._phrase_by_scene[s] = word                 # override THIS kind only
                pl, _ = predict_lat(ds, i, collator, mb)
                ok += (classify(pl, tau) == tc)
                ds._phrase_by_scene[s] = SYNONYMS[kind][0]     # restore
            results[kind][word] = (ok, len(kscenes))
    ds._phrase_by_scene = saved

    print(f"{'kind':>13} | {'canonical':>18} | {'synonym 1':>18} | {'synonym 2':>18}")
    tot = defaultdict(lambda: [0, 0])
    for kind in SYNONYMS:
        cells = []
        for vi, word in enumerate(SYNONYMS[kind]):
            ok, n = results[kind][word]
            cells.append(f"{word}={ok}/{n} ({ok/n:.2f})" if n else f"{word}=-")
            tot[vi][0] += ok; tot[vi][1] += n
        print(f"{kind:>13} | {cells[0]:>18} | {cells[1]:>18} | {cells[2]:>18}")
    print(f"{'ALL':>13} | " + " | ".join(
        f"{tot[vi][0]}/{tot[vi][1]} ({tot[vi][0]/tot[vi][1]:.3f})" for vi in (0, 1, 2)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"synonym_probe_step{args.step}.json").write_text(json.dumps(
        {"step": args.step, "tau": tau, "gt_rule": float(gt_ok),
         "results": {k: {w: results[k][w] for w in results[k]} for k in results}},
        indent=2))
    print(f"\nwrote {OUT}/synonym_probe_step{args.step}.json")


if __name__ == "__main__":
    main()
