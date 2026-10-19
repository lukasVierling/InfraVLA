"""DatasetManager — v2 of the counterfactual machinery (design: Lukas, 2026-07-14).

One manager holds N ordinary IsaacSimDatasets, each with a sampling weight:

    normal      — the full trajectory set (weight 1)
    cf-windows  — the SAME trajectories, index cut to decision windows
                  (created from `normal` via create_counterfactual)
    recovery    — newly collected recovery trajectories (register_recovery;
                  collection happens OFFLINE — see design notes at EOF)

Trainer contract:
    mgr = DatasetManager()
    mgr.add_dataset("normal", weight=1.0, dataset_kwargs={...})
    mgr.create_counterfactual("normal", name="cf", weight=1.0,
                              pre_steps=16, post_steps=8, dev_thresh_m=0.75)
    mgr.register_recovery("recovery", traj_root=..., weight=0.5, like="normal")
    dataset, sample_w = mgr.build()
    # -> DataLoader(dataset, sampler=WeightedRandomSampler(sample_w, ...))

Weights are DATASET-MASS shares: {normal: 1, cf: 1} gives the two datasets
equal total sampling mass regardless of size, so per-sample weight is
w_d / len(d). Each dataset is sampled uniformly internally — reweighting is
between datasets only. weight 0.0 = derivation-only: the entry exists as a
parent for create_counterfactual/register_recovery but contributes no samples
to build() (e.g. window+recovery fine-tunes with no normal-data mass).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset

import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from prismatic.vla.datasets.cctv_isaac_sim_dataset import IsaacSimDataset  # noqa: E402
from prismatic.vla.datasets.dataset_utils import (  # noqa: E402
    mine_cf_windows, print_pair_warnings,
)


class DatasetManager:
    """
    Dataset Manager class creates a dataset manager that contains several datasets as a concat dataset. Each dataset can have weights (useful for upsampling counterfactuals)
    The dataset manager can add normal datasets or create counterfactual datasets based on provided hyperparameters 
    """

    def __init__(self, dataset_cls: type = IsaacSimDataset) -> None:
        """initialize empty dataset dict also stroe the class of datasets"""
        self.dataset_cls = dataset_cls
        self._entries: dict[str, dict] = {}

    def add_dataset(self, name: str, *, weight: float,
                    dataset_kwargs: dict) -> None:
        """Add a dataset to the dataset dict"""
        self._check_new_name(name)
        # weight 0.0 = DERIVATION-ONLY (2026-07-16): the entry is constructed
        # and usable as a parent (create_counterfactual / register_recovery /
        # sanity checks) but contributes no samples to build(). Negative is
        # still an error.
        if weight < 0.0:
            raise ValueError(f"dataset {name!r}: weight must be >= 0, got {weight}")
        #intantiate a new dataset
        dataset = self.dataset_cls(**dataset_kwargs)
        #add the dataset with its weight to the dictioanry
        self._entries[name] = dict(
            dataset=dataset, weight=float(weight),
            kwargs=dict(dataset_kwargs),
        )
        #print new dataset stat
        print(f"[DatasetManager] added {name!r}: {len(dataset)} samples, "
              f"weight={weight}")

    def create_counterfactual(self, source_name: str, *, name: str,
                              weight: float, pre_steps: int, post_steps: int,
                              dev_thresh_m: float,
                              criterion: str = "distance",
                              yaw_eps_deg: float = 1.0) -> dict:
        """
        Create a counterfactual dataset. Pulls the trajectories, then mines the counterfactual scenes in the data. 
        Creates a list of allowed samples to istantiate a new dataset. Lastly, adds the datset to the dictionary.

        Args:
            - source_name: name of the parent dataset in the dict
            - name: name of the new counterfactual dataset
            - weight: weight of the new dataset
            - pre_steps: how many steps before the curx will be added
            - post_steps: how many steps after the crux will be added
            - dev_thresh_m: threshold to consider to trajectories diverging
  # criterion selects how the crux is found — "distance" (twins
  # separate by dev_thresh_m) or "yaw" (last frame before the
  # twins' headings split by yaw_eps_deg). Must match the arm's
  # training-side window definition, or the fine-tune puts its
  # emphasis mass on a different frame set than the model saw.
            - criterion: "distance" | "yaw"
            - yaw_eps_deg: heading tolerance, only used when criterion="yaw"

        Returns:
            - mining: the mined counterfactual points
        """

        #get the parent dataset
        src = self._get(source_name)
        #get parent dataset trajectories
        trajs = src["dataset"].trajectories
        #mine the diverging points within the datset
        mining = mine_cf_windows(
            traj_names=[tr.traj_dir.name for tr in trajs],
            traj_positions=[tr.positions for tr in trajs],
            n_chunk=src["dataset"].num_actions_chunk,
            dev_thresh_m=float(dev_thresh_m),
            pre_steps=int(pre_steps),
            post_steps=int(post_steps),
            criterion=str(criterion),
            traj_yaws=[tr.yaws for tr in trajs],
            yaw_eps_deg=float(yaw_eps_deg),
        )
        # print warning for pairs 
        print_pair_warnings(mining["pair_log"], f"DatasetManager:{name}")
        # create allowed list for instantiating the child dataset mapping traj dir name to steps that are allowed
        allowed = {
            trajs[ti].traj_dir.name: steps
            for ti, steps in mining["windows_by_traj"].items() if steps
        }
        # if allowed empty then we got a problem
        if not allowed:
            raise RuntimeError(
                f"create_counterfactual({source_name!r}): no decision windows "
                "mined — is the source really a counterfactual (_cf) split?")
        # add the allowed list to the args and then create dataset
        cf_kwargs = dict(src["kwargs"])
        cf_kwargs["allowed_steps_by_name"] = allowed
        # add dataset
        self.add_dataset(name, weight=weight, dataset_kwargs=cf_kwargs)
        return mining

    def register_recovery(self, name: str, *, traj_root: Path | str,
                          weight: float, parent: str,
                          require_manifest: bool = True,
                          anchor_after_handoff: int | None = None,
                          anchor_rejoin: dict | None = None) -> None:
        """"
        Registers a recovery dataset. 
        Basically adds a dataset but prints the important parameters since recovery datasets are tied to a model. 
        The model is also tied to a Dataset manager.
        
        Args:
            - name: name of the datset
            - traj_root: where the trajectories are stored
            - weight: weight with which to add the dataset
            - parent: the dataset from which the recovery was generated and passes on hyperparams
            - require_manifest: checks the manifest and printsit 
        """
        #get the trajectories and the 
        src = self._get(parent)
        traj_root = Path(traj_root)
        manifest_path = traj_root / "recovery_manifest.json"
        #checks for manifesto
        if manifest_path.is_file():
            #load manifesto
            manifest = json.loads(manifest_path.read_text())
            print(f"[DatasetManager] {name!r} manifest: "
                  f"checkpoint={manifest.get('checkpoint', '?')} "
                  f"n_failures={manifest.get('n_failures', '?')} "
                  f"n_spawns={manifest.get('n_spawns', '?')}")
        elif require_manifest:
            raise FileNotFoundError(
                f"register_recovery({name!r}): no recovery_manifest.json under "
                f"{traj_root} — collect via the recovery pipeline, or pass "
                "require_manifest=False for a manually curated set.")
        #instantiate the new dataset
        rec_kwargs = dict(src["kwargs"])
        rec_kwargs["traj_root"] = traj_root
        # anchor_after_handoff (Lukas 2026-07-17): full-route recovery trajs
        # duplicate the source route after the corrective segment — restrict
        # sample ANCHORS to [0, handoff + t] (t ~ one cf-snippet length) so the
        # tail never generates its own samples. The tail STAYS in the files:
        # final_goal_prob=1.0 reads the goal from the final frame (= the
        # original goal), and label futures may extend past the anchor cut.
        if anchor_rejoin is not None:
            # Adaptive cut (Lukas 2026-07-17): anchors run until the recovery
            # is back within eps of the original plan (first touch — no sustain,
            # frames themselves = verified lock-in); minimum ONE recovery
            # anchor for spawns already on-path. No constant to defend — the
            # span is data-defined per trajectory.
            from isaac_sim.planning.a_star.goal_graph import GoalGraph
            from isaac_sim.utils.recovery_analysis import rejoin_index
            eps = float(anchor_rejoin["eps_m"])
            allowed = {}
            for d in sorted(Path(traj_root).glob("rec_*")):
                res = json.loads((d / "result.json").read_text())
                n_fail = res["stitched"]["n_failure_frames"]
                g = GoalGraph.load(d / "frames.json")
                rec_xy = [n.xy for n in g.nodes[n_fail:]]
                plan = json.loads((d / "astar_path.json").read_text())["path_xy"]
                k = rejoin_index(rec_xy, plan, eps_m=eps)
                span = max(k if k is not None else 0, 1)
                hi = min(n_fail + span, res["n_frames"] - 1)
                # RECOVERY-ONLY anchors (Lukas 2026-07-17): prefix anchors'
                # label chunks reproduce the model's own drift — the prefix
                # stays in the files as obs history only, never anchored.
                allowed[d.name] = set(range(n_fail, hi + 1))
            rec_kwargs["allowed_steps_by_name"] = allowed
        elif anchor_after_handoff is None:
            rec_kwargs["allowed_steps_by_name"] = None
        else:
            allowed = {}
            for d in sorted(Path(traj_root).glob("rec_*")):
                res = json.loads((d / "result.json").read_text())
                n_fail = res["stitched"]["n_failure_frames"]
                hi = min(n_fail + int(anchor_after_handoff), res["n_frames"] - 1)
                allowed[d.name] = set(range(0, hi + 1))
            rec_kwargs["allowed_steps_by_name"] = allowed
        self.add_dataset(name, weight=weight, dataset_kwargs=rec_kwargs)

    def build(self) -> tuple[ConcatDataset, np.ndarray]:
        """builds a concat datset form the datset manager"""
        if not self._entries:
            raise RuntimeError("DatasetManager.build(): no datasets registered")
        # parts and weights array — weight-0 entries are derivation-only and
        # contribute NO samples (see add_dataset).
        parts, weights = [], []
        # go through all the datasets
        for name, entry in self._entries.items():
            if entry["weight"] == 0.0:
                continue
            #get the dataset and the length (what is the length here?)
            ds, n = entry["dataset"], len(entry["dataset"])
            #get proper weights that sum up to n for the weighted samling
            per_sample = np.full(n, entry["weight"] / n, dtype=np.float64)
            # add dataset to the parts
            parts.append(ds)
            # add the weigth to weights
            weights.append(per_sample)
        if not parts:
            raise RuntimeError(
                "DatasetManager.build(): every entry has weight 0 — nothing "
                "to sample from")
        # createa concat dataset
        concat = ConcatDataset(parts)
        # get the sample weights
        sample_w = np.concatenate(weights)
        # get the total weigths and check the sum of weights
        total = sum(e["weight"] for e in self._entries.values())
        summary = " | ".join(
            (f"{name}: {len(e['dataset'])} samples, mass {e['weight'] / total:.2f}"
             if e["weight"] > 0.0 else f"{name}: derivation-only (weight 0)")
            for name, e in self._entries.items())
        print(f"[DatasetManager] build: {summary} | total {len(concat)} samples")
        return concat, sample_w

    def __contains__(self, name: str) -> bool:
        """Returns bool if dataset exists in the entries dict"""
        return name in self._entries

    def dataset(self, name: str) -> IsaacSimDataset:
        """ get a dataset for a given name"""
        return self._get(name)["dataset"]

    def _get(self, name: str) -> dict:
        """get the dataset from entries but first checks wehther it exsits"""
        if name not in self._entries:
            raise KeyError(
                f"no dataset {name!r} registered (have: {list(self._entries)})")
        return self._entries[name]

    def _check_new_name(self, name: str) -> None:
        """check whether is name is free to use."""
        if name in self._entries:
            raise ValueError(f"dataset {name!r} already registered")
