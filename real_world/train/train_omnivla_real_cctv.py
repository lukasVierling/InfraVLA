"""Real-Spot CCTV fine-tuning entrypoint - wraps the sim CCTV trainer.

Load the shared trainer
(`vla-scripts/train_omnivla_sim_cctv_v2.py`, the one every sim CCTV experiment
ran on) as a module and swap ONLY its dataset class for SpotRealCCTVDataset.
Nothing under vla-scripts/, prismatic/ or isaac_sim/ is modified, so the sim
experiments keep running byte-identically.

    cd <repo root> && conda activate omnivla
    python real_world/train/train_omnivla_real_cctv.py \
        --config real_world/train/configs/spot_cctv_cfboost.yaml \
        [--smoke-test 3 --disable-wandb] [--resume-step N]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location(
    "train_omnivla_sim_cctv_v2", _REPO_ROOT / "vla-scripts" / "train_omnivla_sim_cctv_v2.py"
)
_sim = importlib.util.module_from_spec(_spec)
sys.modules["train_omnivla_sim_cctv_v2"] = _sim
_spec.loader.exec_module(_sim)

from spot_real_cctv_dataset import SpotRealCCTVDataset  # noqa: E402

# The one intervention: every IsaacSimDataset(...) the trainer constructs
# (train/val/test/sanity) becomes a SpotRealCCTVDataset.
_sim.IsaacSimDataset = SpotRealCCTVDataset

if __name__ == "__main__":
    _sim.main()
