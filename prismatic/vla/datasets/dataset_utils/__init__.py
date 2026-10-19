"""Dataset utilities shared by IsaacSimDataset consumers.

    cf_windows    — counterfactual decision-window mining + the two ways to
                    consume it (index filter / sampler mask)
    onboard_swap  — counterfactual onboard-bundle swap builders

Split out of the retired counterfactual_dataset.py (2026-07-15): the
CounterfactualDataset subclass was replaced by DatasetManager views
(counterfactual_dataset_2.py) + apply_cf_window_filter; the free functions
live on here.
"""

from prismatic.vla.datasets.dataset_utils.cf_windows import (
    apply_cf_window_filter,
    cf_window_sample_mask,
    mine_cf_windows,
    print_pair_warnings,
)
from prismatic.vla.datasets.dataset_utils.onboard_swap import (
    build_onboard_swap_extra,
    build_onboard_swap_map,
)

__all__ = [
    "apply_cf_window_filter",
    "cf_window_sample_mask",
    "mine_cf_windows",
    "print_pair_warnings",
    "build_onboard_swap_extra",
    "build_onboard_swap_map",
]
