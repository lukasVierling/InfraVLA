"""CCTV camera geometry.

`bev_mapper` (BEV projection) and `tools/` were retired 2026-09-07 (cleanup
batch 16): nothing outside this package imported them and the thesis reports no
BEV results. `camera_model` stays — viz/_lib/{env_panel_figure,cctv_coverage_tikz}.py
import it at module level to place the CCTV frusta in the thesis figures
(sub_layout, sub_coverage). Recover the rest from tag pre-cleanup-2026-09-04.
"""

from .camera_model import CameraModel, intrinsics_matrix, rotation_world_from_cv

__all__ = ["CameraModel", "intrinsics_matrix", "rotation_world_from_cv"]
