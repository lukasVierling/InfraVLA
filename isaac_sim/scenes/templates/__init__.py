"""Template registry. Each template module exposes a `Config` dataclass
(extending BaseTemplateConfig) and a `build(world, assets_root, config)`.

Only `warehouse_2x3` remains. The May-era testbeds (template1-5, random_open)
and the June wall_layout course were retired 2026-09-07 (cleanup batch 15) --
the thesis reports results on the 2x3 warehouse only. Recover any of them from
tag `pre-cleanup-2026-09-04`.
"""

from . import warehouse_2x3

REGISTRY = {
    "warehouse_2x3": warehouse_2x3,
}
