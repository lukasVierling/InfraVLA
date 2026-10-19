"""Template registry. Each template module exposes a `Config` dataclass
(extending BaseTemplateConfig) and a `build(world, assets_root, config)`.
Only `warehouse_2x3` is included in this release.
"""

from . import warehouse_2x3

REGISTRY = {
    "warehouse_2x3": warehouse_2x3,
}
