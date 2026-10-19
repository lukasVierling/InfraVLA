"""Isaac Sim environment generators for OmniVLA experiments.

Currently a single tunable warehouse env (`warehouse.py`) configured via JSON
presets in `configs/`. The registry pattern (and `flat.py`) was retired
when the parameterized warehouse landed — add new env types here only when
there's a concrete second one to support.
"""
