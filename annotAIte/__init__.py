"""
annotAIte: Self-supervised false-color mapping and harmonized clustering
for annotation assistance in microscopy and remote sensing.
"""

from .types import AnnotAIteResult, QuiltMetadata
from .pipeline import run_falsecolor_pipeline
from .multi_scale import run_multi_scale_falsecolor, create_parameter_grid, MultiScaleResult

__all__ = [
    "AnnotAIteResult",
    "QuiltMetadata",
    "run_falsecolor_pipeline",
    "run_multi_scale_falsecolor",
    "create_parameter_grid",
    "MultiScaleResult",
]

