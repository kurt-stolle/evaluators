"""
Various scene understanding and perception evaluation metrics.
"""
from __future__ import annotations

from .depth import DepthEvaluator
from .dvps import DVPSEvaluator
from .panseg import PanSegEvaluator
from .step import STEPEvaluator

__version__ = "1.0.3"
