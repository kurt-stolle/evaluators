"""
Defines ``Evaluator``-compatible classes that yield a dictionary of metrics upon evaluation.
"""

from __future__ import annotations

__all__ = ["generic", "vision"] # type: ignore

def __getattr__(name: str):
    import importlib
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return __all__