"""Codec-aware video preprocessing for training and inference."""

from .config import PreinferConfig

__all__ = ["PreinferConfig", "PreinferResult", "run_preinfer"]


def run_preinfer(*args, **kwargs):
    from .api import run_preinfer as _run_preinfer

    return _run_preinfer(*args, **kwargs)


def __getattr__(name):
    if name == "PreinferResult":
        from .api import PreinferResult

        return PreinferResult
    raise AttributeError(name)
