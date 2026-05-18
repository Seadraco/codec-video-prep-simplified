"""Composable codec patch selector package."""

from .core.config import BitcostReadinessConfig

__all__ = ["BitcostReadinessConfig", "run_bitcost_readiness"]


def __getattr__(name: str):
    if name == "run_bitcost_readiness":
        from .core.pipeline import run_bitcost_readiness

        return run_bitcost_readiness
    raise AttributeError(name)
