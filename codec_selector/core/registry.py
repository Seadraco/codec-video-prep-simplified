#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small plugin registry used by frontends and CLI adapters."""

from typing import Any, Callable, Dict, Iterable


class PluginRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        key = str(name).lower().strip()
        if not key:
            raise ValueError("plugin name cannot be empty")
        self._items[key] = fn
        return fn

    def get(self, name: str) -> Callable[..., Any]:
        key = str(name).lower().strip()
        if key not in self._items:
            available = ", ".join(sorted(self._items))
            raise KeyError(f"unknown plugin {name!r}; available: {available}")
        return self._items[key]

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._items))

    def as_dict(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._items)


samplers = PluginRegistry()
scorers = PluginRegistry()
groupers = PluginRegistry()
selectors = PluginRegistry()
packers = PluginRegistry()


def available_plugins() -> Dict[str, list[str]]:
    return {
        "samplers": list(samplers.names()),
        "scorers": list(scorers.names()),
        "groupers": list(groupers.names()),
        "selectors": list(selectors.names()),
        "packers": list(packers.names()),
    }
