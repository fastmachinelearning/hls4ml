"""Utilities for discovering and loading external hls4ml backend plugins."""

import os
from collections.abc import Callable
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

from hls4ml.backends.backend import register_backend
from hls4ml.writer.writers import register_writer

ENTRY_POINT_GROUP = 'hls4ml.backends'
ENV_PLUGIN_MODULES = 'HLS4ML_BACKEND_PLUGINS'

_plugins_loaded = False


def load_backend_plugins() -> None:
    """Discover and register backend plugins.

    This function loads plugins published via Python entry points under the
    ``hls4ml.backends`` group as well as modules listed in the
    ``HLS4ML_BACKEND_PLUGINS`` environment variable. The environment variable
    accepts a separator compatible with :data:`os.pathsep`.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return

    _load_entry_point_plugins()
    _load_env_plugins()

    _plugins_loaded = True


def _load_entry_point_plugins() -> None:
    group_eps = entry_points().select(group=ENTRY_POINT_GROUP)

    for ep in group_eps:
        try:
            obj = ep.load()
        except Exception as exc:
            print(f'WARNING: failed to load backend plugin entry "{ep.name}": {exc}')
            continue
        _register_plugin_object(ep.name, obj)


def _load_env_plugins() -> None:
    raw_modules = os.environ.get(ENV_PLUGIN_MODULES, '')
    if not raw_modules:
        return

    for module_name in filter(None, raw_modules.split(os.pathsep)):
        try:
            module = import_module(module_name)
        except Exception as exc:
            print(f'WARNING: failed to import backend plugin module "{module_name}": {exc}')
            continue

        register_callable: Any = getattr(module, 'register', module)
        _register_plugin_object(module_name, register_callable)


def _register_plugin_object(name: str, obj: Any) -> None:
    """Interpret the plugin object and register provided backends."""
    if callable(obj):
        _invoke_registration_callable(name, obj)
        return

    print(f'WARNING: plugin entry "{name}" did not provide a usable backend registration (got {obj!r})')


def _invoke_registration_callable(name: str, func: Callable[..., Any]) -> None:
    try:
        func(register_backend=register_backend, register_writer=register_writer)
        return
    except TypeError:
        try:
            func(register_backend, register_writer)
            return
        except Exception as exc:
            print(f'WARNING: backend plugin callable "{name}" failed: {exc}')
            return
    except Exception as exc:
        print(f'WARNING: backend plugin callable "{name}" failed: {exc}')
        return
