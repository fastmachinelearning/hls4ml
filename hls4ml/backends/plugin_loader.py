"""Utilities for discovering and loading external hls4ml backend plugins."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

try:  # pragma: no cover - fall back for older Python versions
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover
    from importlib_metadata import entry_points  # type: ignore

from hls4ml.backends.backend import Backend, register_backend
from hls4ml.writer.writers import register_writer

ENTRY_POINT_GROUP = 'hls4ml.backends'
ENV_PLUGIN_MODULES = 'HLS4ML_BACKEND_PLUGINS'

_plugins_loaded = False


def load_backend_plugins(logger: logging.Logger | None = None) -> None:
    """Discover and register backend plugins.

    This function loads plugins published via Python entry points under the
    ``hls4ml.backends`` group as well as modules listed in the
    ``HLS4ML_BACKEND_PLUGINS`` environment variable. The environment variable
    accepts a separator compatible with :data:`os.pathsep`.

    Args:
        logger (logging.Logger, optional): Optional logger used for diagnostics.
            When omitted, a module-local logger will be used.
    """

    global _plugins_loaded
    if _plugins_loaded:
        return

    logger = logger or logging.getLogger(__name__)

    _load_entry_point_plugins(logger)
    _load_env_plugins(logger)

    _plugins_loaded = True


def _load_entry_point_plugins(logger: logging.Logger) -> None:
    eps = entry_points()

    if hasattr(eps, 'select'):
        group_eps = eps.select(group=ENTRY_POINT_GROUP)
    else:  # pragma: no cover - legacy importlib_metadata API
        group_eps = eps.get(ENTRY_POINT_GROUP, [])

    for ep in group_eps:
        try:
            obj = ep.load()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                'Failed to load backend plugin entry %s: %s', ep.name, exc, exc_info=logger.isEnabledFor(logging.DEBUG)
            )
            continue
        _register_plugin_object(ep.name, obj, logger)


def _load_env_plugins(logger: logging.Logger) -> None:
    raw_modules = os.environ.get(ENV_PLUGIN_MODULES, '')
    if not raw_modules:
        return

    for module_name in filter(None, raw_modules.split(os.pathsep)):
        try:
            module = import_module(module_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                'Failed to import backend plugin module %s: %s',
                module_name,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            continue

        register_callable: Any = getattr(module, 'register', module)
        _register_plugin_object(module_name, register_callable, logger)


def _register_plugin_object(name: str, obj: Any, logger: logging.Logger) -> None:
    """Interpret the plugin object and register provided backends."""

    if inspect.isclass(obj) and issubclass(obj, Backend):
        _safe_register_backend(name, obj, logger)
        return

    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        for item in obj:
            _register_plugin_object(name, item, logger)
        return

    if callable(obj):
        _invoke_registration_callable(name, obj, logger)
        return

    logger.warning('Plugin entry %s did not provide a usable backend registration (got %r)', name, obj)


def _invoke_registration_callable(name: str, func: Callable[..., Any], logger: logging.Logger) -> None:
    try:
        func(register_backend=register_backend, register_writer=register_writer)
    except TypeError:
        try:
            func(register_backend, register_writer)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning('Backend plugin callable %s failed: %s', name, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        else:
            return
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning('Backend plugin callable %s failed: %s', name, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        return
    else:
        return


def _safe_register_backend(name: str, backend_cls: type[Backend], logger: logging.Logger) -> None:
    try:
        register_backend(name, backend_cls)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            'Failed to register backend %s from plugin: %s', name, exc, exc_info=logger.isEnabledFor(logging.DEBUG)
        )
