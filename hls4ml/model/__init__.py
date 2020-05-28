from __future__ import absolute_import

from .hls_model import HLSModel, HLSConfig

try:
    from . import profiling
    __profiling_enabled__ = True
except ImportError:
    __profiling_enabled__ = False
