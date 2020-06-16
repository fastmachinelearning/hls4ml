from __future__ import absolute_import

from hls4ml.model.hls_model import HLSModel, HLSConfig

try:
    from hls4ml.model import profiling
    __profiling_enabled__ = True
except ImportError:
    __profiling_enabled__ = False
