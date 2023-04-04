from hls4ml.model.graph import HLSConfig, ModelGraph  # noqa: F401

try:
    from hls4ml.model import profiling  # noqa: F401

    __profiling_enabled__ = True
except ImportError:
    __profiling_enabled__ = False
