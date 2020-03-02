from __future__ import absolute_import

from .keras_to_hls import keras_to_hls

try:
    from .pytorch_to_hls import pytorch_to_hls
    __pytorch_enabled__ = True
except ImportError:
    __pytorch_enabled__ = False

try:
    from .onnx_to_hls import onnx_to_hls
    __onnx_enabled__ = True
except ImportError:
    __onnx_enabled__ = False

try:
    from .tf_to_hls import tf_to_hls
    __tensorflow_enabled__ = True
except ImportError:
    __tensorflow_enabled__ = False


