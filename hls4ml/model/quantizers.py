"""
This module contains definitions of hls4ml quantizer classes. These classes apply a quantization function on the
provided data. The quantization function may be defined locally or taken from a library in which case the classes
behave like simple wrappers.
"""

import numpy as np
import tensorflow as tf
from qkeras.quantizers import get_quantizer

from hls4ml.model.types import ExponentPrecisionType, FixedPrecisionType, IntegerPrecisionType, XnorPrecisionType


class Quantizer:
    """
    Base class for representing quantizers in hls4ml.

    Subclasses of ``Quantizer`` are expected to wrap the quantizers of upstream tools (e.g., QKeras).

    Args:
        bits (int): Total number of bits used by the quantizer.
        hls_type (NamedType): The hls4ml type used by the quantizer.
    """

    def __init__(self, bits, hls_type):
        self.bits = bits
        self.hls_type = hls_type

    def __call__(self, data):
        raise NotImplementedError


class BinaryQuantizer(Quantizer):
    """Quantizer that quantizes to 0 and 1 (``bits=1``) or -1 and 1 (``bits==2``).

    Args:
        bits (int, optional): Number of bits used by the quantizer. Defaults to 2.

    Raises:
        Exception: Raised if ``bits>2``
    """

    def __init__(self, bits=2):
        if bits == 1:
            hls_type = XnorPrecisionType()
        elif bits == 2:
            hls_type = IntegerPrecisionType(width=2)
        else:
            raise Exception(f'BinaryQuantizer suppots 1 or 2 bits, but called with bits={bits}')
        super().__init__(bits, hls_type)

    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        quant_data = data
        if self.bits == 1:
            quant_data = np.where(data > 0, ones, zeros).astype('int')
        if self.bits == 2:
            quant_data = np.where(data > 0, ones, -ones)
        return quant_data


class TernaryQuantizer(Quantizer):
    """Quantizer that quantizes to -1, 0 and 1."""

    def __init__(self):
        super().__init__(2, IntegerPrecisionType(width=2))

    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        return np.where(data > 0.5, ones, np.where(data <= -0.5, -ones, zeros))


class QKerasQuantizer(Quantizer):
    """Wrapper around QKeras quantizers.

    Args:
        config (dict): Config of the QKeras quantizer to wrap.
    """

    def __init__(self, config):
        self.quantizer_fn = get_quantizer(config)
        self.alpha = config['config'].get('alpha', None)
        if config['class_name'] == 'quantized_bits':
            self.bits = config['config']['bits']
            self.hls_type = self._get_type(config)
        # ! includes stochastic_ternary
        elif 'ternary' in config['class_name']:
            self.bits = 2
            self.hls_type = IntegerPrecisionType(width=2, signed=True)
        # ! includes stochastic_binary
        elif 'binary' in config['class_name']:
            self.bits = 1
            self.hls_type = XnorPrecisionType()
        else:
            print('Unsupported quantizer: ' + config['class_name'])
            self.bits = 16
            self.hls_type = FixedPrecisionType(width=16, integer=6, signed=True)

    def __call__(self, data):
        tf_data = tf.convert_to_tensor(data)
        return self.quantizer_fn(tf_data).numpy()
        # return self.quantizer_fn(data)

    def _get_type(self, quantizer_config):
        width = quantizer_config['config']['bits']
        integer = quantizer_config['config'].get('integer', 0)
        if quantizer_config['class_name'] == 'quantized_po2':
            return ExponentPrecisionType(width=width, signed=True)
        if width == integer:
            if width == 1:
                return XnorPrecisionType()
            else:
                return IntegerPrecisionType(width=width, signed=True)
        else:
            return FixedPrecisionType(width=width, integer=integer + 1, signed=True)


class QKerasBinaryQuantizer(Quantizer):
    """Wrapper around QKeras binary quantizer.

    Args:
        config (dict): Config of the QKeras quantizer to wrap.
    """

    def __init__(self, config, xnor=False):
        self.bits = 1 if xnor else 2
        self.hls_type = XnorPrecisionType() if xnor else IntegerPrecisionType(width=2, signed=True)
        self.alpha = config['config']['alpha']
        # Use the QKeras quantizer to handle any stochastic / alpha stuff
        self.quantizer_fn = get_quantizer(config)
        # Then we use our BinaryQuantizer to convert to '0,1' format
        self.binary_quantizer = BinaryQuantizer(1) if xnor else BinaryQuantizer(2)

    def __call__(self, data):
        x = tf.convert_to_tensor(data)
        y = self.quantizer_fn(x).numpy()
        return self.binary_quantizer(y)


class QKerasPO2Quantizer(Quantizer):
    """Wrapper around QKeras power-of-2 quantizers.

    Args:
        config (dict): Config of the QKeras quantizer to wrap.
    """

    def __init__(self, config):
        self.bits = config['config']['bits']
        self.quantizer_fn = get_quantizer(config)
        self.hls_type = ExponentPrecisionType(width=self.bits, signed=True)

    def __call__(self, data):
        # Weights are quantized to nearest power of two
        x = tf.convert_to_tensor(data)
        y = self.quantizer_fn(x)
        if hasattr(y, 'numpy'):
            y = y.numpy()
        return y
