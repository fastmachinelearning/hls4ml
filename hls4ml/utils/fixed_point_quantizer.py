from typing import Callable

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

# Nice figure (Figure. 2 and 3) from https://www.researchgate.net/publication/226964494_Formalization_of_Fixed-Point_Arithmetic_in_HOL to illustrate the rounding and saturation modes. # noqa: E501


def TRN(x):
    # Truncate towards negative infinity. Fast. Preferred when possible.
    return tf.floor(x)


def RND(x):
    # Round to nearest, ties to even.
    # Can be reduced to TRN with a bias.
    return tf.floor(x + 0.5)  # type:ignore


def RND_CONV(x):
    # towards nearest integer, ties to even.
    return tf.round(x)


def TRN_ZERO(x):
    # Truncate towards zero.
    sign = K.sign(x)
    return tf.floor(K.abs(x)) * sign


def RND_ZERO(x):
    # Round to nearest, ties to zero.
    sign = K.sign(x)
    return -tf.floor(-K.abs(x) + 0.5) * sign


def RND_MIN_INF(x):
    # Round to nearest, ties to negative infinity.
    return -tf.floor(-x + 0.5)  # type: ignore


def RND_INF(x):
    # Round to nearest, ties away from zero.
    sign = K.sign(x)
    return tf.floor(K.abs(x) + 0.5) * sign


def SAT(x, k, b):
    # Saturate between highest and lowest representable values.
    high = 2 ** (b - k) - 1
    low = -(high + 1) * k
    return tf.clip_by_value(x, low, high)


def SAT_ZERO(x, k, b):
    # Overflow to zero.
    high = 2 ** (b - k) - 1
    low = (-high - 1) * k
    mask = tf.cast((x <= high) & (x >= low), 'float32')
    return x * mask


def SAT_SYM(x, k, b):
    # Saturate between highest and lowest representable values when unsigned; between highest and -highest when signed.
    high = 2 ** (b - k) - 1
    low = -high * k
    return tf.clip_by_value(x, low, high)


def WRAP(x, k, b):
    # Wrap around.
    high = 2 ** (b - k) - 1
    low = -(high + 1) * k
    return tf.math.floormod(x - low, high - low + 1) + low


def WRAP_SYM(x, k, b):
    # High and low bounds are reflective.When overflows, can be less trash than WARP but still more trash than SAT. # noqa: E501
    dtype = x.dtype
    high = 2 ** (b - k) - 1
    low = -(high + 1) * k
    interval = (high - low + 1) * 2
    mapped = K.cast(tf.math.floormod(x - high - 1, interval), 'float32')
    return K.cast(K.abs(mapped - interval / 2 + 0.5) - 0.5 + low, dtype)


RND_MAP = {
    'RND': RND,
    'RND_ZERO': RND_ZERO,
    'RND_MIN_INF': RND_MIN_INF,
    'RND_INF': RND_INF,
    'RND_CONV': RND_CONV,
    'TRN_ZERO': TRN_ZERO,
    'TRN': TRN,
}

SAT_MAP = {
    'SAT': SAT,
    'SAT_ZERO': SAT_ZERO,
    'SAT_SYM': SAT_SYM,
    'WRAP': WRAP,
    'WRAP_SYM': WRAP_SYM,
}


@tf.function(autograph=False, jit_compile=True)
def gfixed_quantizer(x, keep_negative, bits, integer_bits, RND='TRN', SAT='WRAP'):
    '''Generalized fixed point quantizer, should have the same behavior to ap_fixed/ap_ufixed.
    Support high granularity quantization and broadcasting of bitwidths. RND and SAT mode must be strings.'''

    keep_negative = tf.cast(keep_negative, 'float32')
    bits = tf.cast(bits, 'float32')
    integer_bits = tf.cast(integer_bits, dtype='float32')

    two = tf.constant(2, dtype='float32')
    float_bits = bits - integer_bits  # type:ignore
    scale = tf.pow(two, float_bits)

    scaled_input = x * scale
    rnd, sat = RND_MAP[RND], SAT_MAP[SAT]
    quantized = sat(rnd(scaled_input), keep_negative, bits)
    return quantized / scale * tf.cast(bits != 0, 'float32')


def gfixed(keep_negative, bits, integer_bits, RND='TRN', SAT='WRAP') -> Callable:
    '''Functional form of generalized fixed point quantizer, should have the same behavior to ap_fixed/ap_ufixed.
    Support high granularity quantization and broadcasting of bitwidths. RND and SAT mode must be strings.'''

    def compute(x):
        return gfixed_quantizer(x, keep_negative, bits, integer_bits, RND, SAT)  # type:ignore

    return compute


def ufixed(bits, integer_bits, RND='TRN', SAT='WRAP') -> Callable:
    """Grammatical sugar for gfixed(0, bits, integer_bits, RND, SAT)."""
    return gfixed(0, bits, integer_bits, RND, SAT)


def fixed(bits, integer_bits, RND='TRN', SAT='WRAP') -> Callable:
    """Grammatical sugar for gfixed(1, bits, integer_bits, RND, SAT)."""
    return gfixed(1, bits, integer_bits, RND, SAT)


class FixedPointQuantizer(keras.layers.Layer):
    """Fixed point quantizer layer. This layer is not trainable. It is used as a proxy layer when converting a trained model into hls4ml readable form, and can also be used for bit-accurate hls4ml model emulation (up to fp32 representable precision).

    This class is not intended to be instantiated by users.

    Properties:
        - overrides: dict. Stores the precision overrides for layers. Currently only `overrides/layers/{layer_name}` field is used.
        - fusible: bool, property method. If True, this quantizer can be deleted and fused into the layer before it.
        - heterogeneous: bool, property method. If True, this quantizer has different bitwidths for different position.
        - result_t_kif: tuple of int. The (keep_negative, integer_bits, float_bits) of the quantized result.
        - keep_negative: tf.Variable. The keep_negative flag for each position.
        - bits: tf.Variable. The total bitwidth for each position.
        - integers: tf.Variable. The integer bitwidth for each position.
        - RND: str. The rounding mode. Only 'TRN' and 'RND' are fully tested.
        - SAT: str. The saturation mode. Only 'WRAP' and 'SAT' are fully tested.
    """  # noqa: E501

    def __init__(
        self,
        keep_negative,
        bits,
        integers,
        RND: str = 'TRN',
        SAT: str = 'WRAP',
        overrides: dict | None = None,
        accum_bits_bias=None,
        **kwargs,
    ):
        zeros = bits == 0
        keep_negative = tf.where(zeros, tf.zeros_like(keep_negative), keep_negative)
        integers = tf.where(zeros, tf.zeros_like(integers), integers)
        self.keep_negative = tf.Variable(keep_negative, dtype='int8', name='keep_negative', trainable=False)
        self.bits = tf.Variable(bits, dtype='int8', name='bits', trainable=False)
        self.integers = tf.Variable(integers, dtype='int8', name='integers', trainable=False)

        msg = f'Shapes mismatch: keep_negative, bits, and integers must have the same shape. Got {self.keep_negative.shape}, {self.bits.shape}, {self.integers.shape}.'  # noqa: E501
        assert self.keep_negative.shape == self.bits.shape == self.integers.shape, msg

        self.accum_bits_bias = accum_bits_bias
        self.RND = RND
        self.SAT = SAT

        self.overrides = overrides or {'layers': {}}
        kwargs.pop('trainable', None)
        self._quantizer_created = False

        super().__init__(trainable=False, **kwargs)

    def call(self, x):
        if not self.built:
            self.build(x.shape)
        return gfixed_quantizer(x, self.keep_negative, self.bits, self.integers, self.RND, self.SAT)  # type:ignore

    @property
    def result_t_kif(self):
        k, i, f = self.keep_negative, self.integers - self.keep_negative, self.bits - self.integers  # type:ignore
        k, i, f = np.max(k), np.max(i), np.max(f)  # type:ignore
        return k, i, f

    @property
    def fusible(self):
        """Delete this quantizer if no heterogeneity is detected."""
        assert (
            len(self._inbound_nodes) == 1
        ), 'FixedPointQuantizer must not be reused. Create proxy model only via proviced functions.'
        last_layer = self._inbound_nodes[0].inbound_layers
        assert not isinstance(
            last_layer, list
        ), f'FixedPointQuantizer has exactly one inbound layer. Got a list of {len(last_layer)} layers.'
        if len(last_layer._outbound_nodes) != 1:
            return False
        return not self.heterogeneous

    @property
    def heterogeneous(self):
        k0, b0, i0 = tf.reduce_max(self.keep_negative), tf.reduce_max(self.bits), tf.reduce_max(self.integers)
        if not tf.reduce_all(self.keep_negative == k0):
            return True
        if not tf.reduce_all(self.bits == b0):
            return True
        if not tf.reduce_all(self.integers == i0):
            return True
        return False

    def get_config(self):
        assert tf.reduce_all(
            (self.keep_negative == 0) | (self.keep_negative == 1)
        ), 'Illegal bitwidth config: keep_negative must be 0 or 1.'
        assert tf.reduce_all(self.bits >= 0), 'Illegal bitwidth config: bits must be non-negative.'  # type:ignore
        conf = super().get_config()
        conf['RND'] = self.RND
        conf['SAT'] = self.SAT
        conf['shape'] = tuple(self.bits.shape)
        overrides = self.overrides

        conf['overrides'] = overrides
        conf['fusible'] = self.fusible
        return conf

    @classmethod
    def from_config(cls, config: dict):
        dummy_v = np.full(config.pop('shape'), -128, dtype='int8')
        keep_negative = K.variable(dummy_v, dtype='int8', name='keep_negative')
        bits = K.variable(dummy_v, dtype='int8', name='bits')
        integers = K.variable(dummy_v, dtype='int8', name='integers')
        config.pop('fusible', None)
        return cls(keep_negative, bits, integers, **config)
