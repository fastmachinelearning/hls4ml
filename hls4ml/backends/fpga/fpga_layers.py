import numpy as np

from hls4ml.model.attributes import Attribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.layers import Conv1D, Conv2D, Layer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, XnorPrecisionType


class BatchNormalizationQuantizedTanh(Layer):
    """Merged Batch Normalization and quantized (binary or ternary) Tanh layer.
    The mean, variance, beta, gamma parameters are folded into the threshold(s) at which the
    sign of the input flips after the quantized (binary or ternary) Tanh activation.
    """

    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_filt', default=0),
        TypeAttribute('accum'),
        ConfigurableAttribute('reuse_factor', default=1),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        if self.get_attr('quantize') == 2:
            self.add_output_variable(shape, precision=self.binary_precision())
        elif self.get_attr('quantize') == 3:
            self.add_output_variable(shape, precision=IntegerPrecisionType(width=2))
        else:
            raise Exception(
                'Unsupported quantize attribute for BatchNormalizationQuantizedTanh: {}'.format(self.get_attr('quantize'))
            )

    def set_thresholds(self, scale, bias, ternary_threshold=0.5):
        inp = self.get_input_variable()
        shape = inp.shape
        precision = self.model.config.backend.convert_precision_string(inp.type.precision)
        F = precision.fractional
        low, high, threshold_precision = self.threshold_range(precision)
        # A negative scale reverses the comparisons with the thresholds, which the flip of the channel tells the kernel
        flip = scale < 0
        # A zero scale gives the same output for every input, which the bias decides
        zero = scale == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            threshold = -bias / scale
        if self.get_attr('quantize') == 2:
            self.add_output_variable(shape, precision=self.binary_precision())
            # The output is 1 where the input is at least the threshold (at most, if flipped), so the threshold is rounded
            # up (down) to the resolution of the input
            threshold = np.where(flip, np.floor(threshold * 2**F), np.ceil(threshold * 2**F)) / 2**F
            threshold = np.clip(np.where(zero, np.where(bias >= 0, low, high), threshold), low, high)
            self.add_weights_variable(
                name='threshold',
                var_name='t{index}',
                data=threshold,
                type_name='threshold{index}_t',
                precision=threshold_precision,
            )
        elif self.get_attr('quantize') == 3:
            self.add_output_variable(shape, precision=IntegerPrecisionType(width=2))
            with np.errstate(divide='ignore', invalid='ignore'):
                threshold_hi = ternary_threshold / scale + threshold
                threshold_lo = -ternary_threshold / scale + threshold
            # The output is 1 at or above the high threshold and -1 at or below the low one (at or below, and at or
            # above, if flipped). The kernel tests the high threshold with > (< if flipped), so it is moved to the last
            # step of the input below it (the first above it); the low one is rounded down (up)
            threshold_hi = np.where(flip, np.floor(threshold_hi * 2**F) + 1, np.ceil(threshold_hi * 2**F) - 1) / 2**F
            threshold_lo = np.where(flip, np.ceil(threshold_lo * 2**F), np.floor(threshold_lo * 2**F)) / 2**F
            threshold_hi = np.clip(np.where(zero, np.where(bias >= ternary_threshold, low, high), threshold_hi), low, high)
            threshold_lo = np.clip(np.where(zero, np.where(bias <= -ternary_threshold, high, low), threshold_lo), low, high)
            self.add_weights_variable(
                name='threshold_hi',
                var_name='th{index}',
                data=threshold_hi,
                type_name='threshold_hi_{index}_t',
                precision=threshold_precision,
            )
            self.add_weights_variable(
                name='threshold_lo',
                var_name='tl{index}',
                data=threshold_lo,
                type_name='threshold_lo_{index}_t',
                precision=threshold_precision,
            )
        self.add_weights_variable(
            name='flip',
            var_name='f{index}',
            data=flip.astype(int),
            type_name='flip{index}_t',
            precision=IntegerPrecisionType(width=1, signed=False),
        )

    def binary_precision(self):
        """The output type of a binary activation: 1-bit binary, where 0 stands for -1, or 0 and 1 with use_01."""

        return IntegerPrecisionType(width=1, signed=False) if self.get_attr('use_01', False) else XnorPrecisionType()

    def threshold_range(self, precision):
        """The thresholds one step beyond the range of an input of the given precision, at which a comparison is always
        or never true, and a type that holds them. Thresholds outside that range are moved to these."""

        step = 2.0**-precision.fractional
        low = (-(2.0 ** (precision.integer - 1)) if precision.signed else 0.0) - step
        high = 2.0 ** (precision.integer - precision.signed)
        integer = precision.integer + 2 - precision.signed
        return low, high, FixedPrecisionType(integer + precision.fractional, integer, signed=True)


class PointwiseConv1D(Conv1D):
    """Optimized Conv1D implementation for 1x1 kernels."""

    def initialize(self):
        # Do noting, values copied
        pass


class PointwiseConv2D(Conv2D):
    """Optimized Conv2D implementation for 1x1 kernels."""

    def initialize(self):
        # Do noting, values copied
        pass
