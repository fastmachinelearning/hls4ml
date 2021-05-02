import numpy as np
import math
import os
from bisect import bisect_left
from queue import Queue
from collections.abc import Iterable
import re

from hls4ml.backends.backend import Backend
from hls4ml.model.hls_types import IntegerPrecisionType, FixedPrecisionType, XnorPrecisionType, ExponentPrecisionType


class FPGABackend(Backend):
    def __init__(self, name):
        super(FPGABackend, self).__init__(name)

    def compile(self, model):
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())

        lib_name = None
        try:
            ret_val = os.system('bash build_lib.sh')
            if ret_val != 0:
                raise Exception('Failed to compile project "{}"'.format(model.config.get_project_name()))
            lib_name = '{}/firmware/{}-{}.so'.format(model.config.get_output_dir(), model.config.get_project_name(), model.config.get_config_value('Stamp'))
        finally:
            os.chdir(curr_dir)
        
        return lib_name

    def get_valid_reuse_factors(self, layer):
        n_in = 0
        n_out = 0
        if 'Dense' in layer.__class__.__name__:
            n_in = layer.get_attr('n_in')
            n_out = layer.get_attr('n_out')
        elif 'Conv1D' in layer.__class__.__name__:
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')
        elif 'Conv2D' in layer.__class__.__name__:
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_height') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')

        max_rf = n_in * n_out
        valid_reuse_factors = []
        for rf in range(1, max_rf + 1):
            _assert = self._check_conditions(n_in, n_out, rf)
            if _assert:
                valid_reuse_factors.append(rf)
        # Avoid using RF=1
        if valid_reuse_factors[0] == 1:
            valid_reuse_factors.pop(0)
        return valid_reuse_factors

    def _check_conditions(self, n_in, n_out, rf):
        multfactor = min(n_in, rf)
        multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
        #
        # THIS ASSERTION IS FOR THE FUNCTIONAL CORRECTNESS OF THE DENSE LAYER
        #
        _assert = (((multiplier_limit % n_out) == 0) or (rf >= n_in))
        _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
        #
        # THIS ASSERTION IS FOR QoR AND EXECUTION TIME
        #
        _assert = _assert and (((n_in * n_out) % rf) == 0)

        return _assert

    def get_closest_reuse_factor(self, valid_rf, chosen_rf):
        """
        Returns closest value to chosen_rf. valid_rf is sorted (obtained from get_valid_reuse_factors())
        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(valid_rf, chosen_rf)
        if pos == 0:
            return valid_rf[0]
        if pos == len(valid_rf):
            return valid_rf[-1]
        before = valid_rf[pos - 1]
        after = valid_rf[pos]
        if after - chosen_rf < chosen_rf - before:
            return after
        else:
            return before

    def set_closest_reuse_factor(self, layer):
        valid_rf = self.get_valid_reuse_factors(layer)
        chosen_rf = layer.reuse_factor
        if chosen_rf not in valid_rf:
            closest_rf = self.get_closest_reuse_factor(valid_rf, chosen_rf)
            print('WARNING: Invalid ReuseFactor={} in layer "{}". Using ReuseFactor={} instead. Valid ReuseFactor(s): {}.'
                .format(chosen_rf, layer.name, closest_rf, ','.join(map(str, valid_rf))))
            layer.reuse_factor = closest_rf

    def convert_precision_string(self, precision):
        if isinstance(precision, IntegerPrecisionType) or isinstance(precision, FixedPrecisionType):
            return precision

        if precision.startswith('ap_'):
            return self._convert_ap_type(precision)
        elif precision.startswith('ac_'):
            return self._convert_ac_type(precision)
        else:
            raise Exception('Cannot convert precision string: {}'.format(precision))

    def _convert_ap_type(self, precision):
        '''
        Convert a precision string (e.g. "ap_fixed<16,6>" to the internal FixedPrecisionTypes etc)
        '''
        bits = re.search('.+<(.+?)>', precision).group(1).split(',')
        sat_mode = None
        round_mode = None
        sat_bits = None
        if 'fixed' in precision:
            W = int(bits[0])
            I = int(bits[1])
            fields = 2
            signed = ~('u' in precision)
        elif 'int' in precision:
            W = int(bits[0])
            I = W
            fields = 1
            signed = ~('u' in precision)
        if len(bits) > fields:
            sat_mode = bits[fields]
        if len(bits) > fields+1:
            round_mode = bits[fields+1]
        if len(bits) > fields+2:
            sat_bits = int(bits[fields+2])
        if 'fixed' in precision:
            return FixedPrecisionType(W, I, signed, round_mode, sat_mode, sat_bits)
        elif 'int' in precision:
            return IntegerPrecisionType(W, signed)

    def _convert_ac_type(self, precision):
        '''
        Convert a precision string (e.g. "ac_fixed<16,6>" to the internal FixedPrecisionTypes etc)
        '''
        bits = re.search('.+<(.+?)>', precision).group(1).split(',')
        signed = True  # default is signed
        sat_mode = None
        round_mode = None
        if 'fixed' in precision:
            W = int(bits[0])
            I = int(bits[1])
            fields = 2
            if len(bits) > 2:
                signed = bool(bits[2])
                fields = 3
        elif 'int' in precision:
            W = int(bits[0])
            I = W
            fields = 1
            if len(bits) > 1:
                signed = bool(bits[1])
                fields = 2
        if len(bits) > fields:
            sat_mode = bits[fields]
        if len(bits) > fields+1:
            round_mode = bits[fields+1]
        if 'fixed' in precision:
            return FixedPrecisionType(W, I, signed, round_mode, sat_mode)
        elif 'int' in precision:
            return IntegerPrecisionType(W, signed)

    def product_type(self, data_T, weight_T):
        '''
        Helper function to determine which product implementation to use during inference
        '''
        assert not isinstance(data_T, ExponentPrecisionType), "Only ExponentPrecisionType (aka 'power of 2') weights are currently supported, not data."
        product = 'mult'
        if isinstance(weight_T, ExponentPrecisionType):
            product = 'weight_exponential'
        else:
            # if binary
            if isinstance(weight_T, XnorPrecisionType) and isinstance(data_T, XnorPrecisionType):
                product = 'both_binary'
            elif isinstance(weight_T, XnorPrecisionType): # data is not xnor-binary
                product = 'weight_binary'
            elif isinstance(weight_T, IntegerPrecisionType) and weight_T.width == 2 and weight_T.signed:
                product = 'weight_ternary'
            else:
                product = 'mult'
        return product

    def compute_conv1d_instructions(self, in_W, in_C, kernel_size=3, stride=1, pad=0):

        # Current limitations
        assert pad == 0

        min_W = (math.ceil(kernel_size / stride) - 1) * stride + kernel_size
        min_oW = int((min_W - kernel_size) // stride + 1)

        out_W = int((in_W - kernel_size) // stride + 1)
        scaled_W = (out_W - 1) * stride + kernel_size

        if scaled_W < in_W:
            min_W += 1

        windows_bin = [[0 for _ in range(kernel_size)] for _ in range(min_W)]

        for i_ow in range(min_oW):
            for i_fw in range(kernel_size):
                index_data = i_ow * stride + i_fw - pad
                windows_bin[index_data][i_fw] = 1

        windows_int = []

        for i in range(min_W):
            windows_int.append((int(''.join(str(p) for p in reversed(windows_bin[i])), 2)))

        return (min_W, windows_int)

    def compute_conv2d_instructions(self, in_H, in_W, in_C, kernel_size=3, stride=1, pad=0):

        if isinstance(kernel_size, Iterable):
            kernel_height = kernel_size[0]
            kernel_width = kernel_size[1]
        else:
            kernel_height = kernel_size
            kernel_width = kernel_size

        if isinstance(stride, Iterable):
            stride_height = stride[0]
            stride_width = stride[1]
        else:
            stride_height = stride
            stride_width = stride

        # Current limitations
        assert kernel_height == kernel_width
        assert stride_height == stride_width
        assert pad == 0

        min_H = (math.ceil(kernel_height / stride_height) - 1) * stride_height + kernel_height
        min_W = (math.ceil(kernel_width / stride_width) - 1) * stride_width + kernel_width
        min_oH = int((min_H - kernel_height) // stride_height + 1)
        min_oW = int((min_W - kernel_width) // stride_width + 1)

        out_H = int((in_H - kernel_height) // stride_height + 1)
        out_W = int((in_W - kernel_width) // stride_width + 1)
        scaled_H = (out_H - 1) * stride_height + kernel_height
        scaled_W = (out_W - 1) * stride_width + kernel_width

        if scaled_H < in_H:
            min_H += 1
        if scaled_W < in_W:
            min_W += 1

        # Let's hardcode a few common cases:
        if kernel_height == 1 and kernel_width == 1 and stride == 1 and scaled_H == in_H and scaled_W == in_W:
            return (1, 1, map(str, [1]))
        if kernel_height == 3 and kernel_width == 3 and stride == 1 and scaled_H == in_H and scaled_W == in_W:
            return (5, 5, map(str, [1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256]))
        if kernel_height == 5 and kernel_width == 5 and stride == 1 and scaled_H == in_H and scaled_W == in_W:
            return (9, 9, map(str, [1,3,7,15,31,30,28,24,16,33,99,231,495,1023,990,924,792,528,1057,3171,7399,15855,
                             32767,31710,29596,25368,16912,33825,101475,236775,507375,1048575,1014750,947100,
                             811800,541200,1082401,3247203,7576807,16236015,33554431,32472030,30307228,25977624,
                             17318416,1082400,3247200,7576800,16236000,33554400,32472000,30307200,25977600,17318400,
                             1082368,3247104,7576576,16235520,33553408,32471040,30306304,25976832,17317888,1081344,
                             3244032,7569408,16220160,33521664,32440320,30277632,25952256,17301504,1048576,3145728,
                             7340032,15728640,32505856,31457280,29360128,25165824,16777216]))

        windows_bin = [[0 for _ in range(kernel_height * kernel_width)] for _ in range(min_H * min_W)]

        for i_oh in range(min_oH):
            for i_ow in range(min_oW):
                for i_fh in range(kernel_height):
                    for i_fw in range(kernel_width):
                        index_data = (i_oh * stride_height + i_fh - pad) * min_W + (i_ow * stride_width + i_fw - pad)
                        windows_bin[index_data][i_fh * kernel_width + i_fw] = 1

        windows_int = []

        for i in range(min_H):
            for j in range(min_W):
                windows_int.append((int(''.join(str(p) for p in reversed(windows_bin[i * min_W + j])), 2)))

        return (min_H, min_W, windows_int)
