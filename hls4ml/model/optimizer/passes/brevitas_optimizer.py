# Conversion of model from channels_first to channels_last data format
# Based on https://github.com/fastmachinelearning/qonnx/blob/
# 12c96a3ded06beacab08e0f554e4ed014476c0aa/src/qonnx/transformation/channels_last.py
import math

import numpy as np

from hls4ml.model.layers import ApplyAlpha
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.optimizer.passes.quant_opt import _calculate_precision_quantizer
from hls4ml.model.types import NamedType, find_minimum_width


class BrevitasInputOutputOptimizer(OptimizerPass):
    '''Takes nodes parsed from brevitas and inserts Quant nodes into the model if necessary'''

    def match(self, node):
        needs_conversion = False
        if 'convert_io_from_brevitas' in node.attributes.keys():
            needs_conversion = node.attributes['convert_io_from_brevitas'] and (
                'output_quantization' in node.attributes.keys() or 'input_quantization' in node.attributes.keys()
            )

        return needs_conversion

    def transform(self, model, node):

        # See if Quant layer needs to be added for the output
        if 'output_quantization' in node.attributes.keys():

            attributes = {}

            input = node.name
            # Other attributes
            attributes['narrow'] = node.attributes['output_quantization']['narrow']
            attributes['rounding_mode'] = node.attributes['output_quantization']['rounding_mode']
            attributes['signed'] = node.attributes['output_quantization']['signed']
            attributes['bitwidth'] = node.attributes['output_quantization']['bit_width']
            attributes['zeropt'] = node.attributes['output_quantization']['zeropoint']
            attributes['scale'] = np.array([node.attributes['output_quantization']['scale']])

            quant_node = model.make_node('Quant', f'quant_output_for_{node.get_attr("name")}', attributes, [input])
            quant_node.set_attr('name', f'quant_output_for_{node.get_attr("name")}')

            model.insert_node(quant_node)

            node.attributes['convert_io_from_brevitas'] = False

        elif 'input_quantization' in node.attributes.keys():

            attributes = {}

            input = node.inputs[0]
            # Other attributes
            attributes['narrow'] = node.attributes['input_quantization']['narrow']
            attributes['rounding_mode'] = node.attributes['input_quantization']['rounding_mode']
            attributes['signed'] = node.attributes['input_quantization']['signed']
            attributes['bitwidth'] = node.attributes['input_quantization']['bit_width']
            attributes['zeropt'] = node.attributes['input_quantization']['zeropoint']
            attributes['scale'] = np.array([node.attributes['input_quantization']['scale']])

            quant_node = model.make_node('Quant', f'quant_input_for_{node.get_attr("name")}', attributes, [input])
            quant_node.set_attr('name', f'quant_input_for_{node.get_attr("name")}')

            model.insert_node(quant_node)

            node.attributes['convert_io_from_brevitas'] = False
        return True


class BrevitasFactorizeAlpha(OptimizerPass):
    '''OptimizerPass for extracting alpha "scale" from Brevitas quantized layer.
    The weights of the Quant{Dense, Conv} layer are scaled to the common data type,
    and an 'ApplyAlpha' layer is inserted to reapply the scale.
    '''

    def match(self, node):
        q_layer = node.class_name in ['Dense', 'QConv1D', 'Conv2D']

        has_w_alpha = 'weight_quantization' in node.attributes.keys()
        has_b_alpha = 'bias_quantization' in node.attributes.keys()

        needs_conversion = False
        if 'convert_from_brevitas' in node.attributes.keys():
            needs_conversion = node.attributes['convert_from_brevitas']

        is_match = q_layer and needs_conversion and (has_w_alpha or has_b_alpha)
        return is_match

    def transform(self, model, node):
        # The quantizer has to be applied to set the scale attribute
        # This must be applied to the _unquantized_ weights to obtain the correct scale
        if node.attributes['convert_from_brevitas'] is False:
            return False
        scale = np.full(node.weights['weight'].data.shape, [node.attributes['weight_quantization']['scale']])

        # find number of bits to represent unscaled weight tensor (should be the full bit width, but better be sure)
        # and set precision for weight variable
        int_bits = find_minimum_width(node.weights['weight'].data, signed=True)

        unscale_precision, _ = _calculate_precision_quantizer(int_bits, int_bits, True, True, 'FLOOR')
        node.weights['weight'].type = NamedType(node.weights['weight'].name + '_t', unscale_precision)
        res_precision, _ = _calculate_precision_quantizer(int_bits * 2, int_bits, True, True, 'FLOOR')
        node.types['accum_t'] = NamedType(node.name + '_accum_t', res_precision)
        node.types['result_t'].type = res_precision

        # Move the biases from the Dense layer to the ApplyAlpha layer
        bias = node.weights['bias'].data
        node.weights['bias'].data = np.zeros(bias.shape)

        # insert a Batch Normalization layer to apply the alpha scale
        if 'Linear' in node.class_name:
            n_in = node.get_attr('n_out')
        elif 'Conv' in node.class_name:
            n_in = node.get_attr('out_width') * node.get_attr('out_height', 1) * node.get_attr('n_filt')
        else:
            n_in = node.get_attr('n_out')

        # the name of the new ApplyAlpha node
        alpha_name = node.get_attr('name') + '_alpha'

        # make the precision auto
        alpha_precision = {'Precision': 'auto'}
        model.config.set_name_config(alpha_name, alpha_precision)
        model.config.parse_name_config(alpha_name, alpha_precision)

        # This part is very stupid, since this basically just results in the scale being represented at 2*bith width,
        # otherwise it just uses full system float precision. Needs work
        fractional_part, integer_part = math.modf(node.attributes['weight_quantization']['scale'])
        if integer_part > 0:
            int_bits = math.ceil(math.log2(integer_part)) + 1
        else:
            int_bits = 0
        frac_bits = math.ceil(math.log2(fractional_part * (10 ** len(str(fractional_part).split('.')[1]))))
        scale_precision, scale_quantizer = _calculate_precision_quantizer(
            int_bits + frac_bits, int_bits, True, False, 'FLOOR'
        )

        attrs = {
            'name': alpha_name,
            'class_name': 'Alpha',
            'inputs': node.outputs,
            'n_in': n_in,
            'n_filt': node.get_attr('n_filt', -1),
            'reuse_factor': node.get_attr('reuse_factor'),
            'scale_data': scale,
            'scale_quantizer': scale_quantizer,
            'scale_precision': scale_precision,
            'bias_data': bias,
            'bias_quantizer': None,
            'bias_precision': None,
            'trace': node.get_attr('trace', False),
        }
        alpha_layer = model.make_node(ApplyAlpha, node.name + '_alpha', attrs, node.outputs)
        model.insert_node(alpha_layer)
        node.attributes['convert_from_brevitas'] = False
        return True
