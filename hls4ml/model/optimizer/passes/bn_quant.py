import numpy as np
import re

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.hls_model import Layer, IntegerPrecisionType, register_layer
from hls4ml.templates import templates

class BatchNormalizationQuantizedTanh(Layer):
    ''' Merged Batch Normalization and quantized (binary or ternary) Tanh layer.
        The mean, variance, beta, gamma parameters are folded into the threshold(s) at which the
        sign of the input flips after the quantized (binary or ternary) Tanh activation.
    '''

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        precision_bits = re.search('.+<(.+?)>', inp.type.precision).group(1).split(',')
        if 'int' in str(inp.type.precision):
            W = int(precision_bits[0])
            I = W
            F = 0
        elif 'fixed' in str(inp.type.precision):
            W = int(precision_bits[0])
            I = int(precision_bits[1])
            F = W - I
        original_name = self.attributes.get('original_name')
        variance = self.model.get_weights_data(original_name, 'moving_variance')
        mean = self.model.get_weights_data(original_name, 'moving_mean')
        gamma = self.model.get_weights_data(original_name, 'gamma')
        beta = self.model.get_weights_data(original_name, 'beta')
        epsilon = self.attributes.get('epsilon')
        threshold = mean - beta * np.sqrt(variance + epsilon) / gamma
        if self.get_attr('quantize') == 2:
            self.add_output_variable(shape, dims, precision=IntegerPrecisionType(width=1, signed=False))
            threshold = np.floor(threshold * 2**F) / 2**F
            self.add_weights_variable(name='threshold', var_name='t{index}', data=threshold, type_name='threshold{index}_t', precision=inp.type.precision)
        elif self.get_attr('quantize') == 3:
            self.add_output_variable(shape, dims, precision=IntegerPrecisionType(width=2))
            threshold_hi = 0.5/(gamma/np.sqrt(variance + epsilon)) + threshold
            threshold_lo = -0.5/(gamma/np.sqrt(variance + epsilon)) + threshold
            threshold_hi = np.floor(threshold_hi * 2**F) / 2**F
            threshold_lo = np.floor(threshold_lo * 2**F) / 2**F
            self.add_weights_variable(name='threshold_hi', var_name='th{index}', data=threshold_hi, type_name='threshold_hi_{index}_t', precision=inp.type.precision)
            self.add_weights_variable(name='threshold_lo', var_name='tl{index}', data=threshold_lo, type_name='threshold_lo_{index}_t', precision=inp.type.precision)

    def function_cpp(self):
        params = self._default_function_params()
        if self.get_attr('quantize') == 2:
            params['quantize'] = 'binary'
            params['threshold'] = self.get_weights('threshold').name
        elif self.get_attr('quantize') == 3:
            params['quantize'] = 'ternary'
            params['threshold'] = self.get_weights('threshold_hi').name + ', ' + self.get_weights('threshold_lo').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()

        return self._config_template.format(**params)

batchnorm_quantized_tanh_config_template = """struct config{index} : nnet::batchnorm_quantized_tanh_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};\n"""

batchnorm_quantized_tanh_function_template = 'nnet::normalize_{quantize}_tanh<{input_t}, {config}>({input}, {output}, {threshold});'

# Register the layer types to the layer map
register_layer('BatchNormalizationQuantizedTanh', BatchNormalizationQuantizedTanh)

# Register the templates for config and function
templates.get_backend('Vivado').register_templates('BatchNormalizationQuantizedTanh', batchnorm_quantized_tanh_function_template, batchnorm_quantized_tanh_config_template)

class MergeBatchNormAndQuantizedTanh(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Activation'
            and node.get_attr('activation') in ['binary_tanh', 'ternary_tanh']
            and node.get_input_node().__class__.__name__ == 'BatchNormalization')
        return is_match

    def transform(self, model, node):
        bn_layer = node.get_input_node()
        # Remove the Activation layer
        model.remove_node(node, rewire=True)
        # Make a new layer with the new attributes
        quantize = 0
        if node.get_attr('activation') == 'binary_tanh':
            quantize = 2
        if node.get_attr('activation') == 'ternary_tanh':
            quantize = 3
        attrs = {
            'name' : bn_layer.get_attr('name'),
            'original_name' : bn_layer.get_attr('name'),
            'class_name' : 'BatchNormalizationQuantizedTanh',
            'n_in' : bn_layer.get_attr('n_in'),
            'n_out' : bn_layer.get_attr('n_in'),
            'n_filt' : bn_layer.get_attr('n_filt'),
            'epsilon' : bn_layer.get_attr('epsilon'),
            'quantize' : quantize
        }
        bnbt_layer = model.make_node('BatchNormalizationQuantizedTanh', 'bnbt_' + bn_layer.name, attrs, bn_layer.inputs)
        # Replace the old BatchNormalization layer with this one
        model.replace_node(bn_layer, bnbt_layer)

        return True

class QuantizeDenseOutput(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Dense' and node.get_attr('weight_quantizer') is not None
            and node.get_input_node().__class__.__name__ == 'BatchNormalizationQuantizedTanh')
        return is_match

    def transform(self, model, node):
        # Compute the required precision and update the variables
        # Number of bits for output is log2 of number of input nodes
        # Since this is the number of uint<1>'s which are summed
        nbits = int(np.ceil(np.log2(node.attributes['n_in'])) + 2)
        out_type = IntegerPrecisionType(width=nbits)
        node.set_attr('accum_t', out_type)
        out_var = node.get_output_variable()
        out_var.type.precision = out_type

        quantized_data = None
        quantized_precision = None
        quantizer = node.get_attr('weight_quantizer')
        if quantizer.__class__.__name__ == 'BinaryQuantizer':
            quantized_precision = IntegerPrecisionType(width=1, signed=False)
        elif quantizer.__class__.__name__ == 'TernaryQuantizer':
            quantized_precision = IntegerPrecisionType(width=2)
        else:
            print('WARNING: Unknown quantizer - {}. Bailing out'.format(quantizer.__class__.__name__))
            return False
        quantizer.bits = quantized_precision.width
        quantizer.hls_type = quantized_precision
        quantized_data = quantizer(node.weights['weight'].data)

        weights = node.weights['weight']
        weights.data = quantized_data
        weights.type.name = 'weight{index}_t'.format(index=node.index)
        weights.update_precision(quantized_precision)

        bias = node.weights['bias']
        bias.data = np.zeros(shape=(node.get_attr('n_out')))
        bias.type.name = 'bias{index}_t'.format(index=node.index)
        bias.nzeros = 0
        bias.update_precision(quantized_precision)

        # If followed by the BatchNormalizationBinaryTanh, update its input
        # Also requantise the weights
        bd_out_nodes = node.get_output_nodes()
        for out_node in bd_out_nodes:
            if out_node.__class__.__name__ == 'BatchNormalizationQuantizedTanh':
                var_names = []
                if quantizer.__class__.__name__ == 'BinaryQuantizer':
                    var_names.append('threshold')
                elif quantizer.__class__.__name__ == 'TernaryQuantizer':
                    var_names.append('threshold_hi')
                    var_names.append('threshold_lo')
                for var_name in var_names:
                    threshold_var = out_node.weights[var_name]
                    threshold_var.update_precision(out_type)
                    threshold_var.data = np.floor(threshold_var.data)

        return False
