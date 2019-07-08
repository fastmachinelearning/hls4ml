import numpy as np
import sys
import re

sys.path.insert(0, '../')
from optimizer import OptimizerPass
sys.path.insert(0, '../..')
import hls_model
import templates

class BatchNormalizationTernaryTanh(hls_model.Layer):
    ''' Merged Batch Normalization and Ternary Tanh layer.
        The mean, variance, beta, gamma parameters are folded into two thresholds (threshold - 0.5/+0.5) at which the 
        sign of the input flips after the Ternary Tanh activation.
    '''

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims, precision='ap_int<2>')
        precision_bits = re.search('.+<(.+?)>', inp.precision).group(1).split(',')
        if 'ap_int' in inp.precision:
          W = int(precision_bits[0])
          I = W
          F = 0
        elif 'ap_fixed' in inp.precision:
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
        threshold_hi = 0.5/(gamma/np.sqrt(variance + epsilon)) + threshold
        threshold_lo = -0.5/(gamma/np.sqrt(variance + epsilon)) + threshold
        threshold_hi = np.floor(threshold_hi * 2**F) / 2**F
        threshold_lo = np.floor(threshold_lo * 2**F) / 2**F
        self.add_weights_variable(name='threshold_hi_', data=threshold_hi, type_name='threshold_hi_{index}_t', precision=inp.precision)
        self.add_weights_variable(name='threshold_lo_', data=threshold_lo, type_name='threshold_lo_{index}_t', precision=inp.precision)

    def function_cpp(self):
        params = self._default_function_params()
        params['threshold_hi_'] = self.get_weights('threshold_hi_').name
        params['threshold_lo_'] = self.get_weights('threshold_lo_').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        
        return self._config_template.format(**params)

batchnorm_ternarytanh_config_template = """struct config{index} : nnet::batchnorm_ternarytanh_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};\n"""

batchnorm_ternarytanh_function_template = 'nnet::normalize_ternary_tanh<{input_t}, {config}>({input}, {output}, {threshold_hi_}, {threshold_lo_});'

# Register the layer types to the layer map
hls_model.register_layer('BatchNormalizationTernaryTanh', BatchNormalizationTernaryTanh)

# Register the templates for config and function
templates.register_templates('BatchNormalizationTernaryTanh', batchnorm_ternarytanh_function_template, batchnorm_ternarytanh_config_template)

class MergeBatchNormAndTernaryTanh(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Activation'
            and node.get_attr('activation') == 'ternary_tanh'
            and node.get_input_node().__class__.__name__ == 'BatchNormalization')
        return is_match
    
    def transform(self, model, node):
        bn_layer = node.get_input_node()
        # Remove the Activation layer
        model.remove_node(node, rewire=True)
        # Make a new layer with the new attributes
        attrs = {
            'name' : bn_layer.get_attr('name'),
            'original_name' : bn_layer.get_attr('name'),
            'class_name' : 'BatchNormalizationTernaryTanh',
            'n_in' : bn_layer.get_attr('n_in'),
            'n_out' : bn_layer.get_attr('n_in'),
            'n_filt' : bn_layer.get_attr('n_filt'),
            'epsilon' : bn_layer.get_attr('epsilon')
        }
        bnbt_layer = model.make_node('BatchNormalizationTernaryTanh', 'bnbt_' + bn_layer.name, attrs, bn_layer.inputs)
        # Replace the old BatchNormalization layer with this one
        model.replace_node(bn_layer, bnbt_layer)

        return True

class QuantizeTernaryDenseOutput(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Dense' and node.get_attr('quantize') == 3
            and node.get_input_node().__class__.__name__ == 'BatchNormalizationTernaryTanh')
        return is_match
    
    def transform(self, model, node):
        # Compute the required precision and update the variables
        # Number of bits for output is log2 of number of input nodes
        # Since this is the number of uint<1>'s which are summed
        nbits = int(np.ceil(np.log2(node.attributes['n_in'])) + 2)
        out_type = 'ap_int<{}>'.format(nbits)
        node.set_attr('accum_t', out_type)
        out_var = node.get_output_variable()
        out_var.precision = out_type
        node.precision[out_var.type] = out_type
        
        data_1bit = model.quantize_data(node.weights['weight'].data, 3)
        weights = node.weights['weight']
        weights.data = data_1bit
        weights.type = 'weight{index}_t'.format(index=node.index)
        weights.precision = 'ap_int<2>'
        node.precision[weights.type] = weights.precision
        zeros = np.zeros(shape=(node.get_attr('n_out')))
        bias = node.weights['bias']
        bias.data = zeros
        bias.type = 'bias{index}_t'.format(index=node.index)
        bias.nzeros = 0
        bias.precision = 'ap_int<2>'
        node.precision[bias.type] = bias.precision
        
        # If followed by the BatchNormalizationBinaryTanh, update its input
        # Also requantise the weights
        bd_out_nodes = node.get_output_nodes()
        for out_node in bd_out_nodes:
            if out_node.__class__.__name__ == 'BatchNormalizationTernaryTanh':
                threshold_hi_var = out_node.weights['threshold_hi_']
                threshold_hi_var.precision = out_type
                threshold_hi_var.data = np.floor(threshold_hi_var.data)
                out_node.precision[threshold_hi_var.type] = out_type
                threshold_lo_var = out_node.weights['threshold_lo_']
                threshold_lo_var.precision = out_type
                threshold_lo_var.data = np.floor(threshold_lo_var.data)
                out_node.precision[threshold_lo_var.type] = out_type

        return False
