import numpy as np
import sys

sys.path.insert(0, '../')
from optimizer import OptimizerPass
sys.path.insert(0, '../..')
import hls_model
import templates

class BatchNormalizationBinaryTanh(hls_model.Layer):
    ''' Merged Batch Normalization and Binary Tanh layer.
        The mean, variance, beta, gamma parameters are folded into the threshold at which the 
        sign of the input flips after the Binary Tanh activation.
    '''

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims, precision='ap_uint<1>')

        original_name = self.attributes.get('original_name')
        variance = self.model.get_weights_data(original_name, 'moving_variance')
        mean = self.model.get_weights_data(original_name, 'moving_mean')
        gamma = self.model.get_weights_data(original_name, 'gamma')
        beta = self.model.get_weights_data(original_name, 'beta')
        epsilon = self.attributes.get('epsilon')
        threshold = mean - beta * np.sqrt(variance + epsilon) / gamma
        self.add_weights_variable(name='threshold', data=threshold, type_name='threshold{index}_t', precision=inp.precision)

    def function_cpp(self):
        params = self._default_function_params()
        params['threshold'] = self.get_weights('threshold').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        
        return self._config_template.format(**params)

batchnorm_binarytanh_config_template = """struct config{index} : nnet::batchnorm_binarytanh_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};\n"""

batchnorm_binarytanh_function_template = 'nnet::normalize_binary_tanh<{input_t}, {config}>({input}, {output}, {threshold});'

# Register the layer types to the layer map
hls_model.register_layer('BatchNormalizationBinaryTanh', BatchNormalizationBinaryTanh)

# Register the templates for config and function
templates.register_templates('BatchNormalizationBinaryTanh', batchnorm_binarytanh_function_template, batchnorm_binarytanh_config_template)

class MergeBatchNormAndBinaryTanh(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Activation'
            and node.get_attr('activation') == 'binary_tanh'
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
            'class_name' : 'BatchNormalizationBinaryTanh',
            'n_in' : bn_layer.get_attr('n_in'),
            'n_out' : bn_layer.get_attr('n_in'),
            'n_filt' : bn_layer.get_attr('n_filt'),
            'epsilon' : bn_layer.get_attr('epsilon')
        }
        bnbt_layer = model.make_node('BatchNormalizationBinaryTanh', 'bnbt_' + bn_layer.name, attrs, bn_layer.inputs)
        # Replace the old BatchNormalization layer with this one
        model.replace_node(bn_layer, bnbt_layer)

        return True

class QuantizeBinaryDenseOutput(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'BinaryDense'
            and node.get_input_node().__class__.__name__ == 'BatchNormalizationBinaryTanh')
        return is_match
    
    def transform(self, model, node):
        # Compute the required precision and update the variables
        # Number of bits for output is log2 of number of input nodes
        # Since this is the number of uint<1>'s which are summed
        nbits = int(np.ceil(np.log2(node.attributes['n_in'])) + 1)
        out_type = 'ap_int<{}>'.format(nbits)
        node.set_attr('accum_t', out_type)
        out_var = node.get_output_variable()
        out_var.precision = out_type
        node.precision[out_var.type] = out_type
        # If followed by the BatchNormalizationBinaryTanh, update its input
        bd_out_nodes = node.get_output_nodes()
        for out_node in bd_out_nodes:
            if out_node.__class__.__name__ == 'BatchNormalizationBinaryTanh':
                threshold_var = out_node.weights['threshold']
                threshold_var.precision = out_type
                out_node.precision[threshold_var.type] = out_type

        return False
