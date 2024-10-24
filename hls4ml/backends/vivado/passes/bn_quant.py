import numpy as np

from hls4ml.backends.fpga.fpga_layers import BatchNormalizationQuantizedTanh
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import BatchNormalization, register_layer
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import IntegerPrecisionType, NamedType, XnorPrecisionType

batchnorm_quantized_tanh_config_template = """struct config{index} : nnet::batchnorm_quantized_tanh_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};\n"""

batchnorm_quantized_tanh_function_template = (
    'nnet::normalize_{quantize}_tanh<{input_t}, {config}>({input}, {output}, {threshold});'
)

bn_include_list = ['nnet_utils/nnet_batchnorm.h', 'nnet_utils/nnet_batchnorm_stream.h']


class BatchNormalizationQuantizedTanhConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(BatchNormalizationQuantizedTanh)
        self.template = batchnorm_quantized_tanh_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_input_variable().size_cpp()

        return self.template.format(**params)


class BatchNormalizationQuantizedTanhFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(BatchNormalizationQuantizedTanh, include_header=bn_include_list)
        self.template = batchnorm_quantized_tanh_function_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('quantize') == 2:
            params['quantize'] = 'binary'
            params['threshold'] = node.get_weights('threshold').name
        elif node.get_attr('quantize') == 3:
            params['quantize'] = 'ternary'
            params['threshold'] = node.get_weights('threshold_hi').name + ', ' + node.get_weights('threshold_lo').name

        return self.template.format(**params)


def register_bn_quant(backend):
    # Register the layer types to the layer map
    register_layer('BatchNormalizationQuantizedTanh', BatchNormalizationQuantizedTanh)

    # Register the optimization passes
    backend.register_pass('merge_batch_norm_quantized_tanh', MergeBatchNormAndQuantizedTanh)
    backend.register_pass('quantize_dense_output', QuantizeDenseOutput)

    # Register template passes
    backend.register_template(BatchNormalizationQuantizedTanhConfigTemplate)
    backend.register_template(BatchNormalizationQuantizedTanhFunctionTemplate)


class MergeBatchNormAndQuantizedTanh(OptimizerPass):
    def match(self, node):
        is_match = (
            node.class_name == 'Activation'
            and node.get_attr('activation') in ['binary', 'binary_tanh', 'ternary', 'ternary_tanh']
            or node.class_name == 'TernaryTanh'
        )
        is_match = is_match and isinstance(node.get_input_node(), BatchNormalization)
        return is_match

    def transform(self, model, node):
        bn_layer = node.get_input_node()
        # Make a new layer with the new attributes
        quantize = 0
        if 'binary' in node.get_attr('activation'):
            quantize = 2
        if 'ternary' in node.get_attr('activation'):
            quantize = 3
        attrs = {
            'name': bn_layer.get_attr('name'),
            'original_name': bn_layer.get_attr('name'),
            'class_name': 'BatchNormalizationQuantizedTanh',
            'n_in': bn_layer.get_attr('n_in'),
            'n_out': bn_layer.get_attr('n_in'),
            'n_filt': bn_layer.get_attr('n_filt'),
            'quantize': quantize,
            'trace': bn_layer.get_attr('trace'),
        }
        bnbt_layer = model.make_node(BatchNormalizationQuantizedTanh, 'bnbt_' + bn_layer.name, attrs, bn_layer.inputs)
        bnbt_layer.set_thresholds(
            bn_layer.get_weights('scale').data, bn_layer.get_weights('bias').data, node.get_attr('threshold', 0.5)
        )
        # Remove the BatchNormalization layer
        model.remove_node(bn_layer, rewire=True)
        # Replace the old Activation layer with this one
        model.replace_node(node, bnbt_layer)

        return True


class QuantizeDenseOutput(OptimizerPass):
    def match(self, node):
        is_dense = node.class_name == 'Dense'
        input_node = node.get_input_node()
        is_input_bnqt = input_node is not None and input_node.class_name == 'BatchNormalizationQuantizedTanh'
        quantizer = node.get_attr('weight_quantizer')
        is_binary_ternary = quantizer is not None and (
            quantizer.__class__.__name__ == 'BinaryQuantizer' or quantizer.__class__.__name__ == 'TernaryQuantizer'
        )
        return is_dense and is_input_bnqt and is_binary_ternary

    def transform(self, model, node):
        # Compute the required precision and update the variables
        # Number of bits for output is log2 of number of input nodes
        # Since this is the number of uint<1>'s which are summed
        nbits = int(np.ceil(np.log2(node.attributes['n_in'])) + 2)
        out_type = IntegerPrecisionType(width=nbits)
        accum_t = NamedType(f'layer{node.index}_accum_t', out_type)
        node.set_attr('accum_t', accum_t)
        out_var = node.get_output_variable()
        out_var.type.precision = out_type

        quantized_data = None
        quantized_precision = None
        quantizer = node.get_attr('weight_quantizer')
        if quantizer.__class__.__name__ == 'BinaryQuantizer':
            quantized_precision = XnorPrecisionType()
        elif quantizer.__class__.__name__ == 'TernaryQuantizer':
            quantized_precision = IntegerPrecisionType(width=2)
        else:
            print(f'WARNING: Unknown quantizer - {quantizer.__class__.__name__}. Bailing out')
            return False
        quantizer.bits = quantized_precision.width
        quantizer.hls_type = quantized_precision
        quantized_data = quantizer(node.weights['weight'].data)

        weights = node.weights['weight']
        weights.data = quantized_data
        weights.type.name = f'weight{node.index}_t'
        weights.update_precision(quantized_precision)

        bias = node.weights['bias']
        bias.data = np.zeros(shape=(node.get_attr('n_out')))
        bias.type.name = f'bias{node.index}_t'
        bias.nzeros = 0
        bias.update_precision(quantized_precision)

        # If followed by the BatchNormalizationBinaryTanh, update its input
        # Also requantise the weights
        bd_out_nodes = node.get_output_nodes()
        for out_node in bd_out_nodes:
            if isinstance(out_node, BatchNormalizationQuantizedTanh):
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
