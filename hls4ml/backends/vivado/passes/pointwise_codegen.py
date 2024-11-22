from hls4ml.model.layers import Conv1D
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import Source


def generate_pointwise_conv1d_fn(layer_idx, reuse_factor=1):
    """Generate a C++ function for a pointwise convolution layer.

    Args:
        layer_idx (int): Index of layer ('index' attribute).
        reuse_factor (int): Number of partitions to divide the input into.

    Returns:
        str: Generated C++ function
    """

    generated_code = (
        'template<class data_T, class res_T, typename CONFIG_T>\n'
        'class pointwise_conv_{index} : public Conv1DKernel<data_T, res_T, CONFIG_T> {{\n'
        '  public:\n'
        '    static void conv(\n'
        '                     data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],\n'
        '                     res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],\n'
        '                     typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],\n'
        '                     typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {{\n'
        '        data_T data_tmp[CONFIG_T::reuse_factor][CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor];\n'  # noqa: E501
        '        #pragma HLS ARRAY_PARTITION variable=data_tmp complete dim=0\n'
        '        res_T res_tmp[CONFIG_T::reuse_factor][CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor];\n'  # noqa: E501
        '        #pragma HLS ARRAY_PARTITION variable=res_tmp complete dim=0\n\n'
        '    RFInputLoop:\n'
        '        for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {{\n'
        '        #pragma HLS UNROLL\n'
        '        InnerInputLoop:\n'
        '            for (int ii = 0; ii < CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor; ii++) {{\n'
        '                #pragma HLS UNROLL\n'
        '                data_tmp[jj][ii] = data[jj * CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor + ii];\n'  # noqa: E501
        '            }}\n'
        '        }}\n\n'
    ).format(index=layer_idx)
    indent = '        '
    for i in range(reuse_factor):
        generated_code += indent
        generated_code += (
            f'pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[{i}], res_tmp[{i}], weights, biases);\n'
        )

    generated_code += (
        '\n'
        '    RFOutputLoop:\n'
        '        for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {\n'
        '        #pragma HLS UNROLL\n'
        '        InnerOutputLoop:\n'
        '            for (int ii = 0; ii < CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor; ii++) {\n'
        '                #pragma HLS UNROLL\n'
        '                res[jj * CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor + ii] = res_tmp[jj][ii];\n'  # noqa: E501
        '            }\n'
        '        }\n'
        '    }\n'
        '};\n'
    )

    return generated_code


class GeneratePointwiseConv1D(OptimizerPass):
    '''Generates code for pointwise 1D convolution'''

    def match(self, node):
        return (
            isinstance(node, Conv1D)
            and node.model.config.get_config_value('IOType') == 'io_parallel'
            and node.get_attr('filt_width') == 1
        )

    def transform(self, model, node):
        self._generate_pointwise_conv1d(node)

    def _generate_pointwise_conv1d(self, node):
        code_str = generate_pointwise_conv1d_fn(
            node.get_attr('index'),
            node.get_attr('reuse_factor'),
        )

        node.set_attr('pointwise_conv1d_codegen', Source(code_str))
