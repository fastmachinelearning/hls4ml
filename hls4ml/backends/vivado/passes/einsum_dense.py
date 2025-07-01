from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import EinsumDense
from hls4ml.utils.transpose_utils import transpose_config_gen

from .reshaping_templates import transpose_config_template

# Shared Dense template

dense_config_template = '''struct config{index}_dense : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::{dense_function}<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n'''

# EinsumDense template

einsum_dense_config_template = '''
struct config{index} {{
    typedef config{index}_tpose_inp tpose_inp_conf;
    typedef config{index}_tpose_out tpose_out_conf;
    {kernel_config};

    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;

    // Layer Sizes
    static const unsigned n_free_data = {n_free_data};
    static const unsigned n_free_kernel = {n_free_kernel};
    static const unsigned n_contract = {n_contract};
    static const unsigned n_inplace = {n_inplace};

    // Resource reuse info
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned parallelization_factor = {parallelization_factor}; // Only useful when n_inplace > 1
}};
'''

einsum_dense_function_template = 'nnet::einsum_dense<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
einsum_dense_da_function_template = 'nnet::einsum_dense<{input_t}, {output_t}, {config}>({input}, {output}, {b});'

einsum_dense_include_list = ['nnet_utils/nnet_einsum_dense.h', 'nnet_utils/nnet_dense.h']


class EinsumDenseConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(EinsumDense)
        self.template = einsum_dense_config_template
        self.dense_template = dense_config_template

    def dense_config(self, node: EinsumDense):
        dense_params = self._default_config_params(node)
        strategy = node.attributes['strategy']
        dense_params['strategy'] = strategy
        dense_params['n_in'] = node.attributes['n_contract']
        dense_params['n_out'] = node.attributes['n_free_kernel']
        if node.attributes['n_inplace'] == 1:
            dense_params['nzeros'] = node.get_weights('weight').nzeros  # type: ignore
        else:
            dense_params['nzeros'] = '-1; // Not making sense when kernels are switching'
        dense_params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision  # type: ignore
        )

        dense_params['dense_function'] = 'DenseLatency'  # Latency only for now

        dense_config = self.dense_template.format(**dense_params)
        return dense_config

    def format(self, node: EinsumDense):
        default_params = self._default_config_params(node)

        strategy = node.attributes['strategy']
        io_type = node.model.config.get_config_value('IOType')

        assert io_type == 'io_parallel', 'EinsumDense layer only supports io_parallel and distributed_arithmetic'

        # EinsumDense config
        params = default_params.copy()
        params['strategy'] = strategy
        params['n_free_data'] = node.attributes['n_free_data']
        params['n_free_kernel'] = node.attributes['n_free_kernel']
        params['n_contract'] = node.attributes['n_contract']
        params['n_inplace'] = node.attributes['n_inplace']
        if strategy.lower() == 'latency':
            params['kernel_config'] = f'typedef config{node.index}_dense dense_conf'
        else:
            assert strategy.lower() == 'distributed_arithmetic', 'EinsumDense layer only supports Latency strategy for now'
            inp_t = node.get_input_variable().type.name
            result_t = node.get_output_variable().type.name
            index = node.index
            conf = f'constexpr static auto da_kernel = nnet::einsum_dense{index}_da_kernel<{inp_t}, {result_t}>'
            params['kernel_config'] = conf
        pf = node.attributes['parallelization_factor']
        if pf < 0:
            pf = params['n_inplace']
        params['parallelization_factor'] = pf

        einsum_conf = self.template.format(**params)

        # inp/out transpose config
        inp_shape = node.attributes['inp_shape']
        out_interpert_shape = node.attributes['out_interpert_shape']
        inp_tpose_idxs = node.attributes['inp_tpose_idxs']
        out_tpose_idxs = node.attributes['out_tpose_idxs']
        tpose_inp_conf_name = f'config{node.index}_tpose_inp'
        tpose_out_conf_name = f'config{node.index}_tpose_out'

        conf = transpose_config_gen(tpose_inp_conf_name, inp_shape, inp_tpose_idxs)
        inp_tpose_conf = transpose_config_template.format(**conf)
        conf = transpose_config_gen(tpose_out_conf_name, out_interpert_shape, out_tpose_idxs)
        out_tpose_conf = transpose_config_template.format(**conf)

        if strategy.lower() == 'distributed_arithmetic':
            return '\n\n'.join((inp_tpose_conf, out_tpose_conf, einsum_conf))

        dense_config = self.dense_config(node)
        return '\n\n'.join((inp_tpose_conf, out_tpose_conf, dense_config, einsum_conf))


class EinsumDenseFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(EinsumDense, include_header=einsum_dense_include_list)
        self.template = einsum_dense_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['b'] = node.get_weights('bias').name

        strategy = node.attributes['strategy']
        if strategy == 'distributed_arithmetic':
            return einsum_dense_da_function_template.format(**params)

        params['w'] = node.get_weights('weight').name
        return einsum_dense_function_template.format(**params)
