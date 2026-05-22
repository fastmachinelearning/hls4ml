from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import (
    Input,
    Reshape,
    SparseActivation,
    SparseConv2D,
    SparseFlatten,
    SparseInputReduce,
    SparsePooling2D,
)
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer

sparsepixels_include = ['nnet_utils/nnet_sparsepixels.h']

# Optimizer pass: trace hash vars & Flatten->SparseFlatten


class SparseGraphOptimizer(OptimizerPass):
    """Triggered by SparseInputReduce. Walks the full graph to wire hash variable names,
    track spatial dims, and replace Flatten->SparseFlatten."""

    def match(self, node):
        return isinstance(node, SparseInputReduce) and node.get_attr('hash_out_name', None) is None

    def transform(self, model, node):
        hash_map = {}
        spatial = {}
        changed = False

        for name, n in list(model.graph.items()):
            if isinstance(n, SparseInputReduce):
                h_var = f'sparse_hash_{name}'
                n.set_attr('hash_out_name', h_var)
                hash_map[name] = h_var
                spatial[name] = (n.get_attr('in_height'), n.get_attr('in_width'))

            elif isinstance(n, SparseConv2D):
                src = n.inputs[1] if len(n.inputs) > 1 else n.inputs[0]
                h_var = hash_map.get(src, hash_map.get(n.inputs[0]))
                n.set_attr('hash_in_name', h_var)
                n.set_attr('hash_out_name', h_var)
                hash_map[name] = h_var
                spatial[name] = spatial.get(src, spatial.get(n.inputs[0]))

            elif isinstance(n, FixedPointQuantizer):
                src = n.inputs[0]
                if src in hash_map:
                    hash_map[name] = hash_map[src]
                    spatial[name] = spatial.get(src)

            elif isinstance(n, SparseActivation):
                src = n.inputs[0]
                h_var = hash_map.get(src)
                hash_map[name] = h_var
                spatial[name] = spatial.get(src)

            elif isinstance(n, SparsePooling2D):
                src = n.inputs[1] if len(n.inputs) > 1 else n.inputs[0]
                h_in = hash_map.get(src, hash_map.get(n.inputs[0]))
                h_out = f'sparse_hash_{name}'
                n.set_attr('hash_in_name', h_in)
                n.set_attr('hash_out_name', h_out)
                hash_map[name] = h_out
                ps = n.get_attr('pool_size')
                prev_h, prev_w = spatial.get(src, spatial.get(n.inputs[0], (0, 0)))
                spatial[name] = (prev_h // ps, prev_w // ps)

            elif isinstance(n, SparseFlatten):
                src = n.inputs[0]
                h_var = hash_map.get(src)
                if h_var is not None:
                    n.set_attr('hash_in_name', h_var)
                    hash_map[name] = h_var
                    spatial[name] = spatial.get(src, (1, 1))

            elif isinstance(n, Reshape):
                src = n.inputs[0]
                if src in hash_map:
                    src_node = model.graph[src]
                    n_sparse = src_node.get_attr('n_sparse', None)
                    if n_sparse is None:
                        continue
                    n_chan = src_node.get_attr('n_chan', None) or src_node.get_attr('n_filt', None)
                    h_var = hash_map[src]
                    sp = spatial.get(src, (1, 1))

                    attrs = {
                        'n_sparse': n_sparse,
                        'n_chan': n_chan,
                        'out_height': sp[0],
                        'out_width': sp[1],
                        'hash_in_name': h_var,
                    }
                    new_node = model.make_node('SparseFlatten', name, attrs, n.inputs.copy(), outputs=n.outputs.copy())
                    model.replace_node(n, new_node)
                    changed = True

        return changed


#  Config templates (struct definitions)

sparse_input_reduce_config = """struct config{index} {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned n_sparse = {n_sparse};
    static const unsigned hash_bits = {hash_bits};
}};\n"""

sparse_conv2d_config = """struct config{index} {{
    static const unsigned n_sparse = {n_sparse};
    static const unsigned n_chan = {n_chan};
    static const unsigned n_filt = {n_filt};
    static const unsigned kernel_size = {kernel_size};
    typedef {accum_t.name} accum_t;
}};\n"""

sparse_activation_config = """struct config{index} {{
    static const unsigned n_sparse = {n_sparse};
    static const unsigned n_chan = {n_chan};
}};\n"""

sparse_pooling2d_config = """struct config{index} {{
    static const unsigned n_sparse = {n_sparse};
    static const unsigned n_chan = {n_chan};
    static const unsigned pool_size = {pool_size};
    typedef {accum_t.name} accum_t;
}};\n"""

sparse_flatten_config = """struct config{index} {{
    static const unsigned n_sparse = {n_sparse};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
}};\n"""


class SparseInputReduceConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SparseInputReduce)
        self.template = sparse_input_reduce_config

    def format(self, node):
        return self.template.format(**self._default_config_params(node))


class SparseConv2DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SparseConv2D)
        self.template = sparse_conv2d_config

    def format(self, node):
        return self.template.format(**self._default_config_params(node))


class SparseActivationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SparseActivation)
        self.template = sparse_activation_config

    def format(self, node):
        return self.template.format(**self._default_config_params(node))


class SparsePooling2DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SparsePooling2D)
        self.template = sparse_pooling2d_config

    def format(self, node):
        return self.template.format(**self._default_config_params(node))


class SparseFlattenConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SparseFlatten)
        self.template = sparse_flatten_config

    def format(self, node):
        return self.template.format(**self._default_config_params(node))


#  Function-call templates

sparse_input_reduce_function = (
    '{input_t} threshold_{index} = {threshold};\n'
    'ap_uint<{hash_bits}> {hash_out}[{n_sparse} * 2];\n'
    '#pragma HLS ARRAY_PARTITION variable={hash_out} complete dim=0\n'
    'sparse_input_reduce<{input_t}, {output_t}, ap_uint<{hash_bits}>, {in_height}, {in_width}, {n_chan}, {n_sparse}>'
    '({input}, threshold_{index}, {output}, {hash_out});'
)

sparse_conv2d_function = (
    'sparse_conv<{input_t}, {output_t}, ap_uint<{hash_bits}>, {weight_t}, {bias_t}, {accum_t_name}, '
    '{n_sparse}, {n_chan}, {n_filt}, {kernel_size}>'
    '({input}, {output}, {hash_in}, {w}, {b});'
)

sparse_activation_function = 'sparse_relu<{input_t}, {output_t}, {n_sparse}, {n_chan}>({input}, {output});'

sparse_pooling2d_function = (
    'ap_uint<{hash_bits}> {hash_out}[{n_sparse} * 2];\n'
    '#pragma HLS ARRAY_PARTITION variable={hash_out} complete dim=0\n'
    'sparse_pooling_avg<{input_t}, {output_t}, ap_uint<{hash_bits}>, {accum_t_name}, {n_sparse}, {n_chan}, {pool_size}>'
    '({input}, {output}, {hash_in}, {hash_out});'
)

sparse_flatten_function = (
    'sparse_flatten<{input_t}, {output_t}, ap_uint<{hash_bits}>, {out_height}, {out_width}, {n_chan}, {n_sparse}>'
    '({input}, {hash_in}, {output});'
)


def _get_hash_bits(node):
    inp = node
    while inp is not None:
        hb = inp.get_attr('hash_bits', None)
        if hb is not None:
            return hb
        if len(inp.inputs) > 0:
            inp = inp.model.graph.get(inp.inputs[0])
        else:
            break
    return 10


class SparseInputReduceFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SparseInputReduce, include_header=sparsepixels_include)
        self.template = sparse_input_reduce_function

    def format(self, node):
        params = self._default_function_params(node)
        params['in_height'] = node.get_attr('in_height')
        params['in_width'] = node.get_attr('in_width')
        params['n_chan'] = node.get_attr('n_chan')
        params['n_sparse'] = node.get_attr('n_sparse')
        params['hash_bits'] = node.get_attr('hash_bits')
        params['threshold'] = node.get_attr('threshold')
        params['hash_out'] = node.get_attr('hash_out_name')
        return self.template.format(**params)


class SparseConv2DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SparseConv2D, include_header=sparsepixels_include)
        self.template = sparse_conv2d_function

    def format(self, node):
        params = self._default_function_params(node)
        params['n_sparse'] = node.get_attr('n_sparse')
        params['n_chan'] = node.get_attr('n_chan')
        params['n_filt'] = node.get_attr('n_filt')
        params['kernel_size'] = node.get_attr('kernel_size')
        params['hash_bits'] = _get_hash_bits(node)
        params['hash_in'] = node.get_attr('hash_in_name')
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name
        params['weight_t'] = node.get_weights('weight').type.name
        params['bias_t'] = node.get_weights('bias').type.name
        params['accum_t_name'] = node.get_attr('accum_t').name
        return self.template.format(**params)


class SparseActivationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SparseActivation, include_header=sparsepixels_include)
        self.template = sparse_activation_function

    def format(self, node):
        params = self._default_function_params(node)
        params['n_sparse'] = node.get_attr('n_sparse')
        params['n_chan'] = node.get_attr('n_chan')
        return self.template.format(**params)


class SparsePooling2DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SparsePooling2D, include_header=sparsepixels_include)
        self.template = sparse_pooling2d_function

    def format(self, node):
        params = self._default_function_params(node)
        params['n_sparse'] = node.get_attr('n_sparse')
        params['n_chan'] = node.get_attr('n_chan')
        params['pool_size'] = node.get_attr('pool_size')
        params['hash_bits'] = _get_hash_bits(node)
        params['hash_in'] = node.get_attr('hash_in_name')
        params['hash_out'] = node.get_attr('hash_out_name')
        params['accum_t_name'] = node.get_attr('accum_t').name
        return self.template.format(**params)


class SparseFlattenFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SparseFlatten, include_header=sparsepixels_include)
        self.template = sparse_flatten_function

    def format(self, node):
        params = self._default_function_params(node)
        params['n_sparse'] = node.get_attr('n_sparse')
        params['n_chan'] = node.get_attr('n_chan')
        params['out_height'] = node.get_attr('out_height')
        params['out_width'] = node.get_attr('out_width')
        params['hash_bits'] = _get_hash_bits(node)
        params['hash_in'] = node.get_attr('hash_in_name')
        return self.template.format(**params)


#  Optimizer pass: fix Input precision for sparse models


class SparseFixInputPrecision(OptimizerPass):
    """Fix Input precision for sparse models.

    The standard FixInputPrecision cannot find FixedPointQuantizer nodes through
    sparse layers (Input -> SparseInputReduce -> FPQ), so it falls back to a
    minimal type. This pass corrects the Input precision using the downstream
    FPQ's mask, then re-registers SparseInputReduce with the corrected type.
    """

    def match(self, node):
        if not isinstance(node, Input):
            return False
        model = node.model
        for layer in model.graph.values():
            if isinstance(layer, SparseInputReduce) and node.name in layer.inputs:
                return True
        return False

    def transform(self, model, node):
        from hls4ml.model.optimizer.passes.bit_exact import (
            produce_kif,
            register_precision,
            to_hls4ml_fixed,
        )

        sparse_reduce = None
        for layer in model.graph.values():
            if isinstance(layer, SparseInputReduce) and node.name in layer.inputs:
                sparse_reduce = layer
                break
        if sparse_reduce is None:
            return False

        fpq = None
        for layer in model.graph.values():
            if isinstance(layer, FixedPointQuantizer) and sparse_reduce.name in layer.inputs:
                fpq = layer
                break
        if fpq is None:
            return False

        # Read FPQ's output type, which was correctly set by BitExact's
        # register_precision using per-element max(k), max(i), max(f).
        # We do NOT call _produce_kif(fpq) here because that would re-clip
        # against the currently-wrong Input precision (set to ap_ufixed<1,0>
        # by the standard FixInputPrecision which can't recurse through sparse layers).
        fpq_prec = fpq.get_output_variable().type.precision
        k = 1 if fpq_prec.signed else 0
        i = fpq_prec.integer - k
        f = fpq_prec.width - fpq_prec.integer

        new_type = to_hls4ml_fixed(k, i, f + 1, f'{node.name}_t')
        if hasattr(fpq, 'SAT') and fpq.SAT in ('SAT', 'SAT_SYM'):
            new_type.precision.saturation_mode = 'SAT'
        else:
            new_type.precision.saturation_mode = 'WRAP'
        node.get_output_variable().type = new_type
        node.model.config.layer_name_precision[node.name] = str(new_type)
        node.attributes['trusted'] = True

        produce_kif(sparse_reduce, force_reset=True)
        register_precision(sparse_reduce)
        for attr in ('_produce_kif', '_request_kif'):
            if attr in sparse_reduce.attributes:
                del sparse_reduce.attributes[attr]

        return False


#  Backend registration hook


def register_sparsepixels(backend):
    backend.register_pass('sparse_graph_optimizer', SparseGraphOptimizer)
    backend.register_pass('sparse_fix_input_precision', SparseFixInputPrecision)

    backend.register_pass('sparseinputreduce_config_template', SparseInputReduceConfigTemplate)
    backend.register_pass('sparseinputreduce_function_template', SparseInputReduceFunctionTemplate)
    backend.register_pass('sparseconv2d_config_template', SparseConv2DConfigTemplate)
    backend.register_pass('sparseconv2d_function_template', SparseConv2DFunctionTemplate)
    backend.register_pass('sparseactivation_config_template', SparseActivationConfigTemplate)
    backend.register_pass('sparseactivation_function_template', SparseActivationFunctionTemplate)
    backend.register_pass('sparsepooling2d_config_template', SparsePooling2DConfigTemplate)
    backend.register_pass('sparsepooling2d_function_template', SparsePooling2DFunctionTemplate)
    backend.register_pass('sparseflatten_config_template', SparseFlattenConfigTemplate)
    backend.register_pass('sparseflatten_function_template', SparseFlattenFunctionTemplate)
