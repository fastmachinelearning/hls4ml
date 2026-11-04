"""
indexed_conv.py
=================
Registers ``IndexedConvolutionLayer`` (Keras v2 / tf.keras,
``use_3d_conv=False``) into hls4ml using the Extension API.

Mathematical equivalence
------------------------
``IndexedConvolutionLayer`` wraps a Keras ``Conv2D(filters,
kernel_size=(1, K), padding='valid', activation='relu')``. Applied to
gather output of shape ``[batch, n_pixels, K, C_in]``, it produces
``[batch, n_pixels, 1, C_out]`` and squeezes ``axis=2`` to yield
``[batch, n_pixels, C_out]``.

The equivalent hls4ml operation (``io_parallel``) is::

    for each pixel p:
        output[p, f_out] = ReLU(
            bias[f_out]
            + sum_{k,f_in} input[p, k, f_in] * kernel[f_out, k, f_in]
        )

The Keras kernel has shape ``(1, K, C_in, C_out)``. Slice ``[0]`` is
extracted, transposed to ``(C_out, K, C_in)``, and flattened to a 1-D
array of length ``C_out * K * C_in``. This layout matches the indexing
used in ``nnet_indexed_conv.h``::

    weights[f_out * K * C_in + k * C_in + f_in]

reuse_factor controls multiplier reuse in the MAC loop. Total MACs per
pixel = n_neighbors * n_features_in * n_features_out. With
reuse_factor=R, the number of parallel multipliers is reduced by R and
the MAC loop initiation interval becomes R. Set reuse_factor=1 for
fully parallel (latency-optimal) operation.

IR shape contract
-----------------
Input  (from ``HNeighborGather2D``): ``[n_pixels, n_neighbors, n_features_in]``  (3D)
Output (to next layer):              ``[n_pixels, n_features_out]``               (2D)
"""

import os

import numpy as np

import hls4ml
import hls4ml.backends.template
import hls4ml.converters
import hls4ml.model.layers

# ============================================================
# 1.  hls4ml IR layer class
# ============================================================


class HIndexedConv2D(hls4ml.model.layers.Layer):
    """
    hls4ml internal representation of IndexedConvolutionLayer
    (use_3d_conv=False).

    Input shape  (excl. batch): [n_pixels, n_neighbors, n_features_in]  (3D)
    Output shape (excl. batch): [n_pixels, n_features_out]               (2D)
    """

    def initialize(self):
        inp = self.get_input_variable()
        pack_neighbors = self.get_attr('pack_neighbors', False)

        if pack_neighbors:
            assert len(inp.shape) == 2, (
                f'HIndexedConv2D with pack_neighbors=True expects 2D '
                f'input [n_pixels, n_neighbors*n_features_in], got shape '
                f'{inp.shape}. This requires the preceding '
                f'NeighborGatherLayer to also have pack_neighbors=True '
                f'(mismatched pack_neighbors settings between a gather '
                f'and its consumer produce a shape mismatch here).'
            )
        else:
            assert len(inp.shape) == 3, (
                f'HIndexedConv2D expects 3D input '
                f'[n_pixels, n_neighbors, n_features_in], '
                f'got shape {inp.shape}. Verify that NeighborGatherLayer '
                f'emits the correct output shape, or that pack_neighbors '
                f'is set consistently on both layers.'
            )

        n_pixels = inp.shape[0]
        n_features_out = self.get_attr('n_features_out')

        out_shape = [n_pixels, n_features_out]
        self.add_output_variable(out_shape)

        # Select the weight layout matching the active backend. Both
        # layouts are precomputed at parse time (see
        # parse_indexed_convolution_layer); io_type is only known here,
        # at graph-build time, not during parsing.
        io_type = self.model.config.get_config_value('IOType')
        weight_data = (
            self.get_attr('weight_data_stream') if io_type == 'io_stream' else self.get_attr('weight_data_parallel')
        )
        self.add_weights_variable(
            name='weight',
            var_name='w{index}',
            data=weight_data,
            quantizer=self.get_attr('weight_quantizer'),
        )
        self.add_weights_variable(
            name='bias',
            var_name='b{index}',
            data=self.get_attr('bias_data'),
            quantizer=self.get_attr('bias_quantizer'),
        )


# ============================================================
# 2.  Keras v2 parser
# ============================================================


def parse_indexed_convolution_layer(keras_layer, input_names, input_shapes, data_reader):
    """
    Translate a Keras ``IndexedConvolutionLayer`` config dict into an hls4ml
    layer attribute dict and extract the convolution weights.

    The Conv2D kernel stored in the Keras layer has shape
    ``(1, K, C_in, C_out)``. Slice ``[0]`` is extracted, transposed to
    ``(C_out, K, C_in)``, and flattened. This layout matches the inner-loop
    indexing in ``nnet_indexed_conv.h``.
    """
    cfg = keras_layer['config']

    if cfg.get('use_3d_conv', False):
        raise NotImplementedError(
            'IndexedConvolutionLayer with use_3d_conv=True is not supported in hls4ml (Conv3D backend support is absent).'
        )

    layer = {}
    layer['class_name'] = 'IndexedConvolutionLayer'
    layer['name'] = cfg['name']
    layer['n_features_out'] = int(cfg['filters'])

    if input_names is not None:
        layer['inputs'] = input_names

    in_shape = list(input_shapes[0])
    if in_shape and in_shape[0] is None:
        in_shape = in_shape[1:]

    assert len(in_shape) == 3, (
        f'IndexedConvolutionLayer parser expects 3D input [n_pixels, n_neighbors, n_features_in], got {in_shape}.'
    )

    n_pixels = in_shape[0]
    n_neighbors_in = in_shape[1]
    n_features_in = in_shape[2]
    n_features_out = layer['n_features_out']

    # Stored so the config template can read these without inspecting
    # the node's actual input Variable shape, which is 2D (flattened)
    # when the preceding NeighborGatherLayer has pack_neighbors=True.
    layer['n_neighbors_in'] = int(n_neighbors_in)
    layer['n_features_in'] = int(n_features_in)

    keras_conv_layer = data_reader.model.get_layer(cfg['name']).conv
    kernel, bias = keras_conv_layer.get_weights()  # (1, K, C_in, C_out), (C_out,)

    # weight_data_stream layout: (K, C_out, C_in), matching
    # nnet::dense_resource_rf_leq_nin's internal weight indexing
    # convention (weights[out_index * n_in + in_index], i.e.
    # output-major/input-minor -- the transpose of dense_latency's
    # weights[in*n_out+out] convention). Confirmed by tracing the
    # ReuseLoop/MultLoop index arithmetic in nnet_dense_resource.h.
    weight_flat_parallel = kernel[0].transpose(2, 0, 1).flatten().astype(np.float32)
    weight_flat_stream = kernel[0].transpose(0, 2, 1).flatten().astype(np.float32)

    layer['weight_data_parallel'] = weight_flat_parallel
    layer['weight_data_stream'] = weight_flat_stream
    layer['bias_data'] = bias.astype(np.float32)
    layer['weight_quantizer'] = None
    layer['bias_quantizer'] = None

    out_shape = [None, n_pixels, n_features_out]
    return layer, out_shape


# ============================================================
# 3.  HLS config and function call templates
# ============================================================

_indexed_conv_config_template = """\
struct config{index} : nnet::indexed_conv_config {{
    static const unsigned n_pixels       = {n_pixels};
    static const unsigned n_neighbors    = {n_neighbors};
    static const unsigned n_features_in  = {n_features_in};
    static const unsigned n_features_out = {n_features_out};
    static const unsigned reuse_factor   = {reuse_factor};
    static const unsigned n_zeros        = 0;
    static const unsigned multiplier_limit = {multiplier_limit};
    static const bool pack_neighbors     = {pack_neighbors};
    typedef config{index}_dense dense_config;
}};
"""

_indexed_conv_dense_config_template = """\
struct config{index}_dense : nnet::dense_config {{
    static const unsigned n_in  = {n_features_in};
    static const unsigned n_out = {n_features_out};
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
}};
"""

_indexed_conv_function_template = (
    'nnet::indexed_conv_2d<{input_t}, {output_t}, {weight_t}, {bias_t}, config{index}>({input}, {output}, {w}, {b});'
)

_indexed_conv_include_list = ['nnet_utils/nnet_indexed_conv.h', 'nnet_utils/nnet_dense_resource.h']


class HIndexedConv2DConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HIndexedConv2D)
        self.template = _indexed_conv_config_template
        self.dense_template = _indexed_conv_dense_config_template

    def format(self, node):
        params = self._default_config_params(node)
        inp = node.get_input_variable()
        n_neighbors = node.get_attr('n_neighbors_in')
        n_features_in = node.get_attr('n_features_in')
        n_features_out = node.get_attr('n_features_out')

        # reuse_factor is set globally via hls_config and stored in the node
        reuse_factor = node.get_attr('reuse_factor', 1)
        n_macs = n_neighbors * n_features_in * n_features_out
        multiplier_limit = max(1, n_macs // reuse_factor)

        params['n_pixels'] = inp.shape[0]
        params['n_neighbors'] = n_neighbors
        params['n_features_in'] = n_features_in
        params['n_features_out'] = n_features_out
        params['reuse_factor'] = reuse_factor
        params['multiplier_limit'] = multiplier_limit
        params['pack_neighbors'] = 'true' if node.get_attr('pack_neighbors', False) else 'false'

        main_config = self.template.format(**params)

        # Nested config used only by the io_stream overload of
        # indexed_conv_2d, which delegates the per-neighbor MAC step to
        # nnet::dense_resource (a standard hls4ml Dense n_in x n_out
        # kernel). Ignored by the io_parallel overload.
        dense_params = dict(params)
        dense_params['weight_t'] = node.get_weights('weight').type.name
        dense_params['bias_t'] = node.get_weights('bias').type.name
        dense_params['accum_t'] = 'model_default_t'
        dense_config = self.dense_template.format(**dense_params)

        return dense_config + '\n' + main_config


class HIndexedConv2DFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HIndexedConv2D, include_header=_indexed_conv_include_list)
        self.template = _indexed_conv_function_template

    def format(self, node):
        params = self._default_function_params(node)
        # weight_t and bias_t are not filled by _default_function_params
        # for custom layers; extract them from the weight variable objects.
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name
        params['weight_t'] = node.get_weights('weight').type.name
        params['bias_t'] = node.get_weights('bias').type.name
        params['output_t'] = node.get_output_variable().type.name
        return self.template.format(**params)


# ============================================================
# 4.  Registration (executed automatically on import)
# ============================================================


def _register():
    from hls4ml.converters.keras_v2_to_hls import layer_handlers
    from hls4ml.model.layers import layer_map

    if 'IndexedConvolutionLayer' not in layer_handlers:
        hls4ml.converters.register_keras_v2_layer_handler('IndexedConvolutionLayer', parse_indexed_convolution_layer)

    if 'IndexedConvolutionLayer' not in layer_map:
        hls4ml.model.layers.register_layer('IndexedConvolutionLayer', HIndexedConv2D)

    _hls_header = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nnet_utils', 'nnet_indexed_conv.h')
    if not os.path.isfile(_hls_header):
        raise FileNotFoundError(
            f'HLS header not found: {_hls_header}\n'
            "Ensure that 'nnet_indexed_conv.h' is present in the 'nnet_utils/' subdirectory of this package."
        )

    for backend_name in ['Vivado', 'Vitis']:
        try:
            backend = hls4ml.backends.get_backend(backend_name)
        except Exception:
            continue
        backend.register_template(HIndexedConv2DConfigTemplate)
        backend.register_template(HIndexedConv2DFunctionTemplate)
        backend.register_source(_hls_header)


_register()
