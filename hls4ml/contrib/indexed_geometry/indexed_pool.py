"""
indexed_pool.py
=================
Registers ``IndexedPoolingLayer`` (Keras v2 / tf.keras,
``use_3d_conv=False``) into hls4ml using the Extension API.

Mathematical equivalence
------------------------
``IndexedPoolingLayer`` wraps a Keras ``MaxPool2D`` (or
``AveragePooling2D``) with ``pool_size=(1, K)``, ``padding='valid'``.
Applied to gather output of shape ``[batch, n_pixels, K, C]``, it
produces ``[batch, n_pixels, 1, C]`` and squeezes ``axis=2`` to yield
``[batch, n_pixels, C]``.

The equivalent hls4ml operations (``io_parallel``) are::

    MaxPool: output[p, f] = max_{k=0..K-1}         input[p, k, f]
    AvgPool: output[p, f] = (1/K) * sum_{k=0..K-1} input[p, k, f]

Border slots (``index == -1``) are zero-masked by ``NeighborGatherLayer``
before pooling, so they contribute 0 to both max and average -- consistent
with Keras behavior.

IR shape contract
-----------------
Input  (from ``HNeighborGather2D``): ``[n_pixels, n_neighbors, n_features]``  (3D)
Output (to next layer):              ``[n_pixels, n_features]``                (2D)
"""

import os

import hls4ml
import hls4ml.backends.template
import hls4ml.converters
import hls4ml.model.layers

# ============================================================
# 1.  hls4ml IR layer class
# ============================================================


class HIndexedPool2D(hls4ml.model.layers.Layer):
    """
    hls4ml internal representation of IndexedPoolingLayer
    (use_3d_conv=False).

    Input shape  (excl. batch): [n_pixels, n_neighbors, n_features]  (3D)
    Output shape (excl. batch): [n_pixels, n_features]               (2D)

    No learned parameters -- pure reduction over the neighbor dimension.
    """

    def initialize(self):
        inp = self.get_input_variable()
        pack_neighbors = self.get_attr('pack_neighbors', False)

        if pack_neighbors:
            assert len(inp.shape) == 2, (
                f'HIndexedPool2D with pack_neighbors=True expects 2D '
                f'input [n_pixels, n_neighbors*n_features], got shape '
                f'{inp.shape}. This requires the preceding '
                f'NeighborGatherLayer to also have pack_neighbors=True '
                f'(mismatched pack_neighbors settings between a gather '
                f'and its consumer produce a shape mismatch here).'
            )
        else:
            assert len(inp.shape) == 3, (
                f'HIndexedPool2D expects 3D input '
                f'[n_pixels, n_neighbors, n_features], '
                f'got shape {inp.shape}. Verify that NeighborGatherLayer '
                f'emits the correct output shape, or that pack_neighbors '
                f'is set consistently on both layers.'
            )

        n_pixels = inp.shape[0]
        n_features = self.get_attr('n_features')

        out_shape = [n_pixels, n_features]
        self.add_output_variable(out_shape)
        # n_filt is set explicitly here for any downstream consumer that
        # reads it directly. The BatchNormalization parser's own n_filt
        # inference now works natively, since parse_indexed_pooling_layer
        # returns out_shape with the leading batch-dim placeholder.
        self.set_attr('n_filt', n_features)


# ============================================================
# 2.  Keras v2 parser
# ============================================================


def parse_indexed_pooling_layer(keras_layer, input_names, input_shapes, data_reader):
    """
    Translate a Keras ``IndexedPoolingLayer`` config dict into an hls4ml
    layer attribute dict.
    """
    cfg = keras_layer['config']

    if cfg.get('use_3d_conv', False):
        raise NotImplementedError(
            'IndexedPoolingLayer with use_3d_conv=True is not supported in hls4ml (Conv3D backend support is absent).'
        )

    pooling_type = cfg.get('pooling_type', 'max').lower()
    if pooling_type not in ('max', 'average', 'avg'):
        raise ValueError(
            f"IndexedPoolingLayer: unsupported pooling_type '{pooling_type}'. Supported values are 'max' and 'average'."
        )

    layer = {}
    layer['class_name'] = 'IndexedPoolingLayer'
    layer['name'] = cfg['name']
    layer['pooling_type'] = 'max' if pooling_type == 'max' else 'avg'

    if input_names is not None:
        layer['inputs'] = input_names

    in_shape = list(input_shapes[0])
    if in_shape and in_shape[0] is None:
        in_shape = in_shape[1:]

    assert len(in_shape) == 3, (
        f'IndexedPoolingLayer parser expects 3D input [n_pixels, n_neighbors, n_features], got {in_shape}.'
    )

    n_pixels = in_shape[0]
    n_neighbors = in_shape[1]
    n_features = in_shape[2]

    # Stored so the config template can read these without inspecting
    # the node's actual input Variable shape, which is 2D (flattened)
    # when the preceding NeighborGatherLayer has pack_neighbors=True.
    layer['n_neighbors'] = int(n_neighbors)
    layer['n_features'] = int(n_features)

    out_shape = [None, n_pixels, n_features]
    return layer, out_shape


# ============================================================
# 3.  HLS config and function call templates
# ============================================================

_indexed_pool_config_template = """\
struct config{index} : nnet::indexed_pool_config {{
    static const unsigned n_pixels    = {n_pixels};
    static const unsigned n_neighbors = {n_neighbors};
    static const unsigned n_features  = {n_features};
    typedef {accum_t} accum_t;
    static const bool pack_neighbors  = {pack_neighbors};
}};
"""

_indexed_maxpool_function_template = 'nnet::indexed_maxpool_2d<{input_t}, {output_t}, config{index}>({input}, {output});'

_indexed_avgpool_function_template = 'nnet::indexed_avgpool_2d<{input_t}, {output_t}, config{index}>({input}, {output});'

_indexed_pool_include_list = ['nnet_utils/nnet_indexed_pool.h']


class HIndexedPool2DConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HIndexedPool2D)
        self.template = _indexed_pool_config_template

    def format(self, node):
        params = self._default_config_params(node)
        inp = node.get_input_variable()
        params['n_pixels'] = inp.shape[0]
        params['n_neighbors'] = node.get_attr('n_neighbors')
        params['n_features'] = node.get_attr('n_features')
        # Scalar accumulator type for the average-pooling sum-then-divide.
        # Must be a scalar precision (not the packed array/stream packet
        # type), since it's used element-wise as accum_T sum[n_features].
        # model_default_t is hls4ml's own always-present scalar fallback
        # type (same one already used automatically for weight_t/bias_t
        # in indexed_conv_2d's generated calls).
        params['accum_t'] = 'model_default_t'
        params['pack_neighbors'] = 'true' if node.get_attr('pack_neighbors', False) else 'false'
        return self.template.format(**params)


class HIndexedPool2DFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HIndexedPool2D, include_header=_indexed_pool_include_list)

    def format(self, node):
        params = self._default_function_params(node)
        params['output_t'] = node.get_output_variable().type.name
        pooling_type = node.get_attr('pooling_type', 'max')
        if pooling_type == 'max':
            return _indexed_maxpool_function_template.format(**params)
        else:
            return _indexed_avgpool_function_template.format(**params)


# ============================================================
# 4.  Registration (executed automatically on import)
# ============================================================


def _register():
    from hls4ml.converters.keras_v2_to_hls import layer_handlers
    from hls4ml.model.layers import layer_map

    if 'IndexedPoolingLayer' not in layer_handlers:
        hls4ml.converters.register_keras_v2_layer_handler('IndexedPoolingLayer', parse_indexed_pooling_layer)

    if 'IndexedPoolingLayer' not in layer_map:
        hls4ml.model.layers.register_layer('IndexedPoolingLayer', HIndexedPool2D)

    _hls_header = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nnet_utils', 'nnet_indexed_pool.h')
    if not os.path.isfile(_hls_header):
        raise FileNotFoundError(
            f'HLS header not found: {_hls_header}\n'
            "Ensure that 'nnet_indexed_pool.h' is present in the 'nnet_utils/' subdirectory of this package."
        )

    for backend_name in ['Vivado', 'Vitis']:
        try:
            backend = hls4ml.backends.get_backend(backend_name)
        except Exception:
            continue
        backend.register_template(HIndexedPool2DConfigTemplate)
        backend.register_template(HIndexedPool2DFunctionTemplate)
        backend.register_source(_hls_header)


_register()
