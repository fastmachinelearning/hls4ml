"""
neighbor_gather.py
===================
Registers ``NeighborGatherLayer`` (Keras v2 / tf.keras, ``use_3d_conv=False``)
into hls4ml so that ``convert_from_keras_model()`` can translate it to HLS
firmware automatically.

The HLS kernel lives in ``nnet_utils/nnet_neighbor_gather.h`` and implements
a fully-combinational gather (``#pragma HLS INLINE`` + full ``UNROLL``),
suitable for the ``io_parallel`` backend. Border pixels (``index == -1``)
are zero-masked in hardware.

Notes
-----
- Only ``use_3d_conv=False`` is supported; Conv3D is not available in hls4ml.
- Each generated config struct declares its own ``indices[]`` array rather
  than inheriting from the base struct, which avoids C++ member-hiding
  warnings in Vitis HLS.
- Importing this module multiple times is safe; registration is guarded
  against duplication.
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


class HNeighborGather2D(hls4ml.model.layers.Layer):
    """
    hls4ml internal representation of NeighborGatherLayer (use_3d_conv=False).

    Input shape  (excl. batch): [n_pixels, n_features]
    Output shape (excl. batch): [n_pixels, n_neighbors, n_features]

    Neighbor indices are embedded in the HLS config struct; no weight
    variables are registered.
    """

    def initialize(self):
        inp = self.get_input_variable()

        assert len(inp.shape) == 2, (
            f'HNeighborGather2D expects 2D input [n_pixels, n_features], '
            f'got shape {inp.shape}. Verify that the preceding layer '
            f'emits the correct output shape.'
        )

        n_pixels = inp.shape[0]
        n_features = inp.shape[1]
        n_neighbors = self.get_attr('n_neighbors')

        # pack_neighbors=True (io_stream only) flattens the neighbor
        # and feature dimensions into a single wide packet per pixel,
        # matching this library's "packed" conv/pool overloads.
        # Ignored for io_parallel, which always uses the 3D shape.
        pack_neighbors = self.get_attr('pack_neighbors', False)
        io_type = self.model.config.get_config_value('IOType')

        if pack_neighbors and io_type == 'io_stream':
            out_shape = [n_pixels, n_neighbors * n_features]
        else:
            out_shape = [n_pixels, n_neighbors, n_features]
        self.add_output_variable(out_shape)


# ============================================================
# 2.  Keras v2 parser
# ============================================================


def parse_neighbor_gather_layer(keras_layer, input_names, input_shapes, data_reader):
    """
    Translate a Keras ``NeighborGatherLayer`` config dict into an hls4ml
    layer attribute dict.

    ``input_shapes[0]`` includes a leading ``None`` batch dimension when
    the layer follows ``InputLayer``, but not when it follows another custom
    layer. The leading ``None`` is stripped before shape validation.
    """
    cfg = keras_layer['config']

    if cfg.get('use_3d_conv', False):
        raise NotImplementedError(
            'NeighborGatherLayer with use_3d_conv=True is not supported in hls4ml (Conv3D backend support is absent).'
        )

    layer = {}
    layer['class_name'] = 'NeighborGatherLayer'
    layer['name'] = cfg['name']
    layer['neighbor_indices'] = cfg['neighbor_indices']  # list[list[int]]

    indices_array = np.array(cfg['neighbor_indices'], dtype=np.int32)
    assert indices_array.ndim == 2, 'neighbor_indices must be 2-dimensional: [n_pixels, n_neighbors]'
    layer['n_neighbors'] = int(indices_array.shape[1])

    if input_names is not None:
        layer['inputs'] = input_names

    in_shape = list(input_shapes[0])
    if in_shape and in_shape[0] is None:
        in_shape = in_shape[1:]

    assert len(in_shape) == 2, f'NeighborGatherLayer parser expects 2D input [n_pixels, n_features], got {in_shape}.'

    n_pixels = in_shape[0]
    n_features = in_shape[1]
    out_shape = [None, n_pixels, layer['n_neighbors'], n_features]

    return layer, out_shape


# ============================================================
# 3.  Helper: C array literal from neighbor indices
# ============================================================


def _indices_to_c_array(indices_list):
    """
    Flatten a list-of-lists of integer neighbor indices into a C
    brace-initializer string.

    Example: ``[[0, 1, -1], [2, 3, 4]]`` -> ``'{0, 1, -1, 2, 3, 4}'``

    Border-pixel slots are represented as ``-1`` and are preserved here;
    the HLS kernel handles them with conditional zeroing.
    """
    flat = []
    for row in indices_list:
        if isinstance(row, (list, tuple, np.ndarray)):
            flat.extend(int(v) for v in row)
        else:
            flat.append(int(row))
    return '{' + ', '.join(str(v) for v in flat) + '}'


# ============================================================
# 4.  HLS config and function call templates
# ============================================================

_gather2d_config_template = """\
struct config{index} : nnet::neighbor_gather_config {{
    static const unsigned n_pixels    = {n_pixels};
    static const unsigned n_neighbors = {n_neighbors};
    static const unsigned n_features  = {n_features};
    static const bool pack_neighbors  = {pack_neighbors};
    static const int indices[{n_pixels_x_neighbors}];
}};
const int config{index}::indices[{n_pixels_x_neighbors}] = {indices_flat};
"""

_gather2d_function_template = 'nnet::neighbor_gather_2d<{input_t}, {output_t}, config{index}>({input}, {output});'

_gather_include_list = ['nnet_utils/nnet_neighbor_gather.h']


class HNeighborGather2DConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HNeighborGather2D)
        self.template = _gather2d_config_template

    def format(self, node):
        params = self._default_config_params(node)
        inp = node.get_input_variable()
        n_pixels = inp.shape[0]
        n_neighbors = node.get_attr('n_neighbors')
        n_features = inp.shape[1]

        params['n_pixels'] = n_pixels
        params['n_neighbors'] = n_neighbors
        params['n_features'] = n_features
        params['n_pixels_x_neighbors'] = n_pixels * n_neighbors
        params['pack_neighbors'] = 'true' if node.get_attr('pack_neighbors', False) else 'false'
        params['indices_flat'] = _indices_to_c_array(node.get_attr('neighbor_indices'))
        return self.template.format(**params)


class HNeighborGather2DFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HNeighborGather2D, include_header=_gather_include_list)
        self.template = _gather2d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['output_t'] = node.get_output_variable().type.name
        return self.template.format(**params)


# ============================================================
# 5.  Registration (executed automatically on import)
# ============================================================


def _register():
    from hls4ml.converters.keras_v2_to_hls import layer_handlers
    from hls4ml.model.layers import layer_map

    if 'NeighborGatherLayer' not in layer_handlers:
        hls4ml.converters.register_keras_v2_layer_handler('NeighborGatherLayer', parse_neighbor_gather_layer)

    if 'NeighborGatherLayer' not in layer_map:
        hls4ml.model.layers.register_layer('NeighborGatherLayer', HNeighborGather2D)

    _hls_header = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nnet_utils', 'nnet_neighbor_gather.h')
    if not os.path.isfile(_hls_header):
        raise FileNotFoundError(
            f'HLS header not found: {_hls_header}\n'
            "Ensure that 'nnet_neighbor_gather.h' is present in the 'nnet_utils/' subdirectory of this package."
        )

    for backend_name in ['Vivado', 'Vitis']:
        try:
            backend = hls4ml.backends.get_backend(backend_name)
        except Exception:
            continue
        backend.register_template(HNeighborGather2DConfigTemplate)
        backend.register_template(HNeighborGather2DFunctionTemplate)
        backend.register_source(_hls_header)


_register()
