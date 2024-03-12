import numpy as np

from hls4ml.converters.onnx_to_hls import get_onnx_attribute, onnx_handler

pool_operations = ['AveragePool', 'MaxPool']


@onnx_handler(*pool_operations)
def parse_pool_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    if node.domain != 'qonnx.custom_op.channels_last':
        raise RuntimeError("Please convert the model to channels-last format with qonnx-to-channels-last")
    layer['class_name'] = node.op_type
    layer['data_format'] = 'channels_last'  # Default QONNX

    info = layer['class_name'].replace('Pool', '')
    strides = get_onnx_attribute(node, 'strides')
    kernel_shape = get_onnx_attribute(node, 'kernel_shape')
    pads = get_onnx_attribute(node, 'pads')
    layer['pads'] = pads
    dilations = get_onnx_attribute(node, 'dilations')
    if dilations is None:
        dilations = [1] * len(kernel_shape)
    layer['dilations'] = dilations

    if len(input_shapes[0]) == 3:  # 1D
        layer['class_name'] = info + 'Pooling1D'

        layer['n_filt'] = input_shapes[0][1]
        layer['n_in'] = input_shapes[0][2]

        layer['pool_width'] = kernel_shape[0]
        layer['stride_width'] = strides[0]

        # formula from ONNX Operators.md documentation
        layer['n_out'] = int(
            np.floor((layer['n_in'] + np.sum(pads) - ((kernel_shape[0] - 1) * dilations[0] + 1)) / strides[0] + 1)
        )

    elif len(input_shapes[0]) == 4:  # 2D
        layer['class_name'] = info + 'Pooling2D'

        layer['n_filt'] = input_shapes[0][3]
        layer['in_height'] = input_shapes[0][1]
        layer['in_width'] = input_shapes[0][2]

        layer['stride_height'] = strides[0]
        layer['stride_width'] = strides[1]
        layer['pool_height'] = layer['filt_height'] = kernel_shape[0]
        layer['pool_width'] = layer['filt_width'] = kernel_shape[1]

        layer['pad_top'] = pads[0]
        layer['pad_bottom'] = pads[2]
        layer['pad_left'] = pads[1]
        layer['pad_right'] = pads[3]

        # formula from ONNX Operators.md documentation
        layer['out_height'] = int(
            np.floor((layer['in_height'] + pads[0] + pads[2] - ((kernel_shape[0] - 1) * dilations[0] + 1)) / strides[0] + 1)
        )
        layer['out_width'] = int(
            np.floor((layer['in_width'] + pads[1] + pads[3] - ((kernel_shape[1] - 1) * dilations[1] + 1)) / strides[1] + 1)
        )

    return layer


global_pooling_layers = ['GlobalMaxPool', 'GlobalAveragePool']


@onnx_handler(*global_pooling_layers)
def parse_global_pooling_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['class_name'] = node.op_type
    layer['data_format'] = 'channels_last'  # default QONNX

    # Sonme default parameters for global pooling
    layer['n_out'] = 1
    layer['pad_left'] = layer['pad_right'] = 0
    layer['stride'] = 0

    info = layer['class_name'].replace('Pool', '')

    if len(input_shapes[0]) == 3:  # 1D
        layer['class_name'] = info + 'Pooling1D'

        layer['n_in'] = input_shapes[0][2]
        layer['n_filt'] = input_shapes[0][1]

    elif len(input_shapes[0]) == 4:
        layer['class_name'] = info + 'Pooling2D'

        layer['n_filt'] = input_shapes[0][1]
        layer['in_height'] = input_shapes[0][2]
        layer['in_width'] = input_shapes[0][3]

    return layer
