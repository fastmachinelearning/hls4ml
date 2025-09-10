from hls4ml.converters.onnx_to_hls import get_constant_value, get_onnx_attribute, onnx_handler


@onnx_handler('Transpose')
def parse_transpose_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Transpose'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    perm = [list(i.ints) for i in node.attribute][0]  # This will get something like [[a,b,c]][0] = [a,b,c]
    layer['perm'] = [x - 1 for x in perm[1:]]  # Ignore the batch dimension in ONNX, and adjust the perm indexing

    return layer


@onnx_handler('Reshape')
def parse_reshape_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    return layer


@onnx_handler('Flatten')
def parse_flatten_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['target_shape'] = [-1]  # does not contain batch dimension

    return layer


@onnx_handler('Resize')
def parse_resize_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Resize'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['in_height'] = input_shapes[0][2]
    layer['in_width'] = input_shapes[0][1]
    layer['out_width'] = input_shapes[0][1]
    layer['out_height'] = input_shapes[0][2]
    layer['n_chan'] = input_shapes[0][3]
    layer['algorithm'] = get_onnx_attribute(node, 'mode')
    # The following is used in initialize() method.
    # Probably a better solution would be to have a channels last parameter at QONNX level
    layer['data_format'] = (
        'channels_last' if any(node.domain == 'qonnx.custom_op.channels_last' for node in graph.node) else 'channels_first'
    )

    return layer


@onnx_handler('Pad')
def parse_pad_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'ZeroPadding'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['data_format'] = (
        'channels_last' if any(node.domain == 'qonnx.custom_op.channels_last' for node in graph.node) else 'channels_first'
    )

    mode = get_onnx_attribute(node, 'mode')
    if mode is not None and mode != 'constant':
        raise RuntimeError(f'Unsupported padding mode: {mode} in node {node.name}')

    pads = get_constant_value(graph, node.input[1])
    if len(input_names) > 2:
        const_val = get_constant_value(graph, node.input[2])
        if const_val != 0:
            raise RuntimeError(f'Only constant value of 0 supported for Pad node {node.name}, got {const_val}')

    if len(input_names) > 3:
        raise RuntimeError(f'Parsing axes input of Pad node {node.name} is not supported.')

    dim = 0
    if len(input_shapes[0]) == 3:
        dim = 1  # 2D input (batch, channels, width), will use ZeroPadding1D
        if layer['data_format'] == 'channels_first':
            _, channels, width = input_shapes[0]
            pad_left, pad_right = pads[2], pads[5]
        else:
            _, width, channels = input_shapes[0]
            pad_left, pad_right = pads[1], pads[4]
        out_width = width + pad_left + pad_right

        layer['n_chan'] = channels
        layer['in_width'] = width
        layer['out_width'] = out_width

        layer['pad_left'] = pad_left
        layer['pad_right'] = pad_right
    elif len(input_shapes[0]) == 4:
        dim = 2  # 3D input (batch, channels, height, width), will use ZeroPadding2D
        if layer['data_format'] == 'channels_first':
            _, channels, height, width = input_shapes[0]
            pad_top, pad_bottom = pads[2], pads[6]
            pad_left, pad_right = pads[3], pads[7]
        else:
            _, height, width, channels = input_shapes[0]
            pad_top, pad_bottom = pads[1], pads[5]
            pad_left, pad_right = pads[2], pads[6]
        out_height = height + pad_top + pad_bottom
        out_width = width + pad_left + pad_right

        layer['n_chan'] = channels
        layer['in_height'] = height
        layer['in_width'] = width
        layer['out_height'] = out_height
        layer['out_width'] = out_width

        layer['pad_top'] = pad_top
        layer['pad_bottom'] = pad_bottom
        layer['pad_left'] = pad_left
        layer['pad_right'] = pad_right
    else:
        raise RuntimeError(f'Unsupported input shape: {input_shapes[0]} for Pad node {node.name}')

    layer['class_name'] += str(dim) + 'D'

    return layer
