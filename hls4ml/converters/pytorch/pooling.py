from hls4ml.converters.pytorch_to_hls import get_call_arg, pytorch_handler
from hls4ml.converters.utils import compute_padding_1d_pytorch, compute_padding_2d_pytorch, parse_data_format

pooling_layers = ['MaxPool1d', 'MaxPool2d', 'AvgPool1d', 'AvgPool2d']


@pytorch_handler(*pooling_layers)
def parse_pooling_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'Pool' in operation or 'pool' in operation

    layer = {}

    if operation == 'MaxPool1d':
        layer['class_name'] = 'MaxPooling1D'
    if operation == 'MaxPool2d':
        layer['class_name'] = 'MaxPooling2D'
    if operation == 'AvgPool1d':
        layer['class_name'] = 'AveragePooling1D'
    if operation == 'AvgPool2d':
        layer['class_name'] = 'AveragePooling2D'

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['data_format'] = 'channels_first'  # Pytorch default (can't change)

    # Read the layer options, from the module or from the (positional or keyword) call arguments.
    # Call signature: (input, kernel_size, stride, padding, dilation, ceil_mode) for max pooling and
    # (input, kernel_size, stride, padding, ceil_mode, count_include_pad[, divisor_override]) for average pooling.
    if node.op == 'call_module':
        kernel_size = class_object.kernel_size
        stride = class_object.stride
        padding = class_object.padding
        ceil_mode = bool(class_object.ceil_mode)
        dilation = getattr(class_object, 'dilation', 1)  # only max pooling has it
        count_include_pad = bool(getattr(class_object, 'count_include_pad', True))
        divisor_override = getattr(class_object, 'divisor_override', None)  # only AvgPool2d has it
    else:
        kernel_size = get_call_arg(node, 1, 'kernel_size')
        stride = get_call_arg(node, 2, 'stride', None)
        padding = get_call_arg(node, 3, 'padding', 0)
        if 'Max' in operation:
            dilation = get_call_arg(node, 4, 'dilation', 1)
            ceil_mode = bool(get_call_arg(node, 5, 'ceil_mode', False))
            count_include_pad = True
            divisor_override = None
        else:
            dilation = 1
            ceil_mode = bool(get_call_arg(node, 4, 'ceil_mode', False))
            count_include_pad = bool(get_call_arg(node, 5, 'count_include_pad', True))
            divisor_override = get_call_arg(node, 6, 'divisor_override', None) if operation == 'AvgPool2d' else None
    if stride is None:
        # if stride is not set it defaults to the kernel size
        stride = kernel_size

    if ceil_mode:
        raise NotImplementedError(f'Layer {layer_name}: pooling with ceil_mode=True is not supported.')
    if any(d != 1 for d in (dilation if isinstance(dilation, (tuple, list)) else (dilation,))):
        raise NotImplementedError(f'Layer {layer_name}: pooling with dilation != 1 is not supported.')
    if divisor_override is not None:
        raise NotImplementedError(f'Layer {layer_name}: average pooling with divisor_override is not supported.')

    layer['count_pad'] = count_include_pad

    if int(layer['class_name'][-2]) == 1:
        (*_, layer['n_in'], layer['n_filt']) = parse_data_format(input_shapes[0], layer['data_format'])

        layer['pool_width'] = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size
        layer['stride_width'] = stride[0] if isinstance(stride, (tuple, list)) else stride
        if isinstance(padding, (tuple, list)):
            padding = padding[0]

        if padding == 0:  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
            layer['padding'] = 'valid'
        else:  # Only 'valid' and 'same' padding are available in Keras
            layer['padding'] = 'same'

        (layer['n_out'], layer['pad_left'], layer['pad_right']) = compute_padding_1d_pytorch(
            padding, layer['n_in'], layer['stride_width'], layer['pool_width'], 1
        )

        if layer['data_format'] == 'channels_last':
            output_shape = [input_shapes[0][0], layer['n_out'], layer['n_filt']]
        elif layer['data_format'] == 'channels_first':
            output_shape = [input_shapes[0][0], layer['n_filt'], layer['n_out']]

    elif int(layer['class_name'][-2]) == 2:
        (*_, layer['in_height'], layer['in_width'], layer['n_filt']) = parse_data_format(
            input_shapes[0], layer['data_format']
        )

        if isinstance(kernel_size, (tuple, list)):
            layer['pool_height'] = kernel_size[0]
            layer['pool_width'] = kernel_size[1]
        else:
            layer['pool_height'] = kernel_size
            layer['pool_width'] = kernel_size

        if isinstance(stride, (tuple, list)):
            layer['stride_height'] = stride[0]
            layer['stride_width'] = stride[1]
        else:
            layer['stride_height'] = stride
            layer['stride_width'] = stride

        if not isinstance(padding, (tuple, list)):
            padding = [padding, padding]

        if all(x == 0 for x in padding):  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
            layer['padding'] = 'valid'
        else:  # Only 'valid' and 'same' padding are available in Keras
            layer['padding'] = 'same'

        (
            layer['out_height'],
            layer['out_width'],
            layer['pad_top'],
            layer['pad_bottom'],
            layer['pad_left'],
            layer['pad_right'],
        ) = compute_padding_2d_pytorch(
            padding,
            layer['in_height'],
            layer['in_width'],
            layer['stride_height'],
            layer['stride_width'],
            layer['pool_height'],
            layer['pool_width'],
            1,
            1,
        )

        if layer['data_format'] == 'channels_last':
            output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]
        elif layer['data_format'] == 'channels_first':
            output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]

    return layer, output_shape
