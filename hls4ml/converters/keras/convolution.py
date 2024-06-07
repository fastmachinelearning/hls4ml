from hls4ml.converters.keras_to_hls import get_weights_data, keras_handler, parse_default_keras_layer
from hls4ml.converters.utils import compute_padding_1d, compute_padding_2d, parse_data_format


@keras_handler('Conv1D', 'SeparableConv1D', 'DepthwiseConv1D')
def parse_conv1d_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'Conv1D' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    (layer['in_width'], layer['n_chan']) = parse_data_format(input_shapes[0], layer['data_format'])

    if layer['class_name'] in ['Conv1D', 'QConv1D']:
        layer['weight_data'] = get_weights_data(data_reader, layer['name'], 'kernel')
    elif layer['class_name'] in ['SeparableConv1D', 'QSeparableConv1D']:
        layer['depthwise_data'], layer['pointwise_data'] = get_weights_data(
            data_reader, layer['name'], ['depthwise_kernel', 'pointwise_kernel']
        )
    else:  # DepthwiseConv1D
        layer['depthwise_data'] = get_weights_data(data_reader, layer['name'], 'depthwise_kernel')

    layer['bias_data'] = get_weights_data(data_reader, layer['name'], 'bias')

    if 'filters' in keras_layer['config']:
        layer['n_filt'] = keras_layer['config']['filters']
    else:
        layer['n_filt'] = layer['n_chan']
    layer['filt_width'] = keras_layer['config']['kernel_size'][0]
    layer['stride_width'] = keras_layer['config']['strides'][0]
    layer['padding'] = keras_layer['config']['padding']

    (layer['out_width'], layer['pad_left'], layer['pad_right']) = compute_padding_1d(
        layer['padding'], layer['in_width'], layer['stride_width'], layer['filt_width']
    )

    if layer['data_format'] == 'channels_last':
        output_shape = [input_shapes[0][0], layer['out_width'], layer['n_filt']]
    elif layer['data_format'] == 'channels_first':
        output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_width']]

    return layer, output_shape


@keras_handler('Conv2D', 'SeparableConv2D', 'DepthwiseConv2D')
def parse_conv2d_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'Conv2D' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    (layer['in_height'], layer['in_width'], layer['n_chan']) = parse_data_format(input_shapes[0], layer['data_format'])

    if layer['class_name'] in ['Conv2D', 'QConv2D', 'QConv2DBatchnorm']:
        layer['weight_data'] = get_weights_data(data_reader, layer['name'], 'kernel')
    elif layer['class_name'] in ['SeparableConv2D', 'QSeparableConv2D']:
        layer['depthwise_data'], layer['pointwise_data'] = get_weights_data(
            data_reader, layer['name'], ['depthwise_kernel', 'pointwise_kernel']
        )
    else:  # DepthwiseConv2D
        layer['depthwise_data'] = get_weights_data(data_reader, layer['name'], 'depthwise_kernel')

    layer['bias_data'] = get_weights_data(data_reader, layer['name'], 'bias')

    if 'filters' in keras_layer['config']:
        layer['n_filt'] = keras_layer['config']['filters']
    else:
        layer['n_filt'] = layer['n_chan']
    layer['filt_height'] = keras_layer['config']['kernel_size'][0]
    layer['filt_width'] = keras_layer['config']['kernel_size'][1]
    layer['stride_height'] = keras_layer['config']['strides'][0]
    layer['stride_width'] = keras_layer['config']['strides'][1]
    layer['padding'] = keras_layer['config']['padding']

    (
        layer['out_height'],
        layer['out_width'],
        layer['pad_top'],
        layer['pad_bottom'],
        layer['pad_left'],
        layer['pad_right'],
    ) = compute_padding_2d(
        layer['padding'],
        layer['in_height'],
        layer['in_width'],
        layer['stride_height'],
        layer['stride_width'],
        layer['filt_height'],
        layer['filt_width'],
    )

    if layer['data_format'] == 'channels_first':
        output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
    else:
        output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]

    return layer, output_shape
