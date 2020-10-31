import math
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler
from hls4ml.converters.keras_to_hls import compute_padding_1d
from hls4ml.converters.keras_to_hls import compute_padding_2d


@keras_handler('Conv1D')
def parse_conv1d_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('Conv1D' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    
    # weights_shape = (filter_width, n_channels, n_filters)
    weights_shape = data_reader.get_weights_shape(layer['name'], 'kernel')
    layer['n_in'] = input_shapes[0][1]
    layer['filt_width'] = weights_shape[0] # or keras_layer['config']['kernel_size']
    layer['n_chan'] = weights_shape[1]
    layer['n_filt'] = weights_shape[2] # or keras_layer['config']['filters']
    layer['stride'] = keras_layer['config']['strides'][0]
    layer['padding'] = keras_layer['config']['padding']

    (
        layer['n_out'],
        layer['pad_left'],
        layer['pad_right']
    ) = compute_padding_1d(
        layer['padding'],
        layer['n_in'],
        layer['stride'],
        layer['filt_width']
    )
    output_shape=[input_shapes[0][0], layer['n_out'], layer['n_filt']]

    return layer, output_shape


@keras_handler('Conv2D')
def parse_conv2d_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('Conv2D' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    # weights_shape = (filter_height, filter_width, n_channels, n_filters)
    weights_shape = data_reader.get_weights_shape(layer['name'], 'kernel')
    layer['in_height'] = input_shapes[0][1]
    layer['in_width'] = input_shapes[0][2]
    if layer['data_format'] == 'channels_first':
        layer['in_height'] = input_shapes[0][2]
        layer['in_width'] = input_shapes[0][3]
    layer['filt_height'] = weights_shape[0]
    layer['filt_width'] = weights_shape[1]
    layer['n_chan'] = weights_shape[2]
    layer['n_filt'] = weights_shape[3]
    layer['stride_height'] = keras_layer['config']['strides'][0]
    layer['stride_width'] = keras_layer['config']['strides'][1]
    layer['padding'] = keras_layer['config']['padding']
    
    (
        layer['out_height'],
        layer['out_width'],
        layer['pad_top'],
        layer['pad_bottom'],
        layer['pad_left'],
        layer['pad_right']
    ) = compute_padding_2d(
        layer['padding'],
        layer['in_height'],
        layer['in_width'],
        layer['stride_height'],
        layer['stride_width'],
        layer['filt_height'],
        layer['filt_width']
    )

    if layer['data_format'] == 'channels_first':
        output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
    else:
        output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]

    return layer, output_shape
