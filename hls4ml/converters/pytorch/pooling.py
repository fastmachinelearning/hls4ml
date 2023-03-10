from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import compute_padding_1d, compute_padding_2d, parse_data_format

pooling_layers = ['MaxPool1d', 'Max_pool1d', 'MaxPool2d', 'Max_pool2d', 'AvgPool1d', 'Avg_pool1d', 'AvgPool2d', 'Avg_pool2d']


@pytorch_handler(*pooling_layers)
def parse_pooling_layer(operation, layer_name, input_names, input_shapes, arguments, data_reader, config):

    assert 'Pool' in operation or 'pool' in operation

    layer = {}

    if operation == 'MaxPool1d' or operation == 'Max_pool1d':
        layer['class_name'] = 'MaxPooling1D'
    if operation == 'MaxPool2d' or operation == 'Max_pool2d':
        layer['class_name'] = 'MaxPooling2D'
    if operation == 'AvgPool1d' or operation == 'Avg_pool1d':
        layer['class_name'] = 'AveragePooling1D'
    if operation == 'AvgPool2d' or operation == 'Avg_pool2d':
        layer['class_name'] = 'AveragePooling2D'

    layer['name'] = layer_name
    layer['data_format'] = 'channels_first'  # Pytorch default (can't change)

    if int(layer['class_name'][-2]) == 1:
        (layer['n_in'], layer['n_filt']) = parse_data_format(input_shapes[0], layer['data_format'])

        layer['pool_width'] = arguments['kernel_size']
        layer['stride_width'] = arguments['stride']

        if arguments['padding'] == 0:  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
            layer['padding'] = 'valid'
        else:  # Only 'valid' and 'same' padding are available in Keras
            layer['padding'] = 'same'

        (layer['n_out'], layer['pad_left'], layer['pad_right']) = compute_padding_1d(
            layer['padding'], layer['n_in'], layer['stride_width'], layer['pool_width']
        )

        if layer['data_format'] == 'channels_last':
            output_shape = [input_shapes[0][0], layer['n_out'], layer['n_filt']]
        elif layer['data_format'] == 'channels_first':
            output_shape = [input_shapes[0][0], layer['n_filt'], layer['n_out']]

    elif int(layer['class_name'][-2]) == 2:
        (layer['in_height'], layer['in_width'], layer['n_filt']) = parse_data_format(input_shapes[0], layer['data_format'])
        layer['stride_height'] = arguments['stride'][0]
        layer['stride_width'] = arguments['stride'][1]
        layer['pool_height'] = arguments['kernel_size'][0]
        layer['pool_width'] = arguments['kernel_size'][1]

        if all(x == 0 for x in arguments['padding']):  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
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
        ) = compute_padding_2d(
            layer['padding'],
            layer['in_height'],
            layer['in_width'],
            layer['stride_height'],
            layer['stride_width'],
            layer['pool_height'],
            layer['pool_width'],
        )

        if layer['data_format'] == 'channels_last':
            output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]
        elif layer['data_format'] == 'channels_first':
            output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]

    return layer, output_shape
