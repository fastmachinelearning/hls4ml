import math
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler


pooling_layers = ['MaxPooling1D', 'MaxPooling2D', 'AveragePooling1D', 'AveragePooling2D']
@keras_handler(*pooling_layers)
def parse_pooling_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('Pooling' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    if int(layer['class_name'][-2]) == 1:
        layer['n_in']=input_shapes[0][1]
        layer['n_filt']=input_shapes[0][2]
        layer['pool_size']=keras_layer['config']['pool_size'][0]
        layer['stride']=keras_layer['config']['strides'][0]
        layer['padding']=keras_layer['config']['padding']
        if layer['padding']=='same':
            in_width = input_shapes[0][1]
            layer['n_out'] = int(math.ceil(float(in_width) / float(layer['stride'])))
            if (in_width % layer['stride'] == 0):
                pad_along_width = max(layer['pool_size'] - layer['stride'], 0)
            else:
                pad_along_width = max(layer['pool_size'] - (in_width % layer['stride']), 0)
            layer['pad_left']  = pad_along_width // 2
            layer['pad_right']  = pad_along_width - layer['pad_left']
        elif layer['padding']=='valid':
            in_width = input_shapes[0][1]
            layer['n_out'] = int(math.ceil(float(in_width - layer['pool_size'] + 1) / float(layer['stride'])))
            layer['pad_left'] = 0
            layer['pad_right'] = 0
        output_shape=[input_shapes[0][0], layer['n_out'], layer['n_filt']]
    elif int(layer['class_name'][-2]) == 2:
        layer['data_format'] = keras_layer['config'].get('data_format', 'channels_last')
        layer['in_height']=input_shapes[0][1]
        layer['in_width']=input_shapes[0][2]
        layer['n_filt']=input_shapes[0][3]
        if layer['data_format'] == 'channels_first':
            layer['in_height']=input_shapes[0][2]
            layer['in_width']=input_shapes[0][3]
            layer['n_filt']=input_shapes[0][1]
        layer['stride_height']=keras_layer['config']['strides'][0]
        layer['stride_width']=keras_layer['config']['strides'][1]
        layer['pool_height']=keras_layer['config']['pool_size'][0]
        layer['pool_width']=keras_layer['config']['pool_size'][1]
        layer['padding']=keras_layer['config']['padding']
        if layer['padding']=='same':
            #Height
            in_height = input_shapes[0][1]
            if layer['data_format'] == 'channels_first': in_height = input_shapes[0][2]
            layer['out_height'] = int(math.ceil(float(in_height) / float(layer['stride_height'])))
            if (in_height % layer['stride_height'] == 0):
                pad_along_height = max(layer['pool_height'] - layer['stride_height'], 0)
            else:
                pad_along_height = max(layer['pool_height'] - (in_height % layer['stride_height']), 0)
            layer['pad_top'] = pad_along_height // 2
            layer['pad_bottom'] = pad_along_height - layer['pad_top']
            #Width
            in_width = input_shapes[0][2]
            if layer['data_format'] == 'channels_first': in_height = input_shapes[0][3]
            layer['out_width'] = int(math.ceil(float(in_width) / float(layer['stride_width'])))
            if (in_width % layer['stride_width'] == 0):
                pad_along_width = max(layer['pool_width'] - layer['stride_width'], 0)
            else:
                pad_along_width = max(layer['pool_width'] - (in_width % layer['stride_width']), 0)
            layer['pad_left']  = pad_along_width // 2
            layer['pad_right']  = pad_along_width - layer['pad_left']
        elif layer['padding'] == 'valid':
            in_height = input_shapes[0][1]
            in_width = input_shapes[0][2]
            if layer['data_format'] == 'channels_first':
                in_height = input_shapes[0][2]
                in_width = input_shapes[0][3]
            layer['out_width'] = int(math.ceil(float(in_width - layer['pool_width'] + 1) / float(layer['stride_width'])))
            layer['out_height'] = int(math.ceil(float(in_height - layer['pool_height'] + 1) / float(layer['stride_height'])))
            layer['pad_top'] = 0
            layer['pad_bottom'] = 0
            layer['pad_left'] = 0
            layer['pad_right'] = 0
        if layer['data_format'] == 'channels_last':
            output_shape=[input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]
        elif layer['data_format'] == 'channels_first':
            output_shape=[input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
    
    return layer, output_shape
