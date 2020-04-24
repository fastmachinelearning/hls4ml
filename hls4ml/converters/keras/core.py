from ..keras_to_hls import parse_default_keras_layer
from ..keras_to_hls import keras_handler


@keras_handler('InputLayer')
def parse_input_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] == 'InputLayer')

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['input_shape'] = keras_layer['config']['batch_input_shape'][1:]
    if keras_layer['config']['dtype'] == 'int32':
        layer['type_name'] = 'integer_input_t'
        layer['precision'] = 'ap_int<32>'
    output_shape = keras_layer['config']['batch_input_shape']
    
    return layer, output_shape


@keras_handler('Reshape')
def parse_reshape_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer["class_name"] == 'Reshape')

    layer = parse_default_keras_layer(keras_layer, input_names)
    
    layer['target_shape'] = keras_layer['config']['target_shape']
    output_shape = input_shapes[0][:1] + keras_layer['config']['target_shape']
    
    return layer, output_shape


dense_layers = ['Dense', 'BinaryDense', 'TernaryDense']
@keras_handler(*dense_layers)
def parse_dense_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('Dense' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    
    weights_shape = data_reader.get_weights_shape(layer['name'], 'kernel')
    layer['n_in'] = weights_shape[0]
    layer['n_out'] = weights_shape[1]
    if 'Binary' in layer['class_name']:
        layer['quantize'] = 2
    elif 'Ternary' in layer['class_name']:
        layer['quantize'] = 3
    #elif layer['class_name'] == 'QDense':
    #    get_qkeras_quantization(layer, keras_layer)
    else:
        layer['quantize'] = 0
    output_shape = [input_shapes[0][0], layer['n_out']]

    return layer, output_shape


activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU']
@keras_handler(*activation_layers)
def parse_activation_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] in activation_layers)

    layer = parse_default_keras_layer(keras_layer, input_names)

    if layer['class_name'] != 'Activation':
        layer['activation'] = layer['class_name']
    
    if layer['class_name'] == 'LeakyReLU':
        layer['activ_param'] = keras_layer["config"].get('alpha', 0.3)
    elif layer['class_name'] == 'ThresholdedReLU':
        layer['activ_param'] = keras_layer["config"].get('theta', 1.)
    elif layer['class_name'] == 'ELU':
        layer['activ_param'] = keras_layer["config"].get('alpha', 1.)
    
    return layer, [shape for shape in input_shapes[0]]


@keras_handler('BatchNormalization')
def parse_batchnorm_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('BatchNormalization' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
    layer['n_in'] = in_size
    layer['n_out'] = layer['n_in']
    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) == 3:
        layer['n_filt']=input_shapes[0][2]
    elif len(input_shapes[0]) == 4:
        layer['n_filt']=input_shapes[0][3]

    return layer, [shape for shape in input_shapes[0]]
