from hls4ml.converters.keras_to_hls import keras_handler, parse_default_keras_layer
from hls4ml.model.types import BinaryQuantizer, IntegerPrecisionType, TernaryQuantizer


@keras_handler('InputLayer')
def parse_input_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] == 'InputLayer'

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['input_shape'] = keras_layer['config']['batch_input_shape'][1:]

    dtype = keras_layer['config']['dtype']
    if dtype.startswith('int') or dtype.startswith('uint'):
        layer['type_name'] = 'integer_input_t'
        width = int(dtype[dtype.index('int') + 3 :])
        signed = not dtype.startswith('u')
        layer['precision'] = IntegerPrecisionType(width=width, signed=signed)
    # elif bool, q[u]int, ...

    output_shape = keras_layer['config']['batch_input_shape']

    return layer, output_shape


dense_layers = ['Dense', 'BinaryDense', 'TernaryDense']


@keras_handler(*dense_layers)
def parse_dense_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'Dense' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    weights_shape = data_reader.get_weights_shape(layer['name'], 'kernel')
    layer['n_in'] = weights_shape[0]
    layer['n_out'] = weights_shape[1]
    if 'Binary' in layer['class_name']:
        layer['weight_quantizer'] = BinaryQuantizer(bits=2)
        layer['bias_quantizer'] = BinaryQuantizer(bits=2)
    elif 'Ternary' in layer['class_name']:
        layer['weight_quantizer'] = TernaryQuantizer()
        layer['bias_quantizer'] = TernaryQuantizer()
    else:
        layer['weight_quantizer'] = None
        layer['bias_quantizer'] = None
    output_shape = input_shapes[0][:]
    output_shape[-1] = layer['n_out']

    return layer, output_shape


activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'ReLU']


@keras_handler(*activation_layers)
def parse_activation_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] in activation_layers

    layer = parse_default_keras_layer(keras_layer, input_names)

    if layer['class_name'] != 'Activation':
        layer['activation'] = layer['class_name']
    if layer['class_name'] == 'LeakyReLU':
        layer['activ_param'] = keras_layer['config'].get('alpha', 0.3)
    elif layer['class_name'] == 'ThresholdedReLU':
        layer['activ_param'] = keras_layer['config'].get('theta', 1.0)
    elif layer['class_name'] == 'ELU':
        layer['activ_param'] = keras_layer['config'].get('alpha', 1.0)
    elif layer['class_name'] == 'ReLU':
        layer['class_name'] = 'Activation'

    if layer['class_name'] == 'Activation' and layer['activation'] == 'softmax':
        layer['class_name'] = 'Softmax'
    if layer['class_name'] == 'Activation' and layer['activation'] == 'hard_sigmoid':
        layer['class_name'] = 'HardActivation'
    if layer['class_name'] == 'Softmax':
        layer['axis'] = keras_layer['config'].get('axis', -1)

    return layer, [shape for shape in input_shapes[0]]


@keras_handler('BatchNormalization')
def parse_batchnorm_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'BatchNormalization' in keras_layer['class_name'] or 'QConv2DBatchnorm' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
    layer['n_in'] = in_size
    layer['n_out'] = layer['n_in']
    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) == 3:
        layer['n_filt'] = input_shapes[0][2]
    elif len(input_shapes[0]) == 4:
        layer['n_filt'] = input_shapes[0][3]

    layer['use_gamma'] = keras_layer['config']['scale']
    layer['use_beta'] = keras_layer['config']['center']

    return layer, [shape for shape in input_shapes[0]]


@keras_handler('Embedding')
def parse_embedding_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'Embedding' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['n_in'] = input_shapes[0][1]
    layer['vocab_size'] = keras_layer['config']['input_dim']
    layer['n_out'] = keras_layer['config']['output_dim']

    output_shape = input_shapes[0] + [layer['n_out']]

    return layer, output_shape
