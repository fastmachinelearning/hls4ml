import math

import numpy as np

from hls4ml.converters.keras_v2_to_hls import get_weights_data, keras_handler, parse_default_keras_layer
from hls4ml.model.quantizers import BinaryQuantizer, TernaryQuantizer
from hls4ml.model.types import IntegerPrecisionType


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

    layer['weight_data'], layer['bias_data'] = get_weights_data(data_reader, layer['name'], ['kernel', 'bias'])
    layer['n_in'] = layer['weight_data'].shape[0]
    layer['n_out'] = layer['weight_data'].shape[1]
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

    if layer['activation'] == 'elu':
        layer['class_name'] = 'ELU'  # always use ELU type for elu, even if passed as activation

    if layer['class_name'] == 'LeakyReLU':
        # the name changes for version 3
        layer['activ_param'] = keras_layer['config'].get('negative_slope', keras_layer['config'].get('alpha', 0.3))
    elif layer['class_name'] == 'ThresholdedReLU':
        layer['activ_param'] = keras_layer['config'].get('theta', 1.0)
    elif layer['class_name'] == 'ELU':
        layer['activ_param'] = keras_layer['config'].get('alpha', 1.0)
    elif layer['class_name'] == 'ReLU':
        max_value = keras_layer['config'].get('max_value', None)
        if max_value is not None and max_value == float('inf'):
            max_value = None
        negative_slope = keras_layer['config'].get('negative_slope', 0.0)
        threshold = keras_layer['config'].get('threshold', 0.0)
        if max_value is not None and (negative_slope != 0.0 or threshold != 0.0):
            raise NotImplementedError(
                f'Layer {layer["name"]}: ReLU with max_value combined with threshold or negative_slope is not supported.'
            )
        if negative_slope != 0.0 and threshold != 0.0:
            raise NotImplementedError(f'Layer {layer["name"]}: ReLU must have threshold == 0 or negative_slope == 0.')
        if max_value is not None:
            layer['class_name'] = 'ClippedReLU'
            layer['activation'] = 'clippedrelu'
            layer['activ_param'] = max_value
        elif negative_slope != 0.0:
            layer['class_name'] = 'LeakyReLU'
            layer['activation'] = 'LeakyReLU'
            layer['activ_param'] = negative_slope
        elif threshold != 0.0:
            layer['class_name'] = 'ThresholdedReLU'
            layer['activation'] = 'ThresholdedReLU'
            layer['activ_param'] = threshold
        else:
            layer['class_name'] = 'Activation'
            layer['activation'] = 'relu'
    elif layer['class_name'] == 'PReLU':
        if keras_layer['config'].get('shared_axes') is not None:
            raise Exception('PReLU with shared_axes other than None is not supported in hsl4ml')
        layer['param_data'] = get_weights_data(data_reader, layer['name'], 'alpha')

    if (layer['class_name'] == 'Activation' and layer['activation'] == 'softmax') or layer['class_name'] == 'Softmax':
        layer['class_name'] = 'Softmax'
        ax = len(input_shapes[0]) - 1
        n_outer: int = math.prod(input_shapes[0][1:ax])  # type: ignore
        n_inner: int = math.prod(input_shapes[0][ax + 1 :])  # type: ignore
        layer['n_outer'] = n_outer
        layer['n_inner'] = n_inner
    if layer['class_name'] == 'Activation' and layer['activation'] == 'hard_sigmoid':
        layer['class_name'] = 'HardActivation'
    if layer['class_name'] == 'Softmax':
        layer['axis'] = keras_layer['config'].get('axis', -1)
    if layer['class_name'] == 'Activation' and layer['activation'] == 'leaky_relu':
        layer['class_name'] = 'LeakyReLU'
        # The parameter name changes for API v3; the default is different than in LeakyReLU layer
        layer['activ_param'] = keras_layer['config'].get('negative_slope', keras_layer['config'].get('alpha', 0.2))

    return layer, [shape for shape in input_shapes[0]]


@keras_handler('BatchNormalization')
def parse_batchnorm_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'BatchNormalization' in keras_layer['class_name'] or 'QConv2DBatchnorm' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    axis = keras_layer['config'].get('axis', -1)
    if isinstance(axis, (list, tuple)):
        axis = axis[0] if len(axis) == 1 else axis
    if axis not in (-1, len(input_shapes[0]) - 1):
        raise NotImplementedError(
            f'Layer {layer["name"]}: normalization along axis {axis} is not supported; only the last (channel) dimension is.'
        )

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
    if layer['use_gamma']:
        layer['gamma_data'] = get_weights_data(data_reader, layer['name'], 'gamma')
    else:
        layer['gamma_data'] = 1

    layer['use_beta'] = keras_layer['config']['center']
    if layer['use_beta']:
        layer['beta_data'] = get_weights_data(data_reader, layer['name'], 'beta')
    else:
        layer['beta_data'] = 0

    layer['mean_data'], layer['variance_data'] = get_weights_data(
        data_reader, layer['name'], ['moving_mean', 'moving_variance']
    )

    return layer, [shape for shape in input_shapes[0]]


@keras_handler('LayerNormalization')
def parse_layernorm_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'LayerNormalization' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
    layer['n_in'] = layer['n_out'] = in_size

    if not ((len(input_shapes[0])) == 3):
        raise Exception(
            'input size is not currently supported by hls4ml; '
            'only three-dimensional input (including batch dimension) is supported'
        )
    layer['seq_len'] = input_shapes[0][-2]

    axis = keras_layer['config']['axis'][0]
    if axis < 0:
        # Keras 3 serializes the default axis as -1; Keras 2 normalizes it to the positive form
        axis += len(input_shapes[0])
    if axis != 2:
        raise Exception('assigning the axis is not currently supported by hls4ml; only axis 2 is supported')
    layer['axis'] = axis

    n_norm = input_shapes[0][-1]
    if keras_layer['config'].get('scale', True):
        layer['gamma_data'] = get_weights_data(data_reader, layer['name'], 'gamma')
    else:
        layer['gamma_data'] = np.ones(n_norm)
    if keras_layer['config'].get('center', True):
        layer['beta_data'] = get_weights_data(data_reader, layer['name'], 'beta')
    else:
        layer['beta_data'] = np.zeros(n_norm)

    if keras_layer['config']['epsilon'] <= 0:
        raise Exception('epsilon must be positive')
    layer['epsilon_power_of_10'] = -round(math.log10(keras_layer['config']['epsilon']))
    if layer['epsilon_power_of_10'] <= 0:
        raise Exception('epsilon must be less than 1e-1')

    return layer, [shape for shape in input_shapes[0]]


@keras_handler('Embedding')
def parse_embedding_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'Embedding' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    if keras_layer['config'].get('mask_zero', False):
        raise NotImplementedError(f'Layer {layer["name"]}: mask_zero=True is not supported.')

    layer['n_in'] = input_shapes[0][1]
    layer['vocab_size'] = keras_layer['config']['input_dim']
    layer['n_out'] = keras_layer['config']['output_dim']

    layer['embeddings_data'] = get_weights_data(data_reader, layer['name'], 'embeddings')

    output_shape = input_shapes[0] + [layer['n_out']]

    return layer, output_shape
