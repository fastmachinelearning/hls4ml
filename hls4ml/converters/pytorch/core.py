import math

import numpy as np

from hls4ml.converters.pytorch_to_hls import get_call_arg, pytorch_handler
from hls4ml.utils.einsum_utils import _validate_einsum_expr


@pytorch_handler('Constant')
def parse_constant_layer(operation, layer_name, node):
    assert 'Constant' in operation

    layer = {}
    layer['inputs'] = []

    layer['class_name'] = 'Constant'
    layer['name'] = layer_name

    constant = np.array(node._args)
    layer['value'] = constant
    output_shape = constant.shape

    return layer, output_shape


@pytorch_handler('Linear')
def parse_linear_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'Linear' in operation

    layer = {}

    layer['class_name'] = 'Dense'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    layer['weight_data'] = class_object.weight.data.numpy()
    if class_object.bias is not None:
        layer['bias_data'] = class_object.bias.data.numpy()
    else:
        layer['bias_data'] = None

    if class_object is not None:
        layer['n_in'] = class_object.in_features
        layer['n_out'] = class_object.out_features
    else:
        raise Exception('parsing of torch.nn.functional.linear not supported yet, please use torch.nn.Linear class')

    # Handling whether bias is used or not
    if class_object.bias is None:
        layer['use_bias'] = False
    else:
        layer['use_bias'] = True

    output_shape = input_shapes[0][:]
    output_shape[-1] = layer['n_out']

    return layer, output_shape


activation_layers = ['Softmax', 'ReLU', 'ReLU6', 'Hardtanh', 'LeakyReLU', 'Threshold', 'ELU', 'PReLU', 'Sigmoid', 'Tanh']


@pytorch_handler(*activation_layers)
def parse_activation_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    layer = {}

    layer['class_name'] = operation
    layer['activation'] = layer['class_name'].lower()
    layer['name'] = layer_name
    layer['inputs'] = input_names

    if node.op == 'call_module':
        if layer['class_name'] in ['ReLU', 'Sigmoid', 'Tanh']:
            layer['class_name'] = 'Activation'
        if layer['class_name'] == 'LeakyReLU':
            layer['activ_param'] = class_object.negative_slope
        if layer['class_name'] == 'ELU':
            layer['activ_param'] = class_object.alpha
        if layer['class_name'] == 'PReLU':
            layer['param_data'] = class_object.weight.data.numpy()
        if layer['class_name'] == 'Threshold':
            if class_object.value != 0:
                raise NotImplementedError(
                    f'Layer {layer_name}: Threshold with a replacement value other than 0 is not supported.'
                )
            layer['activ_param'] = class_object.threshold
            layer['class_name'] = 'ThresholdedReLU'
            layer['activation'] = 'ThresholdedReLU'
            if layer['activ_param'] < 0:
                raise Exception('negative threshold values not supported')
        if layer['class_name'] in ['ReLU6', 'Hardtanh']:
            min_val = class_object.min_val
            max_val = class_object.max_val
        if hasattr(class_object, 'dim'):
            layer['axis'] = class_object.dim
    else:
        if layer['class_name'] in ['ReLU', 'Sigmoid', 'Tanh']:
            layer['class_name'] = 'Activation'
        if layer['class_name'] == 'LeakyReLU':
            layer['activ_param'] = get_call_arg(node, 1, 'negative_slope', 0.01)
        if layer['class_name'] == 'ELU':
            layer['activ_param'] = get_call_arg(node, 1, 'alpha', 1.0)
        if layer['class_name'] == 'Threshold':
            if get_call_arg(node, 2, 'value', 0) != 0:
                raise NotImplementedError(
                    f'Layer {layer_name}: Threshold with a replacement value other than 0 is not supported.'
                )
            layer['activ_param'] = get_call_arg(node, 1, 'threshold')
            if layer['activ_param'] < 0:
                raise Exception('negative threshold values not supported')
            layer['class_name'] = 'ThresholdedReLU'
            layer['activation'] = 'ThresholdedReLU'
        if layer['class_name'] == 'ReLU6':
            min_val, max_val = 0.0, 6.0
        if layer['class_name'] == 'Hardtanh':
            min_val = get_call_arg(node, 1, 'min_val', -1.0)
            max_val = get_call_arg(node, 2, 'max_val', 1.0)
        if layer['class_name'] == 'Softmax':
            layer['axis'] = get_call_arg(node, 1, 'dim', None)

    if layer['class_name'] in ['ReLU6', 'Hardtanh']:
        # ReLU6 is Hardtanh(0, 6); Hardtanh clipped at 0 from below is a clipped ReLU
        if min_val != 0:
            raise NotImplementedError(f'Layer {layer_name}: Hardtanh is only supported with min_val == 0 (a clipped ReLU).')
        layer['class_name'] = 'ClippedReLU'
        layer['activation'] = 'clippedrelu'
        layer['activ_param'] = max_val

    if layer['class_name'] == 'Softmax':
        if layer.get('axis') is None:
            layer['axis'] = -1
        if layer['axis'] == len(input_shapes[0]) - 1:
            layer['axis'] = -1  # the last dimension, in its canonical form
        if 'IOType' in config:
            if config['IOType'] == 'io_stream' and layer['axis'] != -1:
                raise Exception('dim needs to be -1 for io_stream')

    output_shape = input_shapes[0]
    return layer, output_shape


batchnorm_layers = ['BatchNorm2d', 'BatchNorm1d', 'Batch_norm']


@pytorch_handler(*batchnorm_layers)
def parse_batchnorm_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'BatchNorm' in operation

    layer = {}

    layer['class_name'] = 'BatchNormalization'
    layer['data_format'] = 'channels_first'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    # batchnorm para
    if node.op == 'call_module':
        if not class_object.track_running_stats:
            raise NotImplementedError(
                f'Layer {layer_name}: BatchNorm with track_running_stats=False is not supported '
                '(it normalizes with per-batch statistics at inference time).'
            )
        layer['epsilon'] = class_object.eps
        layer['use_gamma'] = layer['use_beta'] = class_object.affine

        if layer['use_gamma']:
            layer['gamma_data'] = class_object.weight.data.numpy()
        else:
            layer['gamma_data'] = 1

        if layer['use_beta']:
            layer['beta_data'] = class_object.bias.data.numpy()
        else:
            layer['beta_data'] = 0

        layer['mean_data'] = class_object.running_mean.data.numpy()
        layer['variance_data'] = class_object.running_var.data.numpy()

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim

    layer['n_in'] = layer['n_out'] = in_size

    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) > 2:
        layer['n_filt'] = input_shapes[0][1]  # Always channel first for Pytorch

    return layer, [shape for shape in input_shapes[0]]


@pytorch_handler('LayerNorm')
def parse_layernorm_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'LayerNorm' in operation

    layer = {}

    layer['class_name'] = 'LayerNormalization'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
    layer['n_in'] = layer['n_out'] = in_size

    if not ((len(input_shapes[0])) == 3):
        raise Exception(
            f'Input shape {input_shapes[0]} is not currently supported for LayerNorm; '
            'only three-dimensional inputs (including batch dimension) are supported'
        )
    layer['seq_len'] = input_shapes[0][-2]

    layer['axis'] = 2

    # with elementwise_affine=False (or bias=False) the weights do not exist; substitute identity values
    if class_object.weight is not None:
        layer['gamma_data'] = class_object.weight.data.numpy()
    else:
        layer['gamma_data'] = np.ones(class_object.normalized_shape)
    if class_object.bias is not None:
        layer['beta_data'] = class_object.bias.data.numpy()
    else:
        layer['beta_data'] = np.zeros(class_object.normalized_shape)

    if class_object.eps <= 0:
        raise Exception('epsilon must be positive')
    layer['epsilon_power_of_10'] = -round(math.log10(class_object.eps))
    if layer['epsilon_power_of_10'] <= 0:
        raise Exception('epsilon must be less than 1e-1')

    return layer, [shape for shape in input_shapes[0]]


@pytorch_handler('einsum')
def parse_einsum_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'einsum' in operation

    layer = {}

    if len(input_names) == 1:
        input_names += input_names
        input_shapes += input_shapes
    elif len(input_names) > 2:
        raise Exception('Only einsum operations with two inputs are supported')
    layer['class_name'] = 'Einsum'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    # Need to set batch size to a real value instead of 'None'. Using '1' as dummy value
    import copy

    input_shapes_tmp = copy.deepcopy(input_shapes)
    input_shapes_tmp[0][0] = 1
    input_shapes_tmp[1][0] = 1
    layer['inp0_shape'] = tuple(input_shapes_tmp[0])
    layer['inp1_shape'] = tuple(input_shapes_tmp[1])

    layer['equation'], layer['out_shape'] = _validate_einsum_expr(node.args[0], layer['inp0_shape'], layer['inp1_shape'])

    return layer, [shape for shape in input_shapes[0]]
