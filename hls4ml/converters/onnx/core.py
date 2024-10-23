import numpy as np

from hls4ml.converters.onnx_to_hls import get_onnx_attribute, onnx_handler


@onnx_handler('MatMul')
def parse_matmul_layer(node, input_names, input_shapes, graph):
    layer = {}

    layer['class_name'] = 'MatMul'
    layer['name'] = node.name
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    return layer


# TODO: repair HardSigmoid support
# https://github.com/fastmachinelearning/hls4ml/issues/409
activation_layers = [
    'Relu',
    'Tanh',
    'Sigmoid',
    'LeakyRelu',
    'ThresholdedRelu',
    'Elu',
    'Selu',
    'PRelu',
    'Softmax',
    'Softsign',
    'Softplus',
]

activation_map = {
    'Relu': 'ReLU',
    'Tanh': 'Activation',
    'Sigmoid': 'Activation',
    'LeakyRelu': 'LeakyReLU',
    'ThresholdedRelu': 'ThresholdedReLU',
    'HardSigmoid': 'Activation',
    'Elu': 'ELU',
    'Selu': 'Activation',
    'PRelu': 'PReLU',
    'Softmax': 'Softmax',
    'Softsign': 'Activation',
    'Softplus': 'Activation',
}
# ---------


@onnx_handler(*activation_layers)
def parse_activation_layer(node, input_names, input_shapes, graph):
    layer = {}

    layer['name'] = node.name
    layer['class_name'] = activation_map[node.op_type]
    layer['activation'] = node.op_type.lower()
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    if layer['class_name'] != 'Activation':
        if layer['class_name'] == 'Softmax':
            layer['activation'] = 'softmax'
            layer['axis'] = get_onnx_attribute(node, 'axis', -1)
            # because -1 is better supported than an explicit index, check if it's the same
            if layer['axis'] == len(input_shapes[0]) - 1:
                layer['axis'] = -1

        elif layer['class_name'] in ['ELU', 'LeakyReLU', 'ThresholdedReLU']:
            layer['activation'] = layer['class_name']
            layer['activ_param'] = get_onnx_attribute(node, 'alpha', 0.01)

        else:
            layer['activation'] = layer['class_name']
            layer['class_name'] = 'Activation'

    return layer


@onnx_handler('BatchNormalization')
def parse_batchnorm_layer(node, input_names, input_shapes, graph):
    layer = {}

    layer['class_name'] = 'BatchNormOnnx'
    layer['name'] = node.name
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    # Other attributes
    layer['epsilon'] = get_onnx_attribute(node, 'epsilon', 1e-05)
    # layer['momentum'] = get_onnx_attribute(node, 'momentum', 0.9)  # not used

    layer['n_in'] = layer['n_out'] = np.prod(input_shapes[0][1:])

    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) > 2:
        if node.domain != 'qonnx.custom_op.channels_last':
            raise RuntimeError("Please convert the model to channels-last format with qonnx-to-channels-last")
        layer['data_format'] = 'channels_last'  # QONNX needs to be channels-last.
        layer['n_filt'] = input_shapes[0][-1]
    else:
        raise RuntimeError(f"Unexpected input shape: {input_shapes[0]}")

    return layer


@onnx_handler('Quant')
def parse_quant_layer(node, input_names, input_shapes, graph):
    layer = {}

    layer['class_name'] = 'Quant'
    layer['name'] = node.name
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    # Other attributes
    layer['narrow'] = bool(get_onnx_attribute(node, 'narrow'))
    layer['rounding_mode'] = get_onnx_attribute(node, 'rounding_mode')
    layer['signed'] = bool(get_onnx_attribute(node, 'signed'))

    return layer
