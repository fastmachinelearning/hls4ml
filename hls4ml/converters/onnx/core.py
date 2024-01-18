from hls4ml.converters.onnx_to_hls import get_onnx_attribute, get_onnx_input_name, onnx_handler


@onnx_handler(*['Gemm', 'MatMul'])
def parse_gemm_layer(reader, node, inputs_map, input_shapes, graph, config):
    layer = {}

    layer['class_name'] = 'Dense'
    layer['name'] = node.name
    layer['inputs'] = get_onnx_input_name(node, graph)

    tran_weight = get_onnx_attribute(node, 'transB', 0)
    reader.add_input(layer['name'], node.input, tran_weight)

    weights_shape = reader.get_weights_data(layer['name'], 'kernel').shape
    layer['n_in'] = weights_shape[0]
    layer['n_out'] = weights_shape[1]

    output_shape = input_shapes[0][:]
    output_shape[-1] = layer['n_out']

    return layer, output_shape


# ------------------Global paras for activations
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
    'Clip',
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
    'Clip': 'Clip',
}
# ---------


@onnx_handler(*activation_layers)
def parse_activation_layer(reader, node, inputs_map, input_shapes, graph, config):
    layer = {}

    layer['name'] = node.name
    layer['class_name'] = activation_map[node.op_type]
    layer['activation'] = node.op_type.lower()
    layer['inputs'] = get_onnx_input_name(node, graph)

    if layer['class_name'] != 'Activation':
        if layer['class_name'] == 'Softmax':
            layer['activation'] = 'softmax'

        elif layer['class_name'] in ['ELU', 'LeakyReLU', 'ThresholdedReLU']:
            layer['activation'] = layer['class_name']
            layer['activ_param'] = get_onnx_attribute(node, 'alpha', 0.01)

        elif layer['class_name'] == 'Clip':
            clip_min_node = [x for x in graph.initializer if x.name in node.input]
            clip_min = clip_min_node[0].float_data[0]

            # Check if it's relu or not
            if clip_min == 0.0:
                layer['class_name'] = 'Activation'
                layer['activation'] = 'ReLU'
            else:
                raise Exception('Clip with min != 0 is not supported yet!')

        else:
            layer['activation'] = layer['class_name']
            layer['class_name'] = 'Activation'

    return layer, [shape for shape in input_shapes[0]]


@onnx_handler('BatchNormalization')
def parse_batchnorm_layer(reader, node, inputs_map, input_shapes, graph, config):
    layer = {}

    layer['class_name'] = 'BatchNormalization'
    layer['data_format'] = 'channels_first'
    layer['name'] = node.name
    layer['inputs'] = get_onnx_input_name(node, graph)

    # Other attributes
    layer['epsilon'] = get_onnx_attribute(node, 'epsilon')
    layer['momentum'] = get_onnx_attribute(node, 'momentum')

    reader.add_input(layer['name'], node.input)

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim

    layer['n_in'] = layer['n_out'] = in_size

    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) > 2:
        layer['n_filt'] = input_shapes[0][1]  # Always channel first for onnx

    return layer, [shape for shape in input_shapes[0]]
