from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_attribute

@onnx_handler(*['Gemm'])
def parse_gemm_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}

    layer['class_name'] = 'Dense'
    layer['name'] = node.name
    layer['inputs'] = node.input
    layer['outputs'] = node.output

    tran_weight = get_onnx_attribute(node, 'transB', 0)
    reader.add_input(layer['name'], node.input, tran_weight)

    weights_shape = input_shapes[1][:]
    layer['n_in'] = weights_shape[0]
    layer['n_out'] = weights_shape[1]

    return layer

@onnx_handler('MatMul')
def parse_matmul_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}

    layer['class_name'] = 'MatMul'
    layer['name'] = node.name
    layer['inputs'] = node.input
    layer['outputs'] = node.output

    return layer

#------------------Global paras for activations
activation_layers = ['Relu', 'Tanh', 'Sigmoid', 'LeakyRelu', 'ThresholdedRelu', 'HardSigmoid', 'Elu', 'Selu', 'PRelu', 'Softmax', 'Softsign', 'Softplus', 'Clip']

activation_map = {'Relu':'ReLU', 'Tanh':'Activation',
                'Sigmoid':'Activation', 'LeakyRelu':'LeakyReLU',
                'ThresholdedRelu':'ThresholdedReLU', 'HardSigmoid':'Activation',
                'Elu':'ELU', 'Selu':'Activation', 'PRelu':'PReLU', 'Softmax':'Softmax',
                'Softsign':'Activation', 'Softplus':'Activation', 'Clip':'Clip'}
#---------

@onnx_handler(*activation_layers)
def parse_activation_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}

    layer['name'] = node.name
    layer['class_name'] = activation_map[node.op_type]
    layer['activation'] = node.op_type.lower()
    layer['inputs'] = node.input
    layer['outputs'] = node.output

    if layer['class_name'] != 'Activation':

        if layer['class_name'] == 'Softmax':
            layer['activation'] = 'softmax'

        elif layer['class_name'] in ['ELU', 'LeakyReLU', 'ThresholdedReLU']:
            layer['activation'] = layer['class_name']
            layer['activ_param'] = get_onnx_attribute(node, 'alpha', 0.01)

        elif layer['class_name'] == 'Clip':

            clip_min_node = [x for x in graph.initializer if x.name in node.input]
            clip_min =  clip_min_node[0].float_data[0]

            #Check if it's relu or not
            if clip_min == 0.:
                layer['class_name'] = 'Activation'
                layer['activation'] = 'ReLU'
            else:
                raise Exception('Clip with min != 0 is not supported yet!')

        else:
            layer['activation'] = layer['class_name']
            layer['class_name'] = 'Activation'

    return layer

@onnx_handler('BatchNormalization')
def parse_batchnorm_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}

    layer['class_name'] = 'BatchNormalization'
    layer['simple'] = True   # ONNX uses the simpler parsing
    layer['name'] = node.name
    layer['inputs'] = node.input
    layer['outputs'] = node.output

    #Other attributes
    layer['epsilon'] = get_onnx_attribute(node, 'epsilon', 1e-05)
    # layer['momentum'] = get_onnx_attribute(node, 'momentum', 0.9)  # not used


    # reader.add_input(layer['name'], node.input)

    # in_size = 1
    # for dim in input_shapes[0][1:]:
    #     in_size *= dim

    # layer['n_in'] = layer['n_out'] = in_size

    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    else:
        raise RuntimeError("Don't yet support larger dimensions for ONNX BatchNormalization")
    # elif len(input_shapes[0]) > 2:
    #     layer['n_filt']= input_shapes[0][1] #Always channel first for onnx

    return layer

@onnx_handler('Quant')
def parse_quant_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}

    layer['class_name'] = 'Quant'
    layer['name'] = node.name
    layer['inputs'] = node.input
    layer['outputs'] = node.output

    #Other attributes
    layer['narrow'] = get_onnx_attribute(node, 'narrow')
    layer['rounding_mode'] = get_onnx_attribute(node, 'rounding_mode')
    layer['signed'] = get_onnx_attribute(node, 'signed')
    layer['output_shape'] = [shape for shape in input_shapes[0]]

    # reader.add_input(layer['name'], node.input)

    return layer
