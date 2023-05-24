from hls4ml.converters.pytorch_to_hls import get_weights_data, pytorch_handler


@pytorch_handler('Linear')
def parse_linear_layer(operation, layer_name, input_names, input_shapes, arguments, data_reader, config):
    assert 'Linear' in operation

    layer = {}

    layer['class_name'] = 'Dense'
    layer['name'] = layer_name

    layer['weight_data'], layer['bias_data'] = get_weights_data(data_reader, layer['name'], ['weight', 'bias'])
    layer['n_in'] = arguments['in_features']
    layer['n_out'] = arguments['out_features']

    # Handling whether bias is used or not
    if arguments['bias'] is None:
        layer['use_bias'] = False
    else:
        layer['use_bias'] = True

    output_shape = [input_shapes[0][0], layer['n_out']]

    return layer, output_shape


activation_layers = ['Softmax', 'Relu', 'ReLU', 'LeakyReLU', 'Threshold', 'ELU', 'PReLU', 'Sigmoid']


@pytorch_handler(*activation_layers)
def parse_activation_layer(operation, layer_name, input_names, input_shapes, arguments, data_reader, config):

    layer = {}

    layer['class_name'] = operation
    layer['activation'] = layer['class_name']
    layer['name'] = layer_name

    # if layer['class_name'] != 'Activation':
    #    layer['activation'] = layer['class_name']

    if layer['class_name'] == 'Relu' or layer['class_name'] == 'ReLU' or layer['class_name'] == 'Sigmoid':
        layer['class_name'] = 'Activation'
    if layer['class_name'] == 'LeakyReLU':
        layer['activ_param'] = arguments['alpha']
    if layer['class_name'] == 'ELU':
        layer['activ_param'] = arguments['alpha']
    if layer['class_name'] == 'PReLU':
        layer['activ_param'] = arguments['alpha']
        layer['alpha_data'] = get_weights_data(data_reader, layer['name'], 'weight')
    if layer['class_name'] == 'Threshold':
        layer['activ_param'] = arguments['threshold']
        layer['class_name'] = 'ThresholdedReLU'
        layer['activation'] = 'ThresholdedReLU'

    if 'dim' in arguments:
        layer['axis'] = arguments['dim']

    output_shape = input_shapes[0]
    return layer, output_shape


batchnorm_layers = ['BatchNorm2d', 'BatchNorm1d', 'Batch_norm']


@pytorch_handler(*batchnorm_layers)
def parse_batchnorm_layer(operation, layer_name, input_names, input_shapes, arguments, data_reader, config):
    assert 'BatchNorm' in operation

    layer = {}

    layer['class_name'] = 'BatchNormalization'
    layer['data_format'] = 'channels_first'
    layer['name'] = layer_name

    # batchnorm para
    layer['epsilon'] = arguments['eps']
    layer['use_gamma'] = layer['use_beta'] = arguments["affine"]

    if layer['use_gamma']:
        layer['gamma_data'] = get_weights_data(data_reader, layer['name'], 'weight')
    else:
        layer['gamma_data'] = 1

    if layer['use_beta']:
        layer['beta_data'] = get_weights_data(data_reader, layer['name'], 'bias')
    else:
        layer['beta_data'] = 0

    layer['mean_data'], layer['variance_data'] = get_weights_data(
        data_reader, layer['name'], ['running_mean', 'running_variance']
    )

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim

    layer['n_in'] = layer['n_out'] = in_size

    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) > 2:
        layer['n_filt'] = input_shapes[0][1]  # Always channel first for Pytorch

    return layer, [shape for shape in input_shapes[0]]
