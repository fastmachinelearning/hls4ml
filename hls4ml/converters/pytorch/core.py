from hls4ml.converters.pytorch_to_hls import pytorch_handler


# TODO: propagate use_bias info properly
# https://github.com/fastmachinelearning/hls4ml/issues/409
@pytorch_handler('Linear')
def parse_linear_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert 'Linear' in pytorch_layer.__class__.__name__

    layer = {}

    layer['class_name'] = 'Dense'
    layer['name'] = layer_name

    layer['n_in'] = pytorch_layer.in_features
    layer['n_out'] = pytorch_layer.out_features

    # Handling whether bias is used or not
    assert pytorch_layer.bias is not None, "PyTorch Linear with bias=False not yet supported"
    if pytorch_layer.bias is None:
        layer['use_bias'] = False
    else:
        layer['use_bias'] = True

    output_shape = [input_shapes[0][0], layer['n_out']]

    return layer, output_shape


# TODO: propagate parametrized activation parameters
# https://github.com/fastmachinelearning/hls4ml/issues/409
# activation_layers = ['LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'ReLU']
activation_layers = ['Softmax', 'ReLU']


@pytorch_handler(*activation_layers)
def parse_activation_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):

    layer = {}

    layer['class_name'] = pytorch_layer.__class__.__name__
    layer['activation'] = layer['class_name']
    layer['name'] = layer_name

    if layer['class_name'] == 'ReLU':
        layer['class_name'] = 'Activation'

    output_shape = input_shapes[0]

    return layer, output_shape


batchnorm_layers = ['BatchNorm2d', 'BatchNorm1d']


@pytorch_handler(*batchnorm_layers)
def parse_batchnorm_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert 'BatchNorm' in pytorch_layer.__class__.__name__

    layer = {}

    layer['class_name'] = 'BatchNormalization'
    layer['data_format'] = 'channels_first'
    layer['name'] = layer_name

    # batchnorm para
    layer['epsilon'] = pytorch_layer.eps
    layer['use_gamma'] = layer['use_beta'] = pytorch_layer.affine

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim

    layer['n_in'] = layer['n_out'] = in_size

    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) > 2:
        layer['n_filt'] = input_shapes[0][1]  # Always channel first for Pytorch

    return layer, [shape for shape in input_shapes[0]]
