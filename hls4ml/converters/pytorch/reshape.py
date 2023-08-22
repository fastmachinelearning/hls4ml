import numpy as np

from hls4ml.converters.pytorch_to_hls import pytorch_handler

reshape_layers = ['View']


@pytorch_handler(*reshape_layers)
def parse_reshape_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'View'

    layer = {}
    layer['class_name'] = 'Reshape'
    layer['name'] = layer_name

    layer['target_shape'] = [int(i) for i in node.args[1:]]
    # View can have -1 as one as the dimensions,
    # leaving it to us to deduce it from the other dimensions and the overall size
    if -1 in layer['target_shape']:
        size = np.prod(input_shapes[0][1:])
        for i in range(0, len(layer['target_shape'])):
            if layer['target_shape'][i] == -1:
                cl = layer['target_shape'][:]
                cl.remove(-1)
                layer['target_shape'][i] = int(size / np.prod(cl))

    output_shape = input_shapes[0][:1] + layer['target_shape']

    return layer, output_shape


@pytorch_handler('Flatten')
def parse_flatten_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'Flatten'

    layer = {}
    layer['class_name'] = 'Reshape'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    start_dim = class_object.start_dim
    end_dim = class_object.end_dim
    if end_dim + 1 == 0 or end_dim + 1 > len(input_shapes[0]):
        end_dim = len(input_shapes[0])
    else:
        end_dim = end_dim + 1

    layer['target_shape'] = (
        input_shapes[0][0:start_dim] + [np.prod(input_shapes[0][start_dim:end_dim])] + input_shapes[0][end_dim:]
    )
    output_shape = layer['target_shape']

    return layer, output_shape
