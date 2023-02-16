from hls4ml.converters.pytorch_to_hls import pytorch_handler

reshape_layers = ['View']
@pytorch_handler(*reshape_layers)
def parse_reshape_layer(operation, layer_name, input_names, input_shapes, arguments, data_reader, config):
    assert operation == 'View'

    layer = {}
    layer['class_name'] = 'Reshape'
    layer['name'] = layer_name

    layer['target_shape'] = arguments['target_shape']
    output_shape = input_shapes[0][:1] + layer['target_shape']

    return layer, output_shape    