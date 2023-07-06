from hls4ml.converters.pytorch_to_hls import pytorch_handler

concat_layers = ['cat', 'concat', 'concatenate']


@pytorch_handler(*concat_layers)
def parse_concat_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in concat_layers

    layer = {}
    layer['class_name'] = 'Concatenate'
    layer['name'] = layer_name
    layer['op'] = 'concatenate'
    layer['inputs'] = input_names

    if len(layer['inputs']) > 2:
        raise Exception('ERROR: Merging more than two tensors is not yet supported.')

    rank = len(input_shapes[0][1:])
    if rank > 3:
        raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')
    layer['op'] = layer['class_name'].lower() + f'{rank}d'
    layer['axis'] = node.kwargs.get('dim', 0)

    output_shape = input_shapes[0][:]
    output_shape[layer['axis']] += input_shapes[1][layer['axis']]

    return layer, output_shape


add_layers = ['add']
multiply_layers = ['mul', 'multiply']
subtract_layers = ['sub', 'subtract']
min_layers = ['fmin', 'minimum']
max_layers = ['fmax', 'maximum']
merge_layers = [*add_layers, *multiply_layers, *subtract_layers, *min_layers, *max_layers]


@pytorch_handler(*merge_layers)
def parse_merge_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in merge_layers

    layer = {}
    layer['class_name'] = 'Merge'
    layer['name'] = layer_name
    if operation in add_layers:
        layer['op'] = 'add'
    elif operation in multiply_layers:
        layer['op'] = 'multiply'
    elif operation in subtract_layers:
        layer['op'] = 'subtract'
    elif operation in min_layers:
        layer['op'] = 'minimum'
    elif operation in max_layers:
        layer['op'] = 'maximum'

    layer['inputs'] = input_names

    output_shape = input_shapes[0][:]

    return layer, output_shape
