from hls4ml.converters.onnx_to_hls import get_onnx_attribute, onnx_handler

merge_layers = ['Add', 'Sub', 'Mul', 'Div', 'Average', 'Max', 'Min', 'Concat', 'Sum']

op_map = {
    'Add': 'add',
    'Sub': 'subtract',
    'Mul': 'multiply',
    'Div': 'divide',
    'Average': 'average',
    'Max': 'maximum',
    'Min': 'minimum',
    'Sum': 'add',
    'Concat': 'concat',
}


@onnx_handler(*merge_layers)
def parse_merge_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['class_name'] = node.op_type
    layer['name'] = node.name
    layer['op'] = op_map[node.op_type]
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    if layer['class_name'] == 'Concat':
        rank = len(input_shapes[0][1:])
        if rank > 3:
            raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')

        layer['class_name'] = 'Concatenate'
        layer['op'] = layer['class_name'].lower() + f'{rank}d'
        layer['axis'] = get_onnx_attribute(node, 'axis')

    else:
        layer['class_name'] = 'Merge'

    if len(layer['inputs']) > 2:
        raise Exception('ERROR: Merging more than two tensors is not yet supported.')

    return layer
