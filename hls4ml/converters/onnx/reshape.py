from hls4ml.converters.onnx_to_hls import onnx_handler


@onnx_handler('Transpose')
def parse_transpose_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Transpose'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    perm = [list(i.ints) for i in node.attribute][0]  # This will get something like [[a,b,c]][0] = [a,b,c]
    layer['perm'] = [x - 1 for x in perm[1:]]  # Ignore the batch dimension in ONNX, and adjust the perm indexing

    return layer


@onnx_handler('Reshape')
def parse_reshape_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    return layer


@onnx_handler('Flatten')
def parse_flatten_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['target_shape'] = [-1]  # does not contain batch dimension

    return layer
