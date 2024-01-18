import numpy as np

from hls4ml.converters.onnx_to_hls import get_onnx_input_name, onnx_handler


@onnx_handler('Transpose')
def parse_transpose_layer(reader, node, inputs_map, input_shapes, graph, config):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Transpose'
    layer['inputs'] = get_onnx_input_name(node, graph)

    perm = [list(i.ints) for i in node.attribute][0]  # This will get something like [[a,b,c]][0] = [a,b,c]
    layer['perm'] = [x - 1 for x in perm[1:]]  # Ignore the batch dimension in ONNX, and adjust the perm indexing

    output_shape = [input_shapes[0][i] for i in perm]

    return layer, output_shape


@onnx_handler('Reshape')
def parse_reshape_layer(reader, node, inputs_map, input_shapes, graph, config):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = get_onnx_input_name(node, graph)

    target_shape = list([x for x in graph.initializer if x.name == node.input[1]][0].int64_data)[1:]

    if -1 in target_shape:  # Need to infer shape for -1
        print("WARNING: Inferring -1 shape ... ")
        dummy_x = np.ones(input_shapes[0][1:])
        dummy_y = np.reshape(dummy_x, target_shape)
        target_shape = list(dummy_y.shape)

    layer['target_shape'] = target_shape
    output_shape = input_shapes[0][:1] + layer['target_shape']

    return layer, output_shape
