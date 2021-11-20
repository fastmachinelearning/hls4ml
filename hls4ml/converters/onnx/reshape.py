from hls4ml.converters.onnx_to_hls import onnx_handler
import numpy as np

@onnx_handler('Transpose')
def parse_transpose_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Transpose'
    layer['inputs'] = node.input
    layer['outputs'] = node.output

    perm = [list(i.ints) for i in node.attribute][0] #This will get something like [[a,b,c]][0] = [a,b,c]
    layer['perm'] = [x - 1 for x in perm[1:]] #Ignore the batch dimension in ONNX, and adjust the perm indexing

    return layer

@onnx_handler('Reshape')
def parse_reshape_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = node.input
    layer['outputs'] = node.output

    return layer

@onnx_handler('Flatten')
def parse_reshape_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = node.input
    layer['outputs'] = node.output
    layer['target_shape'] = [1, -1]

    return layer