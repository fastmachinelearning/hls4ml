from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_input_name
import numpy as np

reshape_layers = ['Reshape', 'Squeeze', 'Unsqueeze'] #Fold Squeeze and Unsqueeze into reshape
@onnx_handler(*reshape_layers)
def parse_reshape_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'

    layer['inputs'] = get_onnx_input_name(node, graph)

    # if node.op_type == 'Reshape':
    #     layer['target_shape'] = node.shape
    # elif node.op_type == 'Squeeze':
    #     layer['target_shape'] = 
    # elif node.op_type == 'Unsqueeze':
    #     output_shape = input_shapes[0][:1] + layer['target_shape']
    
    return layer, output_shape


@onnx_handler('Transpose')
def parse_transpose_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Transpose'

    layer['inputs'] = get_onnx_input_name(node, graph)
    
    perm = [list(i.ints) for i in node.attribute][0] #This will get something like [[a,b,c]][0] = [a,b,c]    
    layer['perm'] = perm
    
    output_shape = [input_shapes[0][i] for i in perm]
    
    return layer, output_shape

