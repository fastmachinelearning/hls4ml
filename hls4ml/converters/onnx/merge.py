from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_attribute, get_onnx_input_name

merge_layers = ['Add', 'Sub', 'Mul', 'Average', 'Max', 'Min', 'Concat', 'Sum']
@onnx_handler(*merge_layers)
def parse_merge_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
    layer['class_name'] = node.op_type
    layer['name'] = node.name
    layer['op'] = layer['class_name'].lower()
    layer['inputs'] = get_onnx_input_name(node, graph)

    if layer['class_name'] == 'Concatenate':
        rank = len(input_shapes[0][1:])
        if rank > 3:
            raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')
        layer['op'] = layer['class_name'].lower() + '{}d'.format(rank)
        layer['axis'] = get_onnx_attribute(node, 'axis')
   
    elif layer['class_name'] ==  'Add':
        #Check if the layer is an AddBias
        for input in node.input:
            if "bias" in input:
                layer['class_name'] = 'BiasAdd'
                reader.add_input(layer['name'], node.input)
    else:
        layer['class_name'] = 'Merge'
    
    if len(layer['inputs']) > 2:
        raise Exception('ERROR: Merging more than two tensors is not yet supported.')
    
    return layer, input_shapes[0]