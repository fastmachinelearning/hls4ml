from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_attribute

merge_layers = ['Add', 'Sub', 'Mul', 'Div', 'Average', 'Max', 'Min', 'Concat', 'Sum']
@onnx_handler(*merge_layers)
def parse_merge_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}
    layer['class_name'] = node.op_type
    layer['name'] = node.name
    layer['op'] = layer['class_name'].lower()
    layer['inputs'] = node.input
    layer['outputs'] = node.output

    if layer['class_name'] == 'Concat':
        rank = len(input_shapes[0][1:])
        if rank > 3:
            raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')

        layer['class_name'] = 'Concatenate'
        layer['op'] = layer['class_name'].lower() + '{}d'.format(rank)
        layer['axis'] = get_onnx_attribute(node, 'axis')

        # #Calculate output shape
        # new_dim = sum([x.type.tensor_type.shape.dim[layer['axis']].dim_value for x in graph.value_info if x.name in node.input])
        # output_shape[layer['axis']] = new_dim

    elif layer['class_name'] ==  'Add':
        #Check if the layer is an AddBias
        for input in node.input:
            if "bias" in input:
                layer['class_name'] = 'BiasAdd'
                # # Should the line below really be replaced with the one below it?
                # # Going to assume so
                # reader.add_input(layer['name'], node.input)
                reader.add_input(layer['name'], input)

        if layer['class_name'] ==  'Add':
            # If it wasn't changed, just make it a merge node
            layer['class_name'] = 'Merge'

    else:
        layer['class_name'] = 'Merge'

    if len(layer['inputs']) > 2:
        raise Exception('ERROR: Merging more than two tensors is not yet supported.')

    return layer