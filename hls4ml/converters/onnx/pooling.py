import math
from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_attribute, compute_pads_1d, compute_pads_2d

pool_operations = ['AveragePool', 'MaxPool']
@onnx_handler(*pool_operations)
def parse_pool_layer(reader, node, inputs_map, input_shapes, graph, config):
            
    info = layer['class_name'].replace('Pool', '')
    strides = get_onnx_attribute(node, 'strides')
    kernel_shape = get_onnx_attribute(node, 'kernel_shape')
    
    if len(current_shape) == 3: # 1D
        layer['class_name'] = info + 'Pooling1D'
        layer['stride'] = strides[0]
        layer['pool_size'] = layer['y_filt'] = kernel_shape[0]
        
        #Padding
        pads = compute_pads_1d(node, layer)
        layer['pad_left'] = pads[0]
        layer['pad_right'] = pads[1]

        if all(x == 0 for x in pads): # No padding, i.e., 'VALID' padding
            layer['n_out'] = int(math.ceil(float(layer['y_in'] - layer['y_filt'] + 1) / float(layer['stride'])))
        else:
            layer['n_out'] = int(math.ceil(float(layer['y_in']) / float(layer['stride'])))

        output_shape = [input_shapes[0][0], layer['n_filt'], layer['n_out']]
    
    elif len(current_shape) == 4: # 2D
        layer['class_name'] = info + 'Pooling2D'

        layer['n_filt'] = input_shapes[0][1]
        layer['in_height'] = input_shapes[0][2]
        layer['in_width'] = input_shapes[0][3]

        layer['stride_height'] = strides[0]
        layer['stride_width'] = strides[1]
        layer['pool_height'] = layer['filt_height'] = kernel_shape[0]
        layer['pool_width'] = layer['filt_width'] = kernel_shape[1]
        
        pads = compute_pads_2d(node, layer)
        layer['pad_top'] = pads[0]
        layer['pad_bottom'] = pads[2]
        layer['pad_left'] = pads[1]
        layer['pad_right'] = pads[3]

        if all(x == 0 for x in pads): # No padding, i.e., 'VALID' padding in Keras/Tensorflow
            layer['out_width'] = int(math.ceil(float(layer['in_width'] - layer['filt_width'] + 1) / float(layer['stride_width'])))
            layer['out_height'] = int(math.ceil(float(layer['in_height'] - layer['filt_height'] + 1) / float(layer['stride_height'])))
        else:
            layer['out_height'] = int(math.ceil(float(layer['in_height']) / float(layer['stride_height'])))
            layer['out_width'] = int(math.ceil(float(layer['in_width']) / float(layer['stride_width'])))

        layer['n_out'] = layer['out_height'] * layer['out_height'] * layer['n_filt']
        
        output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
        
        return layer, output_shape
