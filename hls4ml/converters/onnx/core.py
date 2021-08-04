from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_attribute, get_onnx_input_name
from hls4ml.converters.utils import compute_padding_1d

@onnx_handler(*['Gemm', 'MatMul'])
def parse_gemm_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
   
    layer['class_name'] = 'Dense'
    layer['name'] = node.name
    layer['inputs'] = get_onnx_input_name(node, graph)
    
    if len(input_shapes[0]) == 3: #Batch dense, remap to pointwise Conv1D
        layer['class_name'] = 'Conv1D'
        layer['data_format'] = 'channels_last' #Batch dense in ONNX is apparently channel last

        layer['in_width']= input_shapes[0][1]
        layer['n_chan']= input_shapes[0][2]
        reader.add_input(layer['name'], node.input)

        layer['stride_width'] = layer['filt_width'] = 1
        layer['pad_left'] = layer['pad_right'] = 0
        layer['n_filt'] = reader.get_weights_data(layer['name'], 'kernel').shape[0]
        
        layer['padding'] = 'valid'

        (layer['out_width'],_,_) = compute_padding_1d(layer['padding'],
                                                      layer['in_width'],
                                                      layer['stride_width'],
                                                      layer['filt_width'])

        output_shape = [input_shapes[0][0], layer['out_width'], layer['n_filt']]
        

    else: #Normal dense
        layer['n_in'] = input_shapes[0][1]
        layer['n_out'] = next((x.type.tensor_type.shape.dim[-1].dim_value for x in graph.value_info if x.name == node.output[0]), None)

        tran_weight = get_onnx_attribute(node, 'transB', 0)
        reader.add_input(layer['name'], node.input, tran_weight)
    
        output_shape = [input_shapes[0][0], layer['n_out']]
    
    return layer, output_shape

#------------------Global paras for activations
activation_layers = ['Relu', 'Tanh', 'Sigmoid', 'LeakyRelu', 'ThresholdedRelu', 'HardSigmoid', 'Elu', 'Selu', 'PRelu', 'Softmax', 'Softsign', 'Softplus']

activation_map = {'Relu':'ReLU', 'Tanh':'Activation', 'Sigmoid':'Activation',
    'LeakyRelu':'LeakyReLU', 'ThresholdedRelu':'ThresholdedReLU', 'HardSigmoid':'Activation',
    'Elu':'ELU', 'Selu':'Activation', 'PRelu':'PReLU', 'Softmax':'Softmax', 'Softsign':'Activation', 'Softplus':'Activation'}
#---------

@onnx_handler(*activation_layers)
def parse_activation_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
    
    layer['name'] = node.name
    layer['class_name'] = activation_map[node.op_type]
    layer['activation'] = node.op_type.lower()
    layer['inputs'] = get_onnx_input_name(node, graph)
    
    if layer['class_name'] != 'Activation':
        
        if layer['class_name'] == 'Softmax':
            layer['activation'] = 'softmax'

        elif layer['class_name'] in ['ELU', 'LeakyReLU', 'ThresholdedReLU']:
            layer['activation'] = layer['class_name']
            layer['activ_param'] = get_onnx_attribute(node, 'alpha', 0.01)
        
        else:
            layer['activation'] = layer['class_name']
            layer['class_name'] = 'Activation'
       
    return layer, [shape for shape in input_shapes[0]]
    
@onnx_handler('BatchNormalization')
def parse_batchnorm_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
   
    layer['class_name'] = 'BatchNormalization'
    layer['data_format'] = 'channels_first'
    layer['name'] = node.name
    layer['inputs'] = get_onnx_input_name(node, graph)
    
    #Other attributes
    layer['epsilon'] = get_onnx_attribute(node, 'epsilon')
    layer['momentum'] = get_onnx_attribute(node, 'momentum')
            
    reader.add_input(layer['name'], node.input)
    
    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
        
    layer['n_in'] = layer['n_out'] = in_size
    
    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) > 2:
        layer['n_filt']= input_shapes[0][1] #Always channel first for onnx
    
    return layer, [shape for shape in input_shapes[0]]