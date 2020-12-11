import numpy as np

from hls4ml.converters.pytorch_to_hls import pytorch_handler

@pytorch_handler('Linear')
def parse_linear_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('Linear' in pytorch_layer.__class__.__name__)
    
    layer = {}
   
    layer['class_name'] = 'Dense'
    layer['name'] = layer_name
    
    layer['n_in'] = pytorch_layer.in_features
    layer['n_out'] = pytorch_layer.out_features

    output_shape = [input_shapes[0][0], layer['n_out']]
    
    return layer, output_shape


activation_layers = ['LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'ReLU']
@pytorch_handler(*activation_layers)
def parse_activation_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    
    layer = {}
    
    layer['class_name'] =  pytorch_layer.__class__.__name__
    layer['activation'] = layer['class_name']
    layer['name'] = layer_name
    
    if layer['class_name'] == 'ReLU':
        layer['class_name'] = 'Activation'
    
    output_shape=input_shapes[0]
    
    return layer, output_shape
        
    