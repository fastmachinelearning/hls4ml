import math
from hls4ml.converters.pytorch_to_hls import pytorch_handler

@pytorch_handler('Conv1d')
def parse_conv1d_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('Conv1d' in pytorch_layer.__class__.__name__)
    
    layer = {}
    
    layer['name'] = layer_name
    layer['class_name'] = 'Conv1D'
    layer['data_format'] = 'channels_first' #Keras's default is channels_last
    
    layer['n_in'] = input_shapes[0][2] #Because channel first
    layer['n_filt'] = pytorch_layer.out_channels
    layer['n_chan'] = pytorch_layer.in_channels
    layer['filt_width'] = pytorch_layer.kernel_size[0] 
    layer['stride'] = pytorch_layer.stride[0]
    layer['pad_left'] = layer['pad_right'] = pytorch_layer.padding[0]
    layer['dilation'] = pytorch_layer.dilation[0]
    layer['out_width'] = int((layer['n_in'] + 2*layer['pad_left'] - layer['dilation']*(layer['filt_width']-1)- 1)/layer['stride'] + 1)
    
    output_shape=[input_shapes[0][0], layer['n_filt'], layer['out_width']]
    
    return layer, output_shape
    