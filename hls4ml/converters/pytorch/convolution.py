import math
from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import *

@pytorch_handler('Conv1d')
def parse_conv1d_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('Conv1d' in pytorch_layer.__class__.__name__)
    
    layer = {}
    
    layer['name'] = layer_name
    layer['class_name'] = 'Conv1D'
    
    #Input info
    (
        layer['in_width'],
        layer['n_chan']
    ) = parse_data_format(input_shapes[0], 'channels_first') #Keras's default is channels_last
    
    #Additional parameters
    layer['n_filt'] = pytorch_layer.out_channels
    layer['filt_width'] = pytorch_layer.kernel_size[0] 
    layer['stride_width'] = pytorch_layer.stride[0]
    layer['pad_left'] = layer['pad_right'] = pytorch_layer.padding[0]
    layer['dilation'] = pytorch_layer.dilation[0]
    
    #Ouput info
    layer['out_width'] = int(
        (layer['in_width'] + 2*layer['pad_left'] - layer['dilation']*(layer['filt_width']-1)- 1)/layer['stride_width'] + 1)
    
    output_shape=[input_shapes[0][0], layer['n_filt'], layer['out_width']] #Channel first as default
    
    return layer, output_shape
    