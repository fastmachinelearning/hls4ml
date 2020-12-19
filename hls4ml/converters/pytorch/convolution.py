import math
from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import *

@pytorch_handler('Conv1d')
def parse_conv1d_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('Conv1d' in pytorch_layer.__class__.__name__)
    
    layer = {}
    
    layer['name'] = layer_name
    layer['class_name'] = 'Conv1D'
    layer['data_format'] = 'channels_first' #Pytorch default (can't change)
    
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
    
    #Only 'same' padding is implemented in the streaming algorithm
    if output_shape == input_shapes[0]:
        layer['padding'] = 'same'
    else:
        raise Exeption('ERROR: Only "same" padding is allowed. (Input shapes must match output shapes in Conv layers)')
      
    
    return layer, output_shape

@pytorch_handler('Conv2d')
def parse_conv2d_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('Conv2d' in pytorch_layer.__class__.__name__)
    
    layer = {}
    
    layer['name'] = layer_name
    layer['class_name'] = 'Conv2D'
    layer['data_format'] = 'channels_first' #Pytorch default (can't change)
    
    #Input info
    (
        layer['in_height'],
        layer['in_width'],
        layer['n_chan']
    ) = parse_data_format(input_shapes[0], 'channels_first') #Keras's default is channels_last
    
    #Additional parameters
    layer['n_filt'] = pytorch_layer.out_channels
    layer['filt_height'] = pytorch_layer.kernel_size[0]
    layer['filt_width'] = pytorch_layer.kernel_size[1]
    layer['stride_height'] = pytorch_layer.stride[0]
    layer['stride_width'] = pytorch_layer.stride[1]
    layer['dilation'] = pytorch_layer.dilation[0]
    layer['pad_top'] = layer['pad_bottom'] = pytorch_layer.padding[0]
    layer['pad_left'] = layer['pad_right'] = pytorch_layer.padding[1]
    
    #Ouput info
    layer['out_height']  = int((layer['in_height'] + 2*layer['pad_top'] - layer['dilation']*(layer['filt_height']-1)- 1)/layer['stride_height'] + 1)
    layer['out_width']  = int((layer['in_width'] + 2*layer['pad_left'] - layer['dilation']*(layer['filt_width']-1)- 1)/layer['stride_width'] + 1)
        
    #Only 'same' padding is implemented in the streaming algorithm
    if output_shape == input_shapes[0]:
        layer['padding'] = 'same'
    else:
        raise Exeption('ERROR: Only "same" padding is allowed. (Input shapes must match output shapes in Conv layers)')

    return layer, output_shape