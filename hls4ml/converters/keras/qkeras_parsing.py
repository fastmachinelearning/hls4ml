from ..keras_to_hls import parse_default_keras_layer
from ..keras_to_hls import keras_handler

from .core import parse_dense_layer
from .convolution import parse_conv1d_layer
from .convolution import parse_conv2d_layer
from .qkeras import *

import tensorflow as tf


@keras_handler('QDense')
def parse_qdense_layer(keras_layer, input_names, input_shapes, data_reader, config):
    
    
    layer, output_shape = parse_dense_layer(keras_layer, input_names, input_shapes, data_reader, config)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    if layer['bias_quantizer'] is not None:
        layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')
    else:
        layer['bias_quantizer'] = None

    return layer, output_shape


@keras_handler('QConv1D', 'QConv2D')
def parse_qconv_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('QConv' in keras_layer['class_name'])
    
    if int(keras_layer['class_name'][-2]) == 1:
        layer, output_shape = parse_conv1d_layer(keras_layer, input_names, input_shapes, data_reader, config)
    elif int(keras_layer['class_name'][-2]) == 2:
        layer, output_shape = parse_conv2d_layer(keras_layer, input_names, input_shapes, data_reader, config)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    layer['use_bias'] = keras_layer['use_bias']
    if keras_layer['use_bias']:
        layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')
    else:
        layer['bias_quantizer'] = None

    return layer, output_shape


@keras_handler('QActivation')
def parse_qactivation_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] == 'QActivation')
    supported_activations = ['quantized_relu', 'quantized_tanh']
    print(keras_layer)
    
    layer = parse_default_keras_layer(keras_layer, input_names)

    activation_config = keras_layer['config']['activation']
    if isinstance(activation_config, str):
        quantizer_obj = get_quantizer(activation_config)
        activation_config = {}
        activation_config['class_name'] = quantizer_obj.__class__.__name__
        activation_config['config'] = quantizer_obj.get_config()

    print(activation_config)
    
    act_class = activation_config['class_name']
    if act_class not in supported_activations:
        raise Exception('Unsupported QKeras activation: {}'.format(act_class))

    layer['class_name'] = 'Activation'
    layer['activation'] = act_class.replace('quantized_', '')
    layer['bits'] = activation_config['config']['bits'] + 1
    layer['integer'] = activation_config['config']['integer'] + 1
    #TODO this needs extra work in HLS model and HLS templates

    return layer, [shape for shape in input_shapes[0]]

