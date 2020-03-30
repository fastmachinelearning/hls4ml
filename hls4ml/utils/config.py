from __future__ import print_function
import numpy as np
import h5py
import json
import math
import six

def get_qkeras_quantization(layer, keras_layer):
    if not layer['class_name'].startswith('Q'): # Not a QKeras layer, nothing to do
        return
    kernel_quantizer = keras_layer['config']['kernel_quantizer']['class_name']
    bias_quantizer = keras_layer['config']['bias_quantizer']['class_name']

    if kernel_quantizer != bias_quantizer:
        raise Exception('Mixing quantizers within QKeras layers is not supported')
    if kernel_quantizer == 'binary':
        layer['quantize'] = 2
    elif kernel_quantizer == 'ternary':
        layer['quantize'] = 3
    else:
        raise Exception('Unsupported quantizer {} in {} layer {}'.format(kernel_quantizer, layer['class_name'], layer['name']))

def config_from_keras_model(model, granularity='model', default_precision='ap_fixed<16,6>', default_reuse_factor=1):

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    if isinstance(model, six.string_types):
        model_arch = model
    else:
        model_arch = json.loads(model.to_json())

    #print(model_arch)

    #Define supported laers
    core_layers = ['InputLayer', 'Dropout', 'Flatten', 'Reshape']
    dense_layers = ['Dense', 'BinaryDense', 'TernaryDense']
    conv_layers = ['Conv1D', 'Conv2D', 'BinaryConv2D']
    pooling_layers = ['MaxPooling1D', 'MaxPooling2D', 'AveragePooling1D', 'AveragePooling2D']
    norm_layers = ['BatchNormalization']
    activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU']
    merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate']
    qkeras_layers = ['QDense', 'QActivation', 'QConv1D', 'QConv2D']
    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten']
    #All supported layers
    supported_layers = core_layers + dense_layers + conv_layers + pooling_layers + norm_layers + activation_layers + merge_layers + qkeras_layers + skip_layers

    keras_layer_config = None
    if model_arch['class_name'] == 'Sequential':
        print('Interpreting Sequential')
        keras_layer_config = model_arch['config']
        if 'layers' in keras_layer_config: # Newer Keras versions have 'layers' in 'config' key
            keras_layer_config = keras_layer_config['layers']
        # Sequential doesn't have InputLayer
        input_layer = {}
        input_layer['name'] = 'input1'
        input_layer['class_name'] = 'InputLayer'
        layer_list.append(input_layer)
        print('Input shape:', input_layer['input_shape'])
    elif model_arch['class_name'] == 'Model':
        print('Interpreting Model')
        keras_layer_config = model_arch['config']['layers']

    print('Topology:')
    for keras_layer in keras_layer_config:
        if keras_layer['class_name'] not in supported_layers:
            raise Exception('ERROR: Unsupported layer type: {}'.format(keras_layer['class_name']))
        if keras_layer['class_name'] in skip_layers:
            continue

        #Dictionary to fill in and append to layer_list
        layer = {}

        #Extract name for finding weights and biases
        layer['name'] = keras_layer['config']['name']
        layer['class_name'] = keras_layer['class_name']
        layer['config'] = keras_layer['config']

        print('Layer name: {}, layer type: {}'.format(layer['name'], layer['class_name']))
        layer_list.append( layer )
        if 'activation' in layer['config'] and layer['class_name'] not in activation_layers + qkeras_layers:
            act_layer = {}
            act_layer['name'] = layer['name'] + '_' + layer['config']['activation']
            act_layer['class_name'] = 'Activation'
            print('  -> Activation ({}), layer name: {}'.format(layer['config']['activation'], layer['name']))
            layer_list.append(act_layer)


    def make_layer_config(layer):
        layer_config = {}
        if layer['class_name'] in dense_layers + conv_layers:
            layer_config['Precision'] = {}
            layer_config['Precision']['weight'] = default_precision
            layer_config['Precision']['bias'] = default_precision
            layer_config['Precision']['result'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor

        elif layer['class_name'] in activation_layers:
            layer_config['Precision'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor
            layer_config['table_size'] = 1024
            layer_config['table_t'] = 'ap_fixed<18,8>'
        
        elif layer['class_name'] in norm_layers:
            layer_config['Precision'] = {}
            layer_config['Precision']['scale'] = default_precision
            layer_config['Precision']['bias'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor
        
        else:
            layer_config['Precision'] = default_precision
        
        return layer_config

    config = {}

    if granularity.lower() == 'model':
        model_config = {}
        model_config['Precision'] = default_precision
        model_config['ReuseFactor'] = default_reuse_factor
        model_config['Strategy'] = 'Latency'
        #model_config['Compression'] = False
        #model_config['Trace'] = False
        
        config['Model'] = model_config
    
    elif granularity.lower() == 'type':
        type_config = {}
        for layer in layer_list:
            if layer['class_name'] in type_config or layer['class_name'] == 'InputLayer':
                continue
            layer_config = make_layer_config(layer)
            type_config[layer['class_name']] = layer_config
        
        config['LayerType'] = type_config

    elif granularity.lower() == 'name':
        name_config = {}
        for layer in layer_list:
            if layer['class_name'] == 'InputLayer': # Skip INputLayer
                continue
            layer_config = make_layer_config(layer)
            name_config[layer['name']] = layer_config
        
        config['LayerName'] = name_config

    else:
        raise Exception('Invalid configuration granularity specified, expected "model", "type" or "name" got "{}"'.format(granularity))

    return config
