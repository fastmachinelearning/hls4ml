from __future__ import print_function
import numpy as np
import os
import yaml
import sys
import torch
import pickle
import re

from hls4ml.model import HLSModel

####---------------Initial Data Reader------------------######
class PyTorchDataReader:
    def __init__(self, config):
        self.config = config

        if not torch.cuda.is_available():
            self.torch_model = torch.load(config['PytorchModel'], map_location=lambda storage, loc: storage)
        else:
            self.torch_model = torch.load(config['PytorchModel'])
        
        #Get input tensor's shape
        self.input_shape = config.get('InputShape')
        
        if self.input_shape == None:
            raise Exception('Must specify input shape ("InputShape") in config!')
        
        #Convert it to a list
        self.input_shape = self.input_shape.strip('(,)').split(',')
        self.input_shape = [None if n == 'None' else int(n) for n in self.input_shape]

        self.state_dict = self.torch_model.state_dict()
    
    def get_weights_data(self, layer_name, var_name):
        if var_name == 'kernel':
            var_name = 'weight'
        data = None
        if var_name in ['weight', 'bias']:
            data = self.state_dict[layer_name + '.' + var_name].numpy().transpose()

        return data

####---------------Layer handling------------------######
layer_handlers = {}

def register_pytorch_layer_handler(layer_name, handler_func):
    if layer_name in layer_handlers:
        raise Exception('Layer {} already registered'.format(layer_name))
    else:
        layer_handlers[layer_name] = handler_func

def get_supported_pytorch_layers():
    return list(layer_handlers.keys())

def pytorch_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function
    return decorator

####---------------Main processing function------------------######
def pytorch_to_hls(config):

    ######################
    ##  Do translation
    ######################
    
    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    print('Interpreting Model ...')
    reader = PyTorchDataReader(config)
    model = reader.torch_model

    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten', 'Sequential']
    
    #All supported layers
    supported_layers = get_supported_pytorch_layers() + skip_layers
    
    #Map inputs of skipped and split (activation) layers
    inputs_map = {}

    input_layers = None
    output_layers = None
    
    layer_config = None
    
    #Input shape tracking
    input_shapes = [list(reader.input_shape)] #In case there are multiple inputs
    print("Input Shape: ", input_shapes)
    
    #Add input layer
    input_layer = {}
    input_layer['name'] = 'input1'
    input_layer['class_name'] = 'InputLayer'
    input_layer['input_shape'] = input_shapes[0][1:]
    layer_list.insert(0, input_layer)
    
    #Output shape tracking
    output_shapes = {}
    output_shape = None
    
    #To skip first printout from the whole model class when looping
    #Can't use layer counter for this
    model_class = True
    
    #Loop through layers
    print('Topology:')
    layer_counter = 0
    for layer_name, pytorch_layer in model.named_modules():
        
        #First module is the whole model's class
        if model_class:
            model_class = False
            continue
        
        pytorch_class = pytorch_layer.__class__.__name__
        
        if pytorch_class not in supported_layers:
            raise Exception('Unsupported layer {}'.format(pytorch_class))
                
        #If not the first layer then input shape is taken from last layer's output
        if layer_counter != 0:
            input_shapes = [output_shape] #In case there are multiple inputs
        
        #Handle skipped layers
        if pytorch_class in skip_layers:
            if pytorch_class == 'Sequential': #Ignore the mother module's class name
                continue

            if pytorch_class == 'Flatten':
                output_shapes[layer_name] = [input_shapes[0][0], np.prod(input_shapes[0][1:])]
            else:
                output_shapes[layer_name] = input_shapes[0]
            continue #!!
        
        #Increment the layer counter after initial screenings
        if pytorch_class in supported_layers:
            layer_counter += 1
        
        #Process the layer
        layer, output_shape = layer_handlers[pytorch_class](pytorch_layer, layer_name, input_shapes, reader, config)

        print('Layer name: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
        layer_list.append(layer)
        
        assert(output_shape is not None)
        output_shapes[layer['name']] = output_shape
           

    #################
    ## Generate HLS
    #################
    
    print('Creating HLS model')
    hls_model = HLSModel(config, reader, layer_list)
    return hls_model
