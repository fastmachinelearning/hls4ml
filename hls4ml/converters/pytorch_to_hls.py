from __future__ import print_function
import numpy as np
import os
import yaml
import sys
import torch
import pickle
import re

from hls4ml.model import HLSModel

class PyTorchDataReader:
    def __init__(self, config):
        self.config = config

        if not torch.cuda.is_available():
            self.torch_model = torch.load(config['PytorchModel'], map_location=lambda storage, loc: storage)
        else:
            self.torch_model = torch.load(config['PytorchModel'])
            
        self.input_shape = config.get('InputShape')
        if self.input_shape == None:
            raise Exception('Must specify input shape ("InputShape") in config!')
        self.input_shape = self.input_shape.strip('(,)').split(',')
        self.input_shape = [int(n) for n in self.input_shape]
        if len(self.input_shape) == 1:  #add dummy channel dimension
            self.input_shape += [1]

        self.state_dict = self.torch_model.state_dict()
    
    def get_weights_data(self, layer_name, var_name):
        if var_name == 'kernel':
            var_name = 'weight'
        data = None
        if var_name in ['weight', 'bias']:
            data = self.state_dict[layer_name + '.' + var_name].numpy().transpose()

        return data
    
def get_tuple_spec(layer_spec, spec, default = None):
    r = re.search(spec+r'=\((.+?),*\)',layer_spec)
    if r is None:
        return default
    nums = r.group(1).split(',')
    return tuple([int(num) for num in nums])

def get_int_spec(layer_spec, spec, default = None):
    r = re.search(spec+r'=(\d+)',layer_spec)
    if r is None:
        return default
    return int(r.group(1))

def get_blank_spec(layer_spec, default = None):
    r = re.search(r'(\d+, )+',layer_spec)
    if r is None:
        return default
    nums = r.group(0).split(', ')[:-1]
    return [int(num) for num in nums]

def pytorch_to_hls(yamlConfig):

    ######################
    ##  Do translation
    ######################

    print('Interpreting Model')
    reader = PyTorchDataReader(yamlConfig)

    core_layers = ['Linear','Conv1d','Conv2d']
    skip_layers = ['Dropout', 'Flatten']
    activation_layers = ['ReLU', 'Sigmoid', 'Tanh', 'SELU', 'LeakyReLU', 'Softmax', 'Softplus', 'Softsign']
    pool_operations = ['MaxPool1d']
    supported_layers = core_layers + skip_layers + activation_layers + pool_operations

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []
    
    input_layer = {}
    input_layer['name'] = 'input1'
    input_layer['class_name'] = 'InputLayer'
    input_layer['input_shape'] = list(reader.input_shape)
    input_layer['n_out'] = reader.input_shape[0]
    layer_list.append(input_layer)

    current_shape = list(reader.input_shape)
    print(current_shape)

    #Loop through layers
    print('Topology:')
    modelstr = repr(reader.torch_model).split('\n')
    print(modelstr,'\n')
    for pytorch_layer in modelstr:
        layer_match = re.match(r'\((\d+)\): (\w+)\((.*)\)', pytorch_layer.strip())
        if layer_match is None:
            continue
        
        layer_idx  = layer_match.group(1)
        layer_type = layer_match.group(2)
        layer_spec = layer_match.group(3)
        in_size = current_shape[0]

        # #Dictionary to fill in and append to layer_list
        layer={}

       
        if layer_type not in supported_layers:
            raise Exception('Unsupported layer {}'.format(layer_type))
            
        if layer_type == 'Flatten':
            current_shape = [np.prod(current_shape), 1]
            print('Flattened!, new shape:', current_shape)

        if layer_type in skip_layers:
            continue

        if layer_type == 'Linear':
            layer['class_name'] = 'Dense'
            layer['name'] = layer_idx
            
            layer['n_in'] = get_int_spec(layer_spec,'in_features')
            layer['n_out'] = get_int_spec(layer_spec,'out_features')

            if None in layer.values():
                raise Exception('Unable to interpret Linear layer ({})'.format(layer_spec))
            assert in_size == layer['n_in'], 'Size mismatch between layer {} and {}. ({} != {})'.format(layer_list[-1]['name'],layer['name'],in_size,layer['n_in'])

            current_shape[0] = layer['n_out']   #internal info for next layer

        if layer_type == 'Conv1d':
            layer['class_name'] = 'Conv1D'
            layer['name'] = layer_idx
            layer['data_format'] = 'channels_first'


            layer['n_in'] = in_size  #TODO dilation
            layer['n_chan'], layer['n_filt'] = get_blank_spec(layer_spec)
            layer['filt_width'] = get_tuple_spec(layer_spec,'kernel_size')[0]
            layer['stride'] = get_tuple_spec(layer_spec,'stride')[0]
            layer['pad_left'] = layer['pad_right'] = get_tuple_spec(layer_spec,'padding',default=(0,))[0]
            layer['dilation'] = get_tuple_spec(layer_spec, 'dilation', default=(1,))[0]
            layer['n_out'] = int((layer['n_in'] + 2*layer['pad_left'] - layer['dilation']*(layer['filt_width']-1)- 1)/layer['stride'] + 1)

            if None in layer.values():
                raise Exception('Unable to interpret Conv1D layer ({})'.format(layer_spec))
            assert layer['n_chan'] == current_shape[1], 'Channel size mismatch between layer {} and {}. ({} != {})'.format(layer_list[-1]['name'],current_shape[1],in_size,layer['n_chan'])

            current_shape = [layer['n_filt'],layer['n_out']]

        if layer_type == 'Conv2d':
            layer['class_name'] = 'Conv2D'
            layer['name'] = layer_idx
            layer['data_format'] = 'channels_first'

            layer['n_in'] = in_size  #TODO dilation
            layer['n_chan'], layer['n_filt'] = get_blank_spec(layer_spec)

            layer['in_height'],layer['in_width'] = current_shape[1],current_shape[2]    #TODO check order
            layer['filt_height'],layer['filt_width'] = get_tuple_spec(layer_spec,'kernel_size')
            layer['stride_height'],layer['stride_width'] = get_tuple_spec(layer_spec,'stride')
            layer['dilation'] = get_tuple_spec(layer_spec, 'dilation', default=(1,1))[0]

            layer['pad_top'] = layer['pad_bottom'] = get_tuple_spec(layer_spec,'padding',default=(0,0))[0]
            layer['pad_left'] = layer['pad_right'] = get_tuple_spec(layer_spec,'padding',default=(0,0))[1]

            layer['out_height']  = int((layer['in_height'] + 2*layer['pad_top'] - layer['dilation']*(layer['filt_height']-1)- 1)/layer['stride_height'] + 1)
            layer['out_width']  = int((layer['in_width'] + 2*layer['pad_left'] - layer['dilation']*(layer['filt_width']-1)- 1)/layer['stride_width'] + 1)

            assert layer['n_chan'] == current_shape[0], 'Channel size mismatch between layer {} and {}. ({} != {})'.format(layer_list[-1]['name'],current_shape[1],in_size,layer['n_chan'])

            current_shape = [layer['n_filt'],layer['out_height'],layer['out_width']]


        if layer_type in activation_layers:
            layer['activation'] = layer_type.lower()
            if layer['activation'] == 'Softmax':
                layer['class_name'] = 'Softmax'
            else:
                layer['class_name'] = 'Activation'
            layer['name'] = layer['activation'] + '_' + str(layer_idx)
            layer['n_in'] = layer['n_out'] = layer_list[-1]['n_out']
            
            current_shape = current_shape   #remains unchaqnged, here for clarity
        
        if layer_type in pool_operations:
            if layer_type == 'MaxPool1d':
                layer['class_name'] = 'MaxPooling1D'
                layer['name'] = layer_idx

                layer['n_in'] = in_size
                layer['n_filt'] = current_shape[1]
                layer['pool_size'] = get_int_spec(layer_spec,'kernel_size') #for some reason, these are ints but Conv1d are tuples. Pytorch consistency?!
                layer['stride'] = get_int_spec(layer_spec,'stride')
                layer['pad_left'] = layer['pad_right'] = get_int_spec(layer_spec,'padding',default=0)
                layer['dilation'] = get_int_spec(layer_spec,'dilation',default=1)
                layer['n_out'] = int((layer['n_in'] + 2*layer['pad_left'] - layer['dilation']*(layer['pool_size']-1)- 1)/layer['stride'] + 1)

                if layer['dilation'] != 1:
                    print('Warning: dilation not yet supported by backend!')

                current_shape = [layer['n_out'],layer['n_filt']]


        #TODO reshape
        print('Layer index: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], current_shape))

        layer_list.append(layer)

    #################
    ## Generate HLS
    #################

    reader = PyTorchDataReader(yamlConfig)
    print('Creating HLS model')
    hls_model = HLSModel(yamlConfig, reader, layer_list)
    return hls_model
