from __future__ import print_function
import numpy as np
import os
import yaml
import sys
import torch
import pickle
import re

from hls4ml.model import ModelGraph

class PyTorchModelReader(object):
    """
    Pytorch data reader to be used for extracting relevant information during conversion.
    """
    def __init__(self, config):
        self.torch_model = config['PytorchModel']
        self.state_dict = self.torch_model.state_dict()
        self.input_shape = config['InputShape']
    
    def get_weights_data(self, layer_name, var_name):
        """Get weights data from layers.
        
        The hls layer classes are based on Keras's default parameters.
        Thus, this function will also need to account for some differences
        between Keras and Pytorch terminology.
        
        Parameters
        ----------
        layer_name : string
            layer's name in the ONNX model
        var_name : string
            variable to be extracted

        Returns
        -------
        data : numpy array
            extracted weights data 
        
        """
        
        data = None
        
        #Parameter mapping from pytorch to keras
        torch_paramap = {
        #Conv
        'kernel': 'weight', 
        #Batchnorm
        'gamma': 'weight',
        'beta': 'bias',
        'moving_mean':'running_mean',
        'moving_variance': 'running_var'}
            
        if var_name not in list(torch_paramap.keys()) + ['weight', 'bias']:
            raise Exception('Pytorch parameter not yet supported!')
        
        elif var_name in list(torch_paramap.keys()):
            var_name = torch_paramap[var_name]
            
        data = self.state_dict[layer_name + '.' + var_name].numpy().transpose() #Look at transpose when systhesis produce lousy results. Might need to remove it.
        
        return data
    
class PyTorchFileReader(PyTorchModelReader): #Inherit get_weights_data method
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
        
        return data

####----------------------Layer handling---------------------######
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
####----------------------------------------------------------------

def pytorch_to_hls(config):
    """ Convert Pytorch model to hls model from configuration.
    
    Parameters
    ----------
    config: dict
        pytorch configuration from yaml file or passed through API.
        
    Returns
    -------
    ModelGraph : hls4ml model object.
    
    Notes
    -----
    Only sequential pytorch models are supported for now.
    """
    
    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    print('Interpreting Model ...')
    
    reader = PyTorchFileReader(config) if isinstance(config['PytorchModel'],str) else PyTorchModelReader(config)
    input_shapes = [list(reader.input_shape)]
        
    model = reader.torch_model
    
    #dict of layer objects in non-traced form for access lateron 
    children = { c[0]: c[1] for c in model.named_children() }
    #use symbolic_trace to get a full graph of the model
    from torch.fx import symbolic_trace
    traced_model = symbolic_trace(model)

    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten', 'Sequential']
    
    #All supported layers
    supported_layers = get_supported_pytorch_layers() + skip_layers
 
    #Map inputs of skipped and split (activation) layers
    inputs_map = {}

    input_layers = None
    output_layers = None
    
    layer_config = None
    
    #Output shape tracking
    output_shapes = {}
    output_shape = None
    
    #Loop through layers
    print('Topology:')
    layer_counter = 0


    n_inputs = 0

    for node in traced_model.graph.nodes:

        if layer_counter != 0:
            input_shapes = [output_shape] #In case there are multiple inputs

        if node.op == 'call_module':
            pytorch_class = children[node.target].__class__.__name__

            if pytorch_class not in supported_layers:
                raise Exception('Unsupported layer {}'.format(pytorch_class))

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

            #parse info from class object

            input_names = tuple([str(i) for i in node.args])
            arguments = {}

            #for Softmax (and probably others)
            if hasattr(children[node.target], 'dim'):
                arguments['dim'] = children[node.target].dim
            #for Linear layer
            if hasattr(children[node.target], 'in_features'):
                arguments['in_features'] =  children[node.target].in_features
            if hasattr(children[node.target], 'out_features'):
                arguments['out_features'] = children[node.target].out_features
            if hasattr(children[node.target], 'bias'):
                arguments['bias'] = children[node.target].bias
            #for Conv/Pool layers
            if hasattr(children[node.target], 'out_channels'):
                arguments['out_channels'] =  children[node.target].out_channels
            if hasattr(children[node.target], 'kernel_size'):
                arguments['kernel_size'] = children[node.target].kernel_size
            if hasattr(children[node.target], 'stride'):
                arguments['stride'] = children[node.target].stride
            if hasattr(children[node.target], 'dilation'):
                arguments['dilation'] = children[node.target].dilation
            if hasattr(children[node.target], 'padding'):
                arguments['padding'] = children[node.target].padding
            #for BatchNorm layers    
            if hasattr(children[node.target], 'eps'):
                arguments['eps'] = children[node.target].eps
            

            layer_name = node.name
            
            #Process the layer
            layer, output_shape = layer_handlers[pytorch_class](pytorch_class,layer_name,input_names, input_shapes, arguments,reader, config)          

            #Process the layer
            #layer, output_shape = layer_handlers[pytorch_class](children[node.target], node.target, input_shapes, reader, config)

            print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
            layer_list.append(layer)
            
            assert(output_shape is not None)
            output_shapes[layer['name']] = output_shape   

            layer_counter += 1     

        if node.op == 'placeholder':
            #'placeholder' indicates the input layer. Only one allowed, throw exceptions if there are more
            if n_inputs > 0:
                raise Exception("Only one input to forward function allowed")
            input_layer = {}
            input_layer['name'] = node.name
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = input_shapes[0][1:]
            layer_list.insert(0, input_layer)
            
            n_inputs += 1

            

        if node.op == 'call_function':
            #Function calls in the graph have to be transformed to layers known to hls4ml
            
            #operations that appear repeatedly have '_n' appended to their name for the nth repetition
            if node.name.split("_")[-1].isdigit():
                operation = "_".join(node.name.split("_")[:-1]).capitalize()
            else:    
                operation = node.name.capitalize()

            #only a limited number of functions are supported
            if operation not in supported_layers:
                raise Exception('Unsupported function {}'.format(operation))

            layer_counter += 1

            #need a copy because kwargs are immutable
            arguments = {}
            for key in node.kwargs:
                arguments[key] = node.kwargs[key]    
            layer_name = node.name

            #arguments of pooling layers need some massaging
            if 'pool' in operation:                
                input_names = str(node.args[0])
                arguments['kernel_size'] = int(node.args[1])
                if '2d' in operation and not type(arguments['kernel_size']) is tuple:
                    arguments['kernel_size'] = [arguments['kernel_size'],arguments['kernel_size']]
                if '2d' in operation and not type(arguments['padding']) is tuple:
                    arguments['padding'] = [arguments['padding'],arguments['padding']]
                if arguments['stride'] == None:
                    arguments['stride'] = arguments['kernel_size']
            else:    
                input_names = tuple([str(i) for i in node.args])

            #Process the layer
            layer, output_shape = layer_handlers[operation](operation,layer_name,input_names, input_shapes, arguments,reader, config)

            print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
            layer_list.append(layer)

            assert(output_shape is not None)
            output_shapes[layer['name']] = output_shape

        if node.op == 'get_attr':
            #this doesn't actually work, can't have multiple input layers. Need to find other way to get this tensor into the graph
            if not "." in node.target:
                obj =  getattr(model,node.name)
            else:
                obj = getattr(children[node.target.split('.')[0],node.name])

            input_shapes.append([obj.size(dim=0),obj.size(dim=1)])    
            input_layers[node.name] = {}
            input_layers[node.name]['name'] = node.name
            input_layers[node.name]['class_name'] = 'InputLayer'
            input_layers[node.name]['input_shape'] = input_shapes[-1]
            layer_counter += 1

        if node.op == 'call_method':
            #Method calls in the graph have to be transformed to layers known to hls4ml
            
            #operations that appear repeatedly have '_n' appended to their name for the nth repetition
            if node.name.split("_")[-1].isdigit():
                operation = "_".join(node.name.split("_")[:-1]).capitalize()
            else:    
                operation = node.name.capitalize()

            #only a limited number of functions are supported
            if operation not in supported_layers:
                raise Exception('Unsupported function {}'.format(operation))

            layer_counter += 1

            #need a copy because kwargs are immutable
            arguments = {}
            for key in node.kwargs:
                arguments[key] = node.kwargs[key]    
            layer_name = node.name
        
            if 'View' in operation:                
                input_names = str(node.args[0])
                arguments['target_shape'] = [int(i) for i in node.args[1:]]
                #View can have -1 as one as the dimensions, leaving it to us to deduce it from the other dimensions and the overall size
                if -1 in arguments['target_shape']:
                    size = np.prod(input_shapes[0][1:])
                    for i in range(0,len(arguments['target_shape'])):
                        if arguments['target_shape'][i] == -1:
                            cl = arguments['target_shape'][:]
                            cl.remove(-1)
                            arguments['target_shape'][i] = int(size/np.prod(cl))

            else:    
                input_names = tuple([str(i) for i in node.args])
            #Process the layer
            layer, output_shape = layer_handlers[operation](operation,layer_name,input_names, input_shapes, arguments,reader, config)

            print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
            layer_list.append(layer)

            assert(output_shape is not None)
            output_shapes[layer['name']] = output_shape

    #################
    ## Generate HLS
    #################
    
    print('Creating HLS model')
    hls_model = ModelGraph(config, reader, layer_list,  inputs=input_layers)
    return hls_model
