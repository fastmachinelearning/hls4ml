from __future__ import print_function
import numpy as np
import h5py
import os
import tarfile
import json
import argparse
import yaml
import sys
from shutil import copyfile
import math
from onnx import ModelProto, GraphProto, NodeProto, TensorProto
from onnx import optimizer, helper, numpy_helper

MAXMULT = 4096

filedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(filedir, "..", "hls-writer"))
from hls_writer import parse_config, print_array_to_cpp, hls_writer

def find_input(graph, inputs, transpose=True, perm=None):
    weights = None
    bias = None
    for i in inputs:
        tensor = next((x for x in graph.initializer if x.name == i), None)
        if tensor is not None:
            if len(tensor.dims) > 1: # Weights
                weights = numpy_helper.to_array(tensor)
            else: # Bias
                bias = numpy_helper.to_array(tensor)
    
    if transpose:
        weights = weights.transpose(perm)
    
    return weights, bias

def find_bn_input(graph, inputs, transpose=True):
    if len(inputs) != 5:
        raise Exception('ERROR: Unexpected number of inputs: Expected {}, got {}'.format(5, len(inputs)))
    
    scale = numpy_helper.to_array(next((x for x in graph.initializer if x.name == inputs[1]), None))
    beta = numpy_helper.to_array(next((x for x in graph.initializer if x.name == inputs[2]), None))
    mean = numpy_helper.to_array(next((x for x in graph.initializer if x.name == inputs[3]), None))
    var = numpy_helper.to_array(next((x for x in graph.initializer if x.name == inputs[4]), None))
    
    if transpose:
        scale = scale.transpose()
        beta = beta.transpose()
        mean = mean.transpose()
        var = var.transpose()

    return scale, beta, mean, var

def find_prelu_input(graph, inputs):
    alpha = None
    for i in inputs:
        tensor = next((x for x in graph.initializer if x.name == i), None)
        if tensor is not None:
            if len(tensor.dims) == 1: # Alpha array
                alpha = numpy_helper.to_array(tensor)
    
    return alpha

def get_onnx_attribute(operation, name, default=None):
    attr = next((x for x in operation.attribute if x.name == name), None)
    if attr is None:
        value = default
    else:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode()
    return value

############################################################################################
## M A I N
############################################################################################
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", action='store', dest='config',
                        help="Configuration file.")
    args = parser.parse_args()
    if not args.config: parser.error('A configuration file needs to be specified.')

    configDir  = os.path.abspath(os.path.dirname(args.config))
    yamlConfig = parse_config(args.config)
    if not os.path.isabs(yamlConfig['OutputDir']):
        yamlConfig['OutputDir'] = os.path.join(configDir, yamlConfig['OutputDir'])
    if not os.path.isabs(yamlConfig['OnnxModel']):
        yamlConfig['OnnxModel'] = os.path.join(configDir, yamlConfig['OnnxModel'])

    if not (yamlConfig["IOType"] == "io_parallel" or yamlConfig["IOType"] == "io_serial"): 
        raise Exception('ERROR: Invalid IO type')

    ######################
    ##  Do translation
    ######################
    if not os.path.isdir("{}/firmware/weights".format(yamlConfig['OutputDir'])):
        os.makedirs("{}/firmware/weights".format(yamlConfig['OutputDir']))

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    #Extract model architecture
    model = ModelProto()
    with open(yamlConfig['OnnxModel'], 'rb') as fid:
        model.ParseFromString(fid.read())
    
    #Define supported laers
    supported_operations = ['Gemm', 'Squeeze', 'Unsqueeze', 'BatchNormalization', 'Conv', 'Transpose', 'Flatten', 'Identity']
    activation_operations = ['Relu', 'Tanh', 'Sigmoid', 'LeakyRelu', 'ThresholdedRelu', 'HardSigmoid', 'Elu', 'Selu', 'PRelu', 'Softmax', 'Softsign', 'Softplus']

    operation_map = {'Gemm':'Dense', 'Relu':'Activation', 'Tanh':'Activation', 'Sigmoid':'Activation',
    'LeakyRelu':'LeakyReLU', 'ThresholdedRelu':'ThresholdedReLU', 'HardSigmoid':'Activation',
    'Elu':'ELU', 'Selu':'Activation', 'PRelu':'PReLU', 'Softmax':'Activation', 'Softsign':'Activation', 'Softplus':'Activation'}
    
    #Define layers to skip for conversion to HLS
    skip_layers = ['Squeeze', 'Unsqueeze', 'Dropout', 'Identity', 'Flatten', 'Transpose'] 

    #Loop through layers
    layer_counter = 0
    input_layer = {}

    passes = ['fuse_matmul_add_bias_into_gemm', 'eliminate_nop_transpose', 'fuse_consecutive_transposes', 'fuse_transpose_into_gemm']
    model = optimizer.optimize(model, passes)

    # Check for unsupported layer type
    for operation in model.graph.node:
        if operation.op_type not in supported_operations + activation_operations:
            raise Exception('ERROR: Unsupported operation type: {}'.format(operation.op_type))
    
    # Get input shape
    current_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    print('Input shape:', current_shape)

    # Set some variables to make the routine after a bit smoother
    is_conv2d = False
    is_dense = False
    for operation in model.graph.node:
        if operation.op_type == 'Conv':
            for i in operation.input:
                tensor = next((x for x in model.graph.initializer if x.name == i), None)
                if tensor is not None and tensor.dims == 4:
                    is_conv2d = True
                    break
        if operation.op_type == 'Gemm':
            is_dense = True
            break

    transpose_input = False
    transpose_perm = None

    print('Topology:')
    for op_id, operation in enumerate(model.graph.node):
        if operation.op_type == 'Flatten':
            current_shape = [current_shape[0], np.prod(current_shape[1:])]
        if operation.op_type == 'Transpose':
            transpose_input = True
            transpose_perm = get_onnx_attribute(operation, 'perm')
        if operation.op_type in skip_layers:
            continue 

        if operation.op_type in supported_operations + activation_operations:
            layer_counter = layer_counter + 1

        #Dictionary to fill in and append to layer_list
        layer = {}

        #Extract name for finding weights and biases
        if operation.name:
            layer['name'] = operation.name
        else:
            layer['name'] = operation.op_type + str(layer_counter)
        layer['class_name'] = operation_map.get(operation.op_type, operation.op_type)

        #Extract type of activation
        if operation.op_type in activation_operations:
            layer['activation'] = operation.op_type.lower()
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = operation.op_type.lower()
        
        # Skip activation layers if possible
        skip_layer = False
        # Default one layer call
        layer['n_part'] = 1
        #Get number of inputs and outputs
        #(We take it from the weights to avoid dealing with InputLayer and Flatten details)
        if layer['class_name'] == 'Dense':
            tran_weight = get_onnx_attribute(operation, 'transB', 0)
            weights, biases = find_input(model.graph, operation.input, tran_weight)
            
            layer['activation'] = 'linear' # Potentially overriden by the following layer 
            layer['n_in']=weights.shape[0]
            layer['n_out']=weights.shape[1]
            cur_n_zeros = print_array_to_cpp("w{}".format(layer_counter), weights, yamlConfig['OutputDir'])
            print_array_to_cpp("b{}".format(layer_counter), biases, yamlConfig['OutputDir'])
            layer['weights_n_zeros'] = cur_n_zeros

            # if this layer is too big (more than MAXMULT multiplications); 
            # break it out into chunks!
            layer['n_subout']=[weights.shape[1]]
            if layer['n_in']*layer['n_out']>MAXMULT and yamlConfig["IOType"] != "io_serial":
                n_subout = int(MAXMULT/layer['n_in'])
                n_totout = 0
                layer['n_subout'] = []
                layer['weights_n_subzeros'] = []
                layer['n_part'] = 0
                while n_totout < layer['n_out']:
                    if n_totout + n_subout <= layer['n_out']:
                        layer['n_subout'].append(n_subout)
                        n_totout += n_subout                    
                    else:
                        layer['n_subout'].append(layer['n_out']-n_totout)
                        n_totout += layer['n_out']-n_totout
                    layer['n_part'] += 1
                for i_part in range(0,layer['n_part']):
                    i_subout = 0
                    if i_part>0:
                        i_subout = sum(layer['n_subout'][0:i_part])
                    cur_n_zeros = print_array_to_cpp("w{}".format(layer_counter), weights, yamlConfig['OutputDir'], i_part, layer['n_part'], i_subout, layer['n_subout'][i_part])
                    print_array_to_cpp("b{}".format(layer_counter), biases, yamlConfig['OutputDir'], i_part, layer['n_part'], i_subout, layer['n_subout'][i_part])
                    layer['weights_n_subzeros'].append(cur_n_zeros)
            
            current_shape = [current_shape[0], layer['n_out']]
        elif layer['class_name']=='Conv':
            weights, biases = find_input(model.graph, operation.input, transpose=False)
            if len(weights.shape) == 3: # Conv1D
                layer['class_name'] = 'Conv1D'
                weights = weights.transpose([2, 1, 0])
                if transpose_input:
                    current_shape = [ current_shape[i] for i in transpose_perm ]
                    transpose_input = False
                # weights.shape = (filter_width, n_channels, n_filters)
                layer['y_in']=current_shape[2]
                layer['y_filt']=weights.shape[0]
                layer['n_chan']=weights.shape[1] 
                layer['n_filt']=weights.shape[2]
                layer['stride']=get_onnx_attribute(operation, 'strides')[0]
                auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
                if auto_pad != 'NOTSET':
                    if (layer['y_in'] % layer['stride'] == 0):
                        pad_along_width = max(layer['y_filt'] - layer['stride'], 0)
                    else:
                        pad_along_width = max(layer['y_filt'] - (layer['y_in'] % layer['stride']), 0)

                    pads = [pad_along_width // 2, pad_along_width - (pad_along_width // 2)]

                    if auto_pad == 'SAME_UPPER':
                        pads = sorted(pads)
                    elif auto_pad == 'SAME_LOWER':
                        pads = sorted(pads, reverse=True)
                    else: # 'VALID' padding
                        pads = [0, 0]
                else:
                    pads = get_onnx_attribute(operation, 'pads', [0, 0])

                layer['pad_left'] = pads[0]
                layer['pad_right'] = pads[1]
                if all(x == 0 for x in pads): # No padding, i.e., 'VALID' padding
                    layer['y_out'] = int(math.ceil(float(layer['y_in'] - layer['y_filt'] + 1) / float(layer['stride'])))
                else:
                    layer['y_out'] = int(math.ceil(float(layer['y_in']) / float(layer['stride'])))

                current_shape=[current_shape[0], layer['n_filt'], layer['y_out']]
            elif len(weights.shape) == 4: # Conv2D
                layer['class_name'] = 'Conv2D'
                weights = weights.transpose([2, 3, 1, 0])
                if transpose_input:
                    current_shape = [ current_shape[i] for i in transpose_perm ]
                    transpose_input = False
                layer['in_height']=current_shape[2]
                layer['in_width']=current_shape[3]
                layer['filt_height']=weights.shape[0]
                layer['filt_width']=weights.shape[1]
                layer['n_chan']=weights.shape[2]
                layer['n_filt']=weights.shape[3]
                strides = get_onnx_attribute(operation, 'strides')
                layer['stride_height'] = strides[0]
                layer['stride_width'] = strides[1]
                auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
                if auto_pad != 'NOTSET':
                    #Height
                    if (layer['in_height'] % layer['stride_height'] == 0):
                        pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
                    else:
                        pad_along_height = max(layer['filt_height'] - (layer['in_height'] % layer['stride_height']), 0)
                    pad_height = [pad_along_height // 2, pad_along_height - pad_along_height // 2]

                    #Width
                    if (layer['in_width'] % layer['stride_width'] == 0):
                        pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
                    else:
                        pad_along_width = max(layer['filt_width'] - (layer['in_width'] % layer['stride_width']), 0)
                    pad_width = [pad_along_width // 2, pad_along_width - pad_along_width // 2]

                    if auto_pad == 'SAME_UPPER':
                        pads = [min(pad_height), min(pad_width), max(pad_height), max(pad_width)]
                    elif auto_pad == 'SAME_LOWER':
                        pads = [max(pad_height), max(pad_width), min(pad_height), min(pad_width)]
                    else: # 'VALID' padding
                        pads = [0, 0, 0, 0]
                else:
                    pads = get_onnx_attribute(operation, 'pads', [0, 0, 0, 0])
                
                layer['pad_top'] = pads[0]
                layer['pad_bottom'] = pads[2]
                layer['pad_left'] = pads[1]
                layer['pad_right'] = pads[3]

                if all(x == 0 for x in pads): # No padding, i.e., 'VALID' padding in Keras/Tensorflow
                    layer['out_width'] = int(math.ceil(float(layer['in_width'] - layer['filt_width'] + 1) / float(layer['stride_width'])))
                    layer['out_height'] = int(math.ceil(float(layer['in_height'] - layer['filt_height'] + 1) / float(layer['stride_height'])))
                else:
                    layer['out_height'] = int(math.ceil(float(layer['in_height']) / float(layer['stride_height'])))
                    layer['out_width'] = int(math.ceil(float(layer['in_width']) / float(layer['stride_width'])))
                
                current_shape=[current_shape[0], layer['n_filt'], layer['out_height'], layer['out_width']]
            cur_n_zeros = print_array_to_cpp("w{}".format(layer_counter), weights, yamlConfig['OutputDir'])
            print_array_to_cpp("b{}".format(layer_counter), biases, yamlConfig['OutputDir'])
            layer['weights_n_zeros'] = cur_n_zeros
        elif layer['class_name']=='BatchNormalization':
            layer['epsilon'] = get_onnx_attribute(operation, 'epsilon')
            layer['momentum'] = get_onnx_attribute(operation, 'momentum')
            scale, beta, mean, var = find_bn_input(model.graph, operation.input)
            
            scale = scale/np.sqrt(var + layer['epsilon'])
            
            print_array_to_cpp("scale{}".format(layer_counter), scale, yamlConfig['OutputDir'])
            print_array_to_cpp("beta{}".format(layer_counter), beta, yamlConfig['OutputDir'])
            print_array_to_cpp("mean{}".format(layer_counter), mean, yamlConfig['OutputDir'])
            
            if is_dense:
                layer['n_in']=mean.shape[0]
                layer['n_out']=mean.shape[0]
                layer['n_filt'] = -1
                current_shape = [current_shape[0], layer['n_out']]
            elif is_conv2d:
                layer['n_in']=current_shape[1]*current_shape[2]*current_shape[3] 
                layer['n_out']=layer['n_in']
                layer['in_height']=current_shape[1]
                layer['in_width']=current_shape[2]
                layer['n_filt']=current_shape[3]
                current_shape=[current_shape[0], layer['n_filt'], layer['in_height'], layer['in_width']]
        elif layer['class_name']=='Activation':
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = layer['activation']
                skip_layer = True
                layer_counter = layer_counter - 1
        elif layer['class_name'] in ['ELU', 'LeakyReLU', 'ThresholdedReLU']:
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = layer['class_name']
                layer_list[-1]['activ_param'] = get_onnx_attribute(operation, 'alpha', 0.01)
                skip_layer = True
                layer_counter = layer_counter - 1
            else:
                layer['activation'] = layer['class_name']
                layer['activ_param'] = get_onnx_attribute(operation, 'alpha', 0.01)
        elif layer['class_name']=='PReLU':
            if layer_list[-1]['class_name'] != 'BatchNormalization':
                layer_list[-1]['activation'] = layer['class_name']
                skip_layer = True
                layer_counter = layer_counter - 1
            else:
                layer['activation'] = layer['class_name']
            
            #Translate learned alpha array from h5 file
            alpha = find_prelu_input(model.graph, operation.input)
            print_array_to_cpp("a{}".format(layer_counter), alpha, yamlConfig['OutputDir'])

        if not skip_layer:
            print('Layer name: {}, layer type: {}, current shape: {}, number of zeros: {}'.format(layer['name'], layer['class_name'], current_shape, cur_n_zeros))
            if layer['n_part'] > 1: 
                print(' -> layer will be divided into {} sublayer calls; output neurons: {} '.format(layer['n_part'], layer['n_subout']))
            layer_list.append( layer )


    #################
    ## Generate HLS
    #################

    #Weights and biases are already dumped to output directory
    #Now generate HLS from list of layer dictionaries
    hls_writer(layer_list, yamlConfig)


if __name__ == "__main__":
    main()
