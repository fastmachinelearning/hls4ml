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

MAXMULT = 4096

filedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(filedir, "..", "hls-writer"))
from hls_writer import parse_config, write_hls
from hls_model import HLSModel

class KerasDataReader:
    def __init__(self, config):
        self.config = config

    def get_weights_data(self, layer_name, var_name):
        def h5_visitor_func(name):
            if var_name in name:
                return name

        with h5py.File(self.config['KerasH5'], 'r') as h5file:
            found_data = h5file[layer_name].visit(h5_visitor_func)
            if found_data:
                data = h5file['/{}/{}'.format(layer_name,found_data)][()]
            else:
                data = None

        return data

def get_weights_shape(h5filename, layer_name, var_name='kernel'):
    def h5_visitor_func(name):
        if var_name in name:
            return name

    with h5py.File(h5filename, 'r') as h5file:
        found_data = h5file[layer_name].visit(h5_visitor_func)
        if found_data:
            shape = h5file['/{}/{}'.format(layer_name,found_data)].shape

    return shape

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
    if not os.path.isabs(yamlConfig['KerasH5']):
        yamlConfig['KerasH5'] = os.path.join(configDir, yamlConfig['KerasH5'])
    if not os.path.isabs(yamlConfig['KerasJson']):
        yamlConfig['KerasJson'] = os.path.join(configDir, yamlConfig['KerasJson'])

    if not (yamlConfig["IOType"] == "io_parallel" or yamlConfig["IOType"] == "io_serial"):
        raise Exception('ERROR: Invalid IO type')

    ######################
    ##  Do translation
    ######################
    if not os.path.isdir("{}/firmware/weights".format(yamlConfig['OutputDir'])):
        os.makedirs("{}/firmware/weights".format(yamlConfig['OutputDir']))

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    #Extract model architecture from json
    with open( yamlConfig['KerasJson'] ) as json_file:
        model_arch = json.load(json_file)
    #print(model_arch)

    #Define supported laers
    core_layers = ['InputLayer', 'Dropout', 'Flatten', 'Dense', 'BinaryDense', 'TernaryDense']
    conv_layers = ['Conv1D', 'Conv2D']
    pooling_layers = ['MaxPooling1D', 'MaxPooling2D', 'AveragePooling1D', 'AveragePooling2D']
    norm_layers = ['BatchNormalization']
    activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU']
    merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate']
    supported_layers = core_layers + conv_layers + pooling_layers + norm_layers + activation_layers + merge_layers

    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten']
    #Map inputs of skipped and split (activation) layers
    inputs_map = {}

    #Loop through layers
    layer_counter = 0

    input_layers = None
    output_layers = None

    layer_config = None
    if model_arch['class_name'] == 'Sequential':
        print('Interpreting Sequential')
        layer_config = model_arch["config"]
        # Sequential doesn't have InputLayer
        input_layer = {}
        input_layer['name'] = 'input1'
        input_layer['class_name'] = 'InputLayer'
        input_layer['input_shape'] = layer_config[0]['config']['batch_input_shape'][1:]
        layer_list.append(input_layer)
        print('Input shape:', input_layer['input_shape'])
    elif model_arch['class_name'] == 'Model':
        print('Interpreting Model')
        layer_config = model_arch["config"]["layers"]
        input_layers = [ inp[0] for inp in model_arch["config"]["input_layers"] ]
        output_layers = [ out[0] for out in model_arch["config"]["output_layers"] ]

    # Get input shape and check for unsupported layer type
    current_shape = None
    for keras_layer in layer_config:
        if keras_layer["class_name"] not in supported_layers:
            raise Exception('ERROR: Unsupported layer type: {}'.format(keras_layer["class_name"]))
        if 'batch_input_shape' in keras_layer['config']:
            current_shape = keras_layer['config']['batch_input_shape'] # [None, 100, 7]

    print('Topology:')
    for keras_layer in layer_config:
        if keras_layer["class_name"] is 'Flatten':
            current_shape = [current_shape[0], np.prod(current_shape[1:])]
        if keras_layer["class_name"] in skip_layers:
            if 'inbound_nodes' in keras_layer:
                name = keras_layer['config']['name']
                #Currently supported skipped layers have only one input
                parent_input = keras_layer['inbound_nodes'][0][0][0]
                #Skipped layers can follow each other (e.g., Dropout -> Flatten)
                inputs_map[name] = inputs_map.get(parent_input, parent_input)
            continue

        if keras_layer["class_name"] in supported_layers:
            layer_counter = layer_counter + 1

        #Dictionary to fill in and append to layer_list
        layer = {}

        #Extract name for finding weights and biases
        layer['name']=keras_layer['config']['name']
        layer['class_name']=keras_layer['class_name']

        #Extract inbound nodes
        if 'inbound_nodes' in keras_layer and len(keras_layer['inbound_nodes']) > 0:
            layer['inputs'] = [ inputs_map.get(inp[0], inp[0]) for inp in keras_layer['inbound_nodes'][0] ]

        #Extract type of activation and number of nodes
        for config,config_value in keras_layer["config"].items():
            if(config=="activation"):
                layer['activation']=config_value
            if(config=="epsilon"):
                layer['epsilon']=config_value
            #if(config=="units"):
                #print("PARSED NUM OF NODES",config_value)

        # Default one layer call
        if layer['class_name'] == 'InputLayer':
            layer['input_shape'] = keras_layer['config']['batch_input_shape'][1:]
        if 'Dense' in layer['class_name']:
            weights_shape = get_weights_shape(yamlConfig['KerasH5'], layer['name'])
            layer['n_in'] = weights_shape[0]
            layer['n_out'] = weights_shape[1]
            if 'Binary' in layer['class_name']:
                layer['quantize'] = 2
            elif 'Ternary' in layer['class_name']:
                layer['quantize'] = 3
            else:
                layer['quantize'] = 0
            current_shape = [current_shape[0], layer['n_out']]
        elif layer['class_name']=='Conv1D':
            # weights_shape = (filter_width, n_channels, n_filters)
            weights_shape = get_weights_shape(yamlConfig['KerasH5'], layer['name'])
            layer['y_in']=current_shape[1]
            layer['y_filt']=weights_shape[0] # or keras_layer['config']['kernel_size']
            layer['n_chan']=weights_shape[1]
            layer['n_filt']=weights_shape[2] # or keras_layer['config']['filters']
            layer['stride']=keras_layer['config']['strides'][0]
            layer['padding']=keras_layer['config']['padding']
            if layer['padding']=='same':
                in_width = current_shape[1]
                layer['y_out'] = int(math.ceil(float(in_width) / float(layer['stride'])))
                if (in_width % layer['stride'] == 0):
                    pad_along_width = max(layer['y_filt'] - layer['stride'], 0)
                else:
                    pad_along_width = max(layer['y_filt'] - (in_width % layer['stride']), 0)
                layer['pad_left']  = pad_along_width // 2
                layer['pad_right']  = pad_along_width - layer['pad_left']
            elif layer['padding']=='valid':
                in_width = current_shape[1]
                layer['y_out'] = int(math.ceil(float(in_width - layer['y_filt'] + 1) / float(layer['stride'])))
                layer['pad_left'] = 0
                layer['pad_right'] = 0
            current_shape=[current_shape[0], layer['y_out'], layer['n_filt']]
        elif layer['class_name']=='Conv2D':
            # weights_shape = (filter_height, filter_width, n_channels, n_filters)
            weights_shape = get_weights_shape(yamlConfig['KerasH5'], layer['name'])
            layer['in_height']=current_shape[1]
            layer['in_width']=current_shape[2]
            layer['filt_height']=weights_shape[0]
            layer['filt_width']=weights_shape[1]
            layer['n_chan']=weights_shape[2]
            layer['n_filt']=weights_shape[3]
            layer['stride_height']=keras_layer['config']['strides'][0]
            layer['stride_width']=keras_layer['config']['strides'][1]
            layer['padding']=keras_layer['config']['padding']
            if layer['padding']=='same':
                #Height
                in_height = current_shape[1]
                layer['out_height'] = int(math.ceil(float(in_height) / float(layer['stride_height'])))
                if (in_height % layer['stride_height'] == 0):
                    pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
                else:
                    pad_along_height = max(layer['filt_height'] - (in_height % layer['stride_height']), 0)
                layer['pad_top']  = pad_along_height // 2
                layer['pad_bottom']  = pad_along_height - layer['pad_top']
                #Width
                in_width = current_shape[2]
                layer['out_width'] = int(math.ceil(float(in_width) / float(layer['stride_width'])))
                if (in_width % layer['stride_width'] == 0):
                    pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
                else:
                    pad_along_width = max(layer['filt_width'] - (in_width % layer['stride_width']), 0)
                layer['pad_left']  = pad_along_width // 2
                layer['pad_right']  = pad_along_width - layer['pad_left']
            elif layer['padding']=='valid':
                in_height = current_shape[1]
                in_width = current_shape[2]
                layer['out_width'] = int(math.ceil(float(in_width - layer['filt_width'] + 1) / float(layer['stride_width'])))
                layer['out_height'] = int(math.ceil(float(in_height - layer['filt_height'] + 1) / float(layer['stride_height'])))
                layer['pad_top'] = 0
                layer['pad_bottom'] = 0
                layer['pad_left'] = 0
                layer['pad_right'] = 0
            current_shape=[current_shape[0], layer['out_height'], layer['out_width'], layer['n_filt']]
        elif layer['class_name']=='BatchNormalization':
            in_size = 1
            for dim in current_shape[1:]:
                in_size *= dim
            layer['n_in'] = in_size
            layer['n_out'] = layer['n_in']
            if len(current_shape) == 2:
                layer['n_filt'] = -1
            elif len(current_shape) == 3:
                layer['n_filt']=current_shape[2]
            elif len(current_shape) == 4:
                layer['n_filt']=current_shape[3]
        elif 'Pooling' in layer['class_name']:
            info = layer['class_name'].split('Pooling')
            d = int(info[1].split('D')[0])
            op = info[0]
            if d == 1:
                layer['pool_size']=keras_layer['config']['pool_size']
                layer['stride']=keras_layer['config']['stride']
            elif d == 2:
                layer['in_height']=current_shape[1]
                layer['in_width']=current_shape[2]
                layer['n_filt']=current_shape[3]
                layer['stride_height']=keras_layer['config']['strides'][0]
                layer['stride_width']=keras_layer['config']['strides'][1]
                layer['pool_height']=keras_layer['config']['pool_size'][0]
                layer['pool_width']=keras_layer['config']['pool_size'][1]
                layer['padding']=keras_layer['config']['padding']
            if layer['padding']=='same':
                #Height
                in_height = current_shape[1]
                layer['out_height'] = int(math.ceil(float(in_height) / float(layer['stride_height'])))
                if (in_height % layer['stride_height'] == 0):
                    pad_along_height = max(layer['pool_height'] - layer['stride_height'], 0)
                else:
                    pad_along_height = max(layer['pool_height'] - (in_height % layer['stride_height']), 0)
                layer['pad_top']  = pad_along_height // 2
                layer['pad_bottom']  = pad_along_height - layer['pad_top']
                #Width
                in_width = current_shape[2]
                layer['out_width'] = int(math.ceil(float(in_width) / float(layer['stride_width'])))
                if (in_width % layer['stride_width'] == 0):
                    pad_along_width = max(layer['pool_width'] - layer['stride_width'], 0)
                else:
                    pad_along_width = max(layer['pool_width'] - (in_width % layer['stride_width']), 0)
                layer['pad_left']  = pad_along_width // 2
                layer['pad_right']  = pad_along_width - layer['pad_left']
                layer['n_out'] = layer['out_height'] * layer['out_width'] * layer['n_filt']
            elif layer['padding']=='valid':
                in_height = current_shape[1]
                in_width = current_shape[2]
                layer['out_width'] = int(math.ceil(float(in_width - layer['pool_width'] + 1) / float(layer['stride_width'])))
                layer['out_height'] = int(math.ceil(float(in_height - layer['pool_height'] + 1) / float(layer['stride_height'])))
                layer['pad_top'] = 0
                layer['pad_bottom'] = 0
                layer['pad_left'] = 0
                layer['pad_right'] = 0
                layer['n_out'] = layer['out_height'] * layer['out_height'] * layer['n_filt']
            current_shape=[current_shape[0], layer['out_height'], layer['out_width'], layer['n_filt']]

        elif layer['class_name']=='LeakyReLU':
            layer['activation'] = layer['class_name']
            layer['activ_param'] = keras_layer["config"].get('alpha', 0.3)
        elif layer['class_name']=='ThresholdedReLU':
            layer['activation'] = layer['class_name']
            layer['activ_param'] = keras_layer["config"].get('theta', 1.)
        elif layer['class_name']=='ELU':
            layer['activation'] = layer['class_name']
            layer['activ_param'] = keras_layer["config"].get('alpha', 1.)
        elif layer['class_name']=='PReLU':
            layer['activation'] = layer['class_name']

        elif layer['class_name'] in merge_layers:
            layer['op'] = layer['class_name'].lower()
            if layer['class_name'] == 'Concatenate':
                rank = len(current_shape[1:])
                if rank > 3:
                    raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')
                layer['op'] = layer['class_name'].lower() + '{}d'.format(rank)
                layer['axis'] = keras_layer['config']['axis']
            else:
                layer['class_name'] = 'Merge'
            if len(layer['inputs']) > 2:
                raise Exception('ERROR: Merging more than two tensors is not yet supported.')

        print('Layer name: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], current_shape))
        layer_list.append( layer )
        if 'activation' in layer and layer['class_name'] not in activation_layers:
            act_layer = {}
            act_layer['name'] = layer['name'] + '_' + layer['activation']
            act_layer['activation'] = layer['activation']
            if 'activ_param' in layer:
                act_layer['activ_param'] = layer['activ_param']
                act_layer['class_name'] = layer['activation']
            else:
                act_layer['class_name'] = 'Activation'
            inputs_map[layer['name']] = act_layer['name']
            if output_layers is not None and layer['name'] in output_layers:
                output_layers = [act_layer['name'] if name == layer['name'] else name for name in output_layers]
            layer_list.append(act_layer)


    #################
    ## Generate HLS
    #################

    reader = KerasDataReader(yamlConfig)
    hls_model = HLSModel(yamlConfig, reader, layer_list, input_layers, output_layers)
    write_hls(hls_model)


if __name__ == "__main__":
    main()
