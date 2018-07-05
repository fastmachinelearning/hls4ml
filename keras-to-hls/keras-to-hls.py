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
from hls_writer import parse_config, print_array_to_cpp, hls_writer

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

    h5File = h5py.File( yamlConfig['KerasH5'] )

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    #Extract model architecture from json
    with open( yamlConfig['KerasJson'] ) as json_file:
        model_arch = json.load(json_file)
    #print(model_arch)

    #Define supported laers
    supported_layers = ['InputLayer','Dropout', 'Flatten', 'Dense', 'Conv1D', 'Conv2D']

    #Define layers to skip for conversion to HLS
    skip_layers = ['InputLayer','Dropout', 'Flatten'] 

    #Loop through layers
    layer_counter = 0
    input_layer = {}

    layer_config = None
    if model_arch['class_name'] == 'Sequential':
        print 'Interpreting Sequential'
        layer_config = model_arch["config"]
    elif model_arch['class_name'] == 'Model':
        print 'Interpreting Model'
        layer_config = model_arch["config"]["layers"]

    # Get input shape and check for unsupported layer type
    current_shape = None
    for keras_layer in layer_config:
        if keras_layer["class_name"] not in supported_layers:
            raise Exception('ERROR: Unsupported layer type: %s'%keras_layer["class_name"])            
        if 'batch_input_shape' in keras_layer['config']:
            current_shape = keras_layer['config']['batch_input_shape'] # [None, 100, 7]    
    print 'Input shape:', current_shape

    print 'Topology:' 
    for keras_layer in layer_config:
        if keras_layer["class_name"] is 'Flatten':
            current_shape = [current_shape[0], np.prod(current_shape[1:])]
        if keras_layer["class_name"] in skip_layers:
            continue 

        layer_counter = layer_counter+1

        #Dictionary to fill in and append to layer_list
        layer = {}

        #Extract name for finding weights and biases
        layer['name']=keras_layer['config']['name']
        layer['class_name']=keras_layer['class_name']

        #Extract type of activation and number of nodes
        for config,config_value in keras_layer["config"].items():
            if(config=="activation"):
                layer['activation']=config_value
            #if(config=="units"):
                #print("PARSED NUM OF NODES",config_value)

        #Translate weights and biases from h5 file
        weights = h5File['/{}/{}/kernel:0'.format(layer['name'],layer['name'])][()]
        biases = h5File['/{}/{}/bias:0'.format(layer['name'],layer['name'])][()]
        cur_n_zeros = print_array_to_cpp("w{}".format(layer_counter), weights, yamlConfig['OutputDir'])
        print_array_to_cpp("b{}".format(layer_counter), biases, yamlConfig['OutputDir'])
        layer['weights_n_zeros'] = cur_n_zeros 

        #Get number of inputs and outputs
        #(We take it from the weights to avoid dealing with InputLayer and Flatten details)
        if layer['class_name']=='Dense':
            layer['n_in']=weights.shape[0]
            layer['n_out']=weights.shape[1]
            # if this layer is too big (more than MAXMULT multiplications); 
            # break it out into chunks!
            layer['n_subout']=[weights.shape[1]]
            layer['n_part'] = 1
            if layer['n_in']*layer['n_out']>MAXMULT:
                n_subout = int(MAXMULT/layer['n_in'])
                n_totout = 0
                layer['n_subout'] = []
                layer['n_part'] = 0
                while n_totout < layer['n_out']:
                    if n_totout + n_subout <= layer['n_out']:
                        layer['n_subout'].append(n_subout)
                        n_totout += n_subout                    
                    else:
                        layer['n_subout'].append(layer['n_out']-n_totout)
                        n_totout += layer['n_out']-n_totout

                    layer['n_part'] += 1
                
            current_shape = [current_shape[0], layer['n_out']]
        elif layer['class_name']=='Conv1D':
            # weights.shape = (filter_width, n_channels, n_filters)
            layer['y_in']=current_shape[1]
            layer['y_filt']=weights.shape[0] # or keras_layer['config']['kernel_size']
            layer['n_chan']=weights.shape[1] 
            layer['n_filt']=weights.shape[2] # or keras_layer['config']['filters']
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
            layer['in_height']=current_shape[1]
            layer['in_width']=current_shape[2]
            layer['filt_height']=weights.shape[0]
            layer['filt_width']=weights.shape[1]
            layer['n_chan']=weights.shape[2]
            layer['n_filt']=weights.shape[3]
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
        print 'Layer name: %s, layer type: %s, current shape: %s, number of zeros: %s'%(layer['name'], layer['class_name'], current_shape, cur_n_zeros)
        if layer['n_part'] > 1: 
            print ' -> layer will be divided into %s sublayer calls; output neurons: %s '%(layer['n_part'], layer['n_subout'])
        layer_list.append( layer )
        

    #################
    ## Generate HLS
    #################

    #Weights and biases are already dumped to output directory
    #Now generate HLS from list of layer dictionaries
    hls_writer(layer_list, yamlConfig)


if __name__ == "__main__":
    main()
