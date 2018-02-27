import numpy as np
import h5py
import os
import tarfile
import json
import argparse
import yaml
import sys
import torch
import pickle
import re
from shutil import copyfile

filedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(filedir, "..", "hls-writer"))
from hls_writer import hls_writer
## hyphen doesn't work for python import...Below won't work
# sys.path.insert(0,os.path.join(filedir, "..", "keras-to-hls"))
# from keras-to-hls import parse_config, print_array_to_cpp

#######################################
## Config module
#######################################
def parse_config(config_file) :

    print "Loading configuration from " + str(config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

#######################################
## Print a bias or weight array to C++
#######################################
def print_array_to_cpp(name, a, odir ):

    #count zeros
    zero_ctr = 0;
    for x in np.nditer(a, order='C'):
        if x == 0:
            zero_ctr += 1

    #put output in subdir for tarballing later
    f=open("{}/firmware/weights/{}.h".format(odir,name),"w")

    #meta data
    f.write("//Numpy array shape {}\n".format(a.shape))
    f.write("//Min {}\n".format(np.min(a)))
    f.write("//Max {}\n".format(np.max(a)))
    f.write("//Number of zeros {}\n".format(zero_ctr))
    f.write("\n")

    #c++ variable
    if "w" in name:
        f.write("weight_default_t {}".format(name))
    elif "b" in name:
        f.write("bias_default_t {}".format(name))
    else:
        raise Exception('ERROR: Unkown weights type')

    for x in a.shape:
        f.write("[{}]".format(x))
    f.write(" = {")

    #fill c++ array.
    #not including internal brackets for multidimensional case
    i=0;
    for x in np.nditer(a, order='C'):
        if i==0:
            f.write("{}".format(x))
        else:
            f.write(", {}".format(x))
        i=i+1
    f.write("};\n")
    f.close()

    return zero_ctr;


############################################################################################
## M A I N
############################################################################################
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", action='store', dest='config', default="pytorch-config.yml",
                        help="Configuration file.")
    args = parser.parse_args()
    if not args.config: parser.error('A configuration file needs to be specified.')

    configDir  = os.path.abspath(os.path.dirname(args.config))
    yamlConfig = parse_config(args.config)
    if not os.path.isabs(yamlConfig['OutputDir']):
        yamlConfig['OutputDir'] = os.path.join(configDir, yamlConfig['OutputDir'])
    if not os.path.isabs(yamlConfig['PytorchModel']):
        yamlConfig['PytorchModel'] = os.path.join(configDir, yamlConfig['PytorchModel'])
    if not os.path.isabs(yamlConfig['PytorchDict']):
        yamlConfig['PytorchDict'] = os.path.join(configDir, yamlConfig['PytorchDict'])

    if not (yamlConfig["IOType"] == "io_parallel" or yamlConfig["IOType"] == "io_serial"):
        raise Exception('ERROR: Invalid IO type')

    ######################
    ##  Do translation
    ######################
    if not os.path.isdir("{}/firmware/weights".format(yamlConfig['OutputDir'])):
        os.makedirs("{}/firmware/weights".format(yamlConfig['OutputDir']))

    t = pickle.load(open( yamlConfig['PytorchModel'] , 'rb' ))
    n = torch.load(yamlConfig['PytorchDict'])
    t.load_state_dict(n)
    modelstr = repr(t).split('\n')

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    matchlayer = re.compile("^\s*(\d):\s\W*")
    #Loop through layers
    layer_counter = 1;
    for i, pytorch_layer in enumerate(modelstr):
        Nlayer = -1
        NlayerMatch =re.search("\((\d)\):\s", pytorch_layer)
        if NlayerMatch is not None:
            print pytorch_layer, NlayerMatch.group(1)
            Nlayer = NlayerMatch.group(1)

        layerFun = pytorch_layer.split(":")[-1]

        matchname = re.match("(\w+)\(in_features=(\d+), out_features=(\d+).*\)", layerFun.strip())
        if matchname is None:
            continue

        # #Dictionary to fill in and append to layer_list
        layer={}

        # #Extract name for finding weights and biases
        layer['name'] = matchname.group(1)+"_"+Nlayer

        # #Get number of inputs and outputs
        layer["n_in"] =  int(matchname.group(2))
        layer["n_out"] =  int(matchname.group(3))

        # #Extract type of activation and number of nodes
        layer["activation"] = modelstr[i+1].split(":")[-1].strip().lower()[:-2]

        # Translate weights and biases from tensorfile
        weights = n[Nlayer+".weight"].numpy().transpose()
        biases =  n[Nlayer+".bias"].numpy().transpose()
        cur_n_zeros = print_array_to_cpp("w{}".format(layer_counter), weights, yamlConfig['OutputDir'])
        print_array_to_cpp("b{}".format(layer_counter), biases, yamlConfig['OutputDir'])
        layer['weights_n_zeros'] = cur_n_zeros

        layer_list.append(layer)

        NlayerMatch =re.search("\((\d)\):\s", pytorch_layer)

        layer_counter = layer_counter+1


    #################
    ## Generate HLS
    #################

    #Weights and biases are already dumped to output directory
    #Now generate HLS from list of layer dictionaries
    hls_writer(layer_list, yamlConfig)

if __name__ == "__main__":
    main();
