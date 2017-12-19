import numpy as np
import h5py
import os
import tarfile
import json
import argparse
import yaml
import random
from shutil import copyfile

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

    #put output in subdir for tarballing later
    f=open("{}/firmware/weights/{}.h".format(odir,name),"w")

    #meta data
    f.write("//Numpy array shape {}\n".format(a.shape))
    f.write("//Min {}\n".format(np.min(a)))
    f.write("//Max {}\n".format(np.max(a)))
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
    zero_ctr = 0;
    for x in np.nditer(a, order='C'):
        if x == 0: zero_ctr += 1
        if i==0:
             f.write("{}".format(x))
         else:
             f.write(", {}".format(x))
        i=i+1
    f.write("};")
    f.close()

    return zero_ctr;

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

    yamlConfig = parse_config(args.config)
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

    #Define layers to skip for conversion to HLS
    skip_layers = ['InputLayer', 'Dropout', 'Flatten'] 

    #Loop through layers
    layer_counter = 0;
    weights_n_zeros = [];
    for keras_layer in model_arch["config"]["layers"]:
        if keras_layer["class_name"] in skip_layers:
            continue 

        layer_counter = layer_counter+1

        #Dictionary to fill in and append to layer_list
        layer = {}

        #Extract name for finding weights and biases
        layer['name']=keras_layer['name']

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
        weights_n_zeros.append( cur_n_zeros );

        #Get number of inputs and outputs
        #(We take it from the weights to avoid dealing with InputLayer and Flatten details)
        shape_count = 0#more elegant way of doing this?
        for x in weights.shape:
            if(shape_count==0):
                layer['n_in']=x
            elif(shape_count==1):
                layer['n_out']=x
            else :
                raise Exception('ERROR: WRONG DIMENSIONS')
            shape_count = shape_count+1

        print layer
        layer_list.append( layer )
        


    ###################
    ## myproject.cpp
    ###################

    #this part of code uses list of layer dictionaries from above, 
    #so it is independent of keras and can be used for tensorflow, etc

    f = open('../hls-template/firmware/myproject.cpp','r')
    fout = open('{}/firmware/myproject.cpp'.format(yamlConfig['OutputDir']),'w')

    for line in f.readlines():
        #Add headers to weights and biases
        if '//hls-fpga-machine-learning insert weights' in line:
            newline = line
            for i in range(1,len(layer_list)+1):
                newline = newline + '#include "weights/w{}.h"\n'.format(i)
                newline = newline + '#include "weights/b{}.h"\n'.format(i)

        #Add input/output type
        elif '//hls-fpga-machine-learning insert IO' in line:
            newline = line;
            if yamlConfig["IOType"] == "io_parallel":
                newline = newline + '    #pragma HLS ARRAY_PARTITION variable=data complete \n';
                newline = newline + '    #pragma HLS ARRAY_PARTITION variable=res complete \n';
            if yamlConfig["IOType"] == "io_serial":
                newline = newline + '    #pragma HLS STREAM variable=data dim=1\n';
                newline = newline + '    #pragma HLS STREAM variable=res dim=1\n';

        #Add layers
        elif '//hls-fpga-machine-learning insert layers' in line:
            newline = line + '\n'
            for i in range(1,len(layer_list)+1):
                
                #Input to compute_layer
                if(i==1):
                    input_type = 'input_t'
                    input_object = 'data'
                    n_in = 'N_INPUTS'
                else:
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'N_LAYER_{}'.format(i-1);

                #Outputs of compute_layer and activation 
                if(i==len(layer_list)):
                    output_type = 'result_t'
                    output_object = 'res'
                    n_out = 'N_OUTPUTS'
                else:
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    n_out = 'N_LAYER_{}'.format(i)

                if(i!=len(layer_list)):
                    newline = newline + '    {} layer{}_out[{}];\n'.format(output_type,i,n_out)
                    if yamlConfig["IOType"] == "io_parallel": newline = newline + '    #pragma HLS ARRAY_PARTITION variable=layer{}_out complete\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline = newline + '    #pragma HLS STREAM variable=layer{}_out dim=1\n'.format(i)

                #Compute layer
                if layer_list[i-1]['activation'] == "linear":
                    newline = newline + '    nnet::compute_layer<{}, {}, config{}>({}, {}, w{}, b{});\n'.format(input_type, output_type, i, input_object, output_object, i, i)
                else:
                    newline = newline + '    {} logits{}[{}];\n'.format(output_type,i,n_out)
                    if yamlConfig["IOType"] == "io_parallel": newline = newline + '    #pragma HLS ARRAY_PARTITION variable=logits{} complete\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline = newline + '    #pragma HLS STREAM variable=logits{} dim=1\n'.format(i)
                    newline = newline + '    nnet::compute_layer<{}, {}, config{}>({}, logits{}, w{}, b{});\n'.format(input_type, output_type, i, input_object, i, i, i, i)
                
                #Activations
                activation_name = layer_list[i-1]['activation']+'_config'+str(i)
                if layer_list[i-1]['activation'] == "relu":
                    newline = newline + '    nnet::relu<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, activation_name, i, output_object)
                elif layer_list[i-1]['activation'] =="softmax":
                    newline = newline + '    nnet::softmax<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, activation_name, i, output_object)
                elif layer_list[i-1]['activation'] =="sigmoid":
                    newline = newline + '    nnet::sigmoid<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, activation_name, i, output_object)
                elif layer_list[i-1]['activation'] =="tanh":
                    newline = newline + '    nnet::tanh<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, activation_name, i, output_object)
                elif layer_list[i-1]['activation'] =="linear":
                    newline = newline + '    //linear activation\n'
                else:
                    raise Exception('ERROR: MISSING ACTIVATION')

                newline = newline + '\n'

        #Just copy line
        else: 
            newline = line
        fout.write(newline)
    f.close()
    fout.close()

    ###################
    ## parameters.h
    ###################

    f = open('../hls-template/firmware/parameters.h','r')
    fout = open('{}/firmware/parameters.h'.format(yamlConfig['OutputDir']),'w')

    config_template = """struct config{index} : nnet::layer_config {{
        static const unsigned n_in = {n_in};
        static const unsigned n_out = {n_out};
        static const unsigned io_type = nnet::{iotype};
        static const unsigned reuse_factor = {reuse};
        static const bool store_weights_in_bram = false;
        static const unsigned n_zeros = {nzeros};
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        }};\n"""

    activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
        static const unsigned n_in = {n_in};
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::{iotype};
        }};\n"""


    for line in f.readlines():

        #Insert numbers
        if '//hls-fpga-machine-learning insert numbers' in line:
            newline = line
            newline = newline + 'typedef {precision} accum_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline = newline + 'typedef {precision} weight_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline = newline + 'typedef {precision} bias_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline = newline + 'typedef {precision} input_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline = newline + 'typedef {precision} result_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            for i in range(1,len(layer_list)+1):

                if i==1 :
                    newline = newline + '#define N_INPUTS {}\n'.format(layer_list[i-1]['n_in'])
                    newline = newline + '#define N_LAYER_1 {}\n'.format(layer_list[i-1]['n_out'])
                elif i==len(layer_list):
                    newline = newline + '#define N_OUTPUTS {}\n'.format(layer_list[i-1]['n_out'])
                else:
                    newline = newline + '#define N_LAYER_{} {}\n'.format(i, layer_list[i-1]['n_out'])
                    
        elif '//hls-fpga-machine-learning insert layer-precision' in line:
            newline = line
            for i in range(1,len(layer_list)):
                newline = newline + 'typedef {precision} layer{index}_t;\n'.format(precision=yamlConfig["DefaultPrecision"], index=i)

        elif "//hls-fpga-machine-learning insert layer-config" in line:
            newline = line
            for i in range(1,len(layer_list)+1):
                if i==1 :
                    layer_in_name = "N_INPUTS"
                    layer_out_name = "N_LAYER_1"
                elif i==len(layer_list):
                    layer_in_name = "N_LAYER_%i" % (i-1)
                    layer_out_name = "N_OUTPUTS"
                else:
                    layer_in_name = "N_LAYER_%i" % (i-1)
                    layer_out_name = "N_LAYER_%i" % (i)
                
                newline = newline + config_template.format(index=str(i), 
                                                           n_in=layer_in_name, 
                                                           n_out=layer_out_name,
                                                           iotype=yamlConfig["IOType"],
                                                           reuse=yamlConfig["ReuseFactor"],
                                                           nzeros=weights_n_zeros[i-1])

                newline = newline + activ_config_template.format(type=layer_list[i-1]['activation'],
                                                                 index=str(i), 
                                                                 n_in=layer_out_name,
                                                                 iotype=yamlConfig["IOType"]) 

        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()


    ###################
    ## test bench
    ###################

    f = open('../hls-template/myproject_test.cpp','r')
    fout = open('{}/myproject_test.cpp'.format(yamlConfig['OutputDir']),'w')

    for line in f.readlines():

        #Insert numbers
        if '//hls-fpga-machine-learning insert data' in line:
            newline = line
            newline = newline + '  input_t  data_str[N_INPUTS] = {'
            for i in range(0,layer_list[0]['n_in']-1):
                newline = newline + '0,'
            newline = newline + '0};\n'
        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()


    #######################
    ## plain copy of rest
    #######################
    copyfile('../hls-template/firmware/myproject.h', '{}/firmware/myproject.h'.format(yamlConfig['OutputDir']))
    copyfile('../hls-template/build_prj.tcl', '{}/build_prj.tcl'.format(yamlConfig['OutputDir']))
    copyfile('../hls-template/myproject.tcl', '{}/myproject.tcl'.format(yamlConfig['OutputDir']))


    #tarball output
    with tarfile.open(yamlConfig['OutputDir'] + '.tar.gz', mode='w:gz') as archive:
        archive.add(yamlConfig['OutputDir'], recursive=True)

if __name__ == "__main__":
    main();    
