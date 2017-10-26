import numpy as np
import h5py
import os
import tarfile
import json


#####################
## User parameters
#####################
json_file_name = "h3l.json"
h5_file_name = "KERAS_check_best_model_weights.h5"
out_dir_name = "my-hls-dir"



#######################################
## Print a bias or weight array to C++
#######################################
def print_array_to_cpp(name, a):
    #put output in subdir for tarballing later

    f=open("{}/firmware/weights/{}.h".format(out_dir_name,name),"w")

    #meta data
    f.write("//Numpy array shape {}\n".format(a.shape))
    f.write("//Min {}\n".format(np.min(a)))
    f.write("//Max {}\n".format(np.max(a)))
    f.write("\n")
    
    #c++ variable 
    f.write("weight_t {}".format(name))
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
    f.write("};")
    f.close()




######################
##  Do translation
######################
if not os.path.isdir("{}/firmware/weights".format(out_dir_name)):
    os.makedirs("{}/firmware/weights".format(out_dir_name))

h5File = h5py.File(h5_file_name)

#This is a list of dictionaries to hold all the layer info we need to generate HLS
layer_list = []

#Extract model architecture from json
with open(json_file_name) as json_file:
    model_arch = json.load(json_file)
#print(model_arch)

#Define layers to skip for conversion to HLS
skip_layers = ['InputLayer', 'Dropout', 'Flatten'] 

#Loop through layers
layer_counter = 0;
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
    print_array_to_cpp("w{}".format(layer_counter), weights)
    print_array_to_cpp("b{}".format(layer_counter), biases)

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
fout = open('{}/firmware/myproject.cpp'.format(out_dir_name),'w')

for line in f.readlines():
    #Add headers to weights and biases
    if '//hls-fpga-machine-learning insert weights' in line:
        newline = line
        for i in range(1,len(layer_list)+1):
            newline = newline + '#include "weights/w{}.h";\n'.format(i)
            newline = newline + '#include "weights/b{}.h";\n'.format(i)

    #Add layers
    elif '//hls-fpga-machine-learning insert layers' in line:
        newline = line;
        for i in range(1,len(layer_list)+1):
            
            if(i==1):
                input_type = 'input_t'
                input_object = 'data'
                n_in = 'N_INPUTS'
            else:
                input_type = 'layer{}_t'.format(i-1)
                input_object = 'layer{}_out'.format(i-1)
                n_in = 'N_LAYER_{}'.format(i-1);

            if(i==len(layer_list)):
                output_type = 'result_t'
                output_object = 'res'
                n_out = 'N_OUPUTS'
            else:
                output_type = 'layer{}_t'.format(i)
                output_object = 'layer{}_out'.format(i)
                n_out = 'N_LAYER_{}'.format(i)

            newline = newline + '    layer{}_t logits{}[N_LAYER_{}];\n'.format(i,i,i)
            newline = newline + '    layer{}_t layer{}_out[N_LAYER_{}];\n'.format(i,i,i)
            newline = newline + '    #pragma HLS ARRAY_PARTITION variable=logits{} complete\n'.format(i)
            newline = newline + '    #pragma HLS ARRAY_PARTITION variable=layer{}_out complete\n'.format(i)
            newline = newline + '    nnet::compute_layer<{}, {}, weight_t, bias_t, accum_t, {}, {}>({}, logits{}, w{}, b{});\n'.format(input_type, output_type, n_in, n_out, input_object, i, i, i, i)
            
            if layer_list[i-1]['activation'] == "relu":
                newline = newline + '    nnet::relu<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, n_out, i, output_object)
            elif layer_list[i-1]['activation'] =="softmax":
                newline = newline + '    nnet::softmax<{}, {}, {}, 2048>(logits{}, {});\n'.format(output_type, output_type, n_out, i, output_object)
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


#tarball output
with tarfile.open(out_dir_name + '.tar.gz', mode='w:gz') as archive:
    archive.add(out_dir_name, recursive=True)
