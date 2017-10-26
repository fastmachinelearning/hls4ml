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

#Extract model architecture from json
with open(json_file_name) as json_file:
    model_arch = json.load(json_file)
#print(model_arch)

#Define layers to skip for conversion to HLS
skip_layers = ['InputLayer', 'Dropout', 'Flatten'] 

#Loop through layers
layer_counter = 0;
for layer in model_arch["config"]["layers"]:
    if layer["class_name"] in skip_layers:
        continue 

    layer_counter = layer_counter+1

    #Extract name for finding weights and biases
    print("PARSED LAYER NAME",layer["name"])

    #Extract typoe of activation and number of nodes
    for config,config_value in layer["config"].items():
        if(config=="activation"):
            print("PARSED ACTIVATION",config_value)
        if(config=="units"):
            print("PARSED NUM OF NODES",config_value)

    #Get Weights and biases from h5 file
    print_array_to_cpp("b{}".format(layer_counter),h5File['/{}/{}/bias:0'.format(layer["name"],layer["name"])][()])
    print_array_to_cpp("w{}".format(layer_counter),h5File['/{}/{}/kernel:0'.format(layer["name"],layer["name"])][()])


#tarball output
with tarfile.open(out_dir_name + '.tar.gz', mode='w:gz') as archive:
    archive.add(out_dir_name, recursive=True)
