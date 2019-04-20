
import tarfile
import yaml
from shutil import copyfile
import numpy as np
import os
import re
from collections import OrderedDict

#######################################
## Config module
#######################################
def parse_config(config_file) :

    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

#######################################
## Print weight array to C++
#######################################
def print_array_to_cpp(name, a, odir, i_part = 0, n_part = 1, i_subout = 0, n_subout = 1):

    f=open("{}/firmware/weights/{}.h".format(odir,name),"w")

    #count zeros
    zero_ctr = 0
    for x in np.nditer(a, order='C'):
        if x == 0:
            zero_ctr += 1

    #meta data
    f.write("//Numpy array shape {}\n".format(a.shape))
    f.write("//Min {:.12f}\n".format(np.min(a)))
    f.write("//Max {:.12f}\n".format(np.max(a)))
    f.write("//Number of zeros {}\n".format(zero_ctr))
    f.write("\n")

    #c++ variable
    if re.match(r"^w\d*$", name) or re.match(r"^a\d*$", name):
        if n_part > 1:
            f.write("weight_default_t {}_{}".format(name,i_part))
        else:
            f.write("weight_default_t {}".format(name))
    elif re.match(r"^b\d*$", name):
        if n_part > 1:
            f.write("bias_default_t {}_{}".format(name,i_part))
        else:
            f.write("bias_default_t {}".format(name))
    elif re.match(r"^beta\d*$", name):
        f.write("beta_default_t {}".format(name))
    elif re.match(r"^mean\d*$", name):
        f.write("mean_default_t {}".format(name))
    elif re.match(r"^scale\d*$", name):
        f.write("scale_default_t {}".format(name))
    else:
        raise Exception('ERROR: Unkown weights type')

    #hls doesn't like 3d arrays... unrolling to 1d
    #also doing for all (including 2d) arrays now
    f.write("[{}]".format(np.prod(a.shape)))
    f.write(" = {")

    #fill c++ array.
    #not including internal brackets for multidimensional case
    i=0
    for x in np.nditer(a, order='C'):
        if i==0:
            f.write("%.12f" % x)
        else:
            f.write(", %.12f" % x)
        i=i+1
    f.write("};\n")
    f.close()

    return zero_ctr

def write_project_cpp(model):
    ###################
    ## myproject.cpp
    ###################

    filedir = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(filedir,'../hls-template/firmware/myproject.cpp'),'r')
    fout = open('{}/firmware/{}.cpp'.format(model.get_output_dir(), model.get_project_name()),'w')

    model_inputs = model.get_input_variables()
    model_outputs = model.get_output_variables()

    indent = '    '

    for line in f.readlines():
        #Add headers to weights and biases
        if 'myproject' in line:
            newline = line.replace('myproject', model.get_project_name())
        elif '//hls-fpga-machine-learning insert header' in line:
            inputs_str = ', '.join([i.definition_cpp() for i in model_inputs])
            outputs_str = ', '.join([o.definition_cpp() for o in model_outputs])
            insize_str = ', '.join(['unsigned short &const_size_in_{}'.format(i) for i in range(1, len(model_inputs) + 1)])
            outsize_str = ', '.join(['unsigned short &const_size_out_{}'.format(i) for i in range(1, len(model_outputs) + 1)])

            newline = ''
            newline += indent + inputs_str + ',\n'
            newline += indent + outputs_str + ',\n'
            newline += indent + insize_str + ',\n'
            newline += indent + outsize_str + '\n'

        elif '//hls-fpga-machine-learning insert weights' in line:
            newline = line
            for layer in model.get_layers():
                for w in layer.get_weights():
                    newline += '#include "weights/{}.h"\n'.format(w.name)

        #Add input/output type
        elif '//hls-fpga-machine-learning insert IO' in line:
            newline = line
            all_inputs = [i.name for i in model_inputs]
            all_outputs = [o.name for o in model_outputs]
            if model.get_config_value("IOType") == "io_parallel":
                for i in model_inputs: newline += indent + '#pragma HLS ARRAY_RESHAPE variable={} complete dim=0 \n'.format(i.name)
                for o in model_outputs: newline += indent + '#pragma HLS ARRAY_RESHAPE variable={} complete dim=0 \n'.format(o.name)
                newline += indent + '#pragma HLS INTERFACE ap_vld port={},{} \n'.format(','.join(all_inputs), ','.join(all_outputs))
                newline += indent + '#pragma HLS PIPELINE \n'
            if model.get_config_value("IOType") == "io_serial":
                newline += indent + '#pragma HLS INTERFACE axis port={},{} \n'.format(','.join(all_inputs), ','.join(all_outputs))
                newline += indent + '#pragma HLS DATAFLOW \n'

            inval_str = '\n    '.join(['const_size_in_{} = {};'.format(i, inp.size_cpp()) for i, inp in enumerate(model_inputs, 1)])
            outval_str = '\n    '.join(['const_size_out_{} = {};'.format(i, out.size_cpp()) for i, out in enumerate(model_outputs, 1)])
            newline += '\n' + indent + inval_str
            newline += '\n' + indent + outval_str

        elif '//hls-fpga-machine-learning insert layers' in line:
            newline = line + '\n'

            for i in range(1,len(layer_list)+1):
                
                #Input to compute_layer

                #First layer and dense
                if i==1 and (layer_list[i-1]['class_name']=='Dense' or layer_list[i-1]['class_name']=='BinaryDense' or layer_list[i-1]['class_name']=='TernaryDense' or (layer_list[i-1]['class_name']=='BatchNormalization' and is_dense)):
                    input_type = 'input_t'
                    input_object = 'data'
                    n_in = 'N_INPUTS'
                #Layer is Dense and previous layer was Conv1D
                elif layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv1D':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'Y_OUTPUTS_{}*N_FILT_{}'.format(i-1,i-1)
                #Layer is Dense and previous layer was Conv2D
                elif layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv2D':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'IN_HEIGHT_{}*IN_WIDTH_{}*N_FILT_{}'.format(i-1,i-1,i-1)
                #Layer is Dense, BatchNormalization or Activation
                elif layer_list[i-1]['class_name']=='Dense' or layer_list[i-1]['class_name']=='BinaryDense' or layer_list[i-1]['class_name']=='TernaryDense' or layer_list[i-1]['class_name'] in activation_layers:
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'N_LAYER_{}'.format(i-1)
                elif is_dense and layer_list[i-1]['class_name']=='BatchNormalization':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'N_LAYER_{}'.format(i-1)
                    n_filt = 'N_FILT_{}'.format(i-1)
                elif (i==1 and layer_list[i-1]['class_name']=='BatchNormalization' and is_conv2d):
                    input_type = 'input_t'
                    input_object = 'data'
                    in_height = 'IN_HEIGHT_{}'.format(i)
                    in_width = 'IN_WIDTH_{}'.format(i)
                    n_chan = 'N_FILT_{}'.format(i)
                elif is_conv2d and layer_list[i-1]['class_name']=='BatchNormalization':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'OUT_HEIGHT_{}*OUT_WIDTH_{}*N_FILT_{}'.format(i-1,i-1,i-1)
                    n_filt = 'N_FILT_{}'.format(i-1)
                #First layer and Conv1D
                elif (i==1 and layer_list[i-1]['class_name']=='Conv1D'):
                    input_type = 'input_t'
                    input_object = 'data'
                    y_in = 'Y_INPUTS_{}'.format(i)
                    n_chan = 'N_CHAN_{}'.format(i)
                #Layer is Conv1D
                elif layer_list[i-1]['class_name']=='Conv1D':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    y_in = 'Y_INPUTS_{}'.format(i)
                    n_chan = 'N_CHAN_{}'.format(i)
                #First layer and Conv2D
                elif (i==1 and layer_list[i-1]['class_name']=='Conv2D'):
                    input_type = 'input_t'
                    input_object = 'data'
                    in_height = 'IN_HEIGHT_{}'.format(i)
                    in_width = 'IN_WIDTH_{}'.format(i)
                    n_chan = 'N_CHAN_{}'.format(i)
                #Layer is Conv2D
                elif layer_list[i-1]['class_name']=='Conv2D':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    in_height = 'IN_HEIGHT_{}'.format(i)
                    in_width = 'IN_WIDTH_{}'.format(i)
                    n_chan = 'N_CHAN_{}'.format(i)
                #Pooling layer
                elif 'Pooling' in layer_list[i-1]['class_name']:
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    output_object = 'layer{}_out'.format(i)
                    in_height = 'IN_HEIGHT_{}'.format(i)
                    in_width = 'IN_WIDTH_{}'.format(i)
                    out_height = 'OUT_HEIGHT_{}'.format(i)
                    out_width = 'OUT_WIDTH_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                #Currently doesn't allow all combinations


                #Outputs of compute_layer and activation 
                if i==len(layer_list) and (layer_list[i-1]['class_name']=='Dense' or layer_list[i-1]['class_name']=='BinaryDense' or layer_list[i-1]['class_name']=='TernaryDense'):
                    output_type = 'result_t'
                    output_object = 'res'
                    n_out = 'N_OUTPUTS'
                    if layer_list[i-1]['class_name'] in activation_layers: input_type = 'result_t'
                elif i==len(layer_list) and layer_list[i-1]['class_name'] in activation_layers and is_dense:
                    output_type = 'result_t'
                    output_object = 'res'
                    n_out = 'N_OUTPUTS'
                    input_type = 'result_t'
                elif i==len(layer_list) and is_dense and layer_list[i-1]['class_name']=='BatchNormalization':
                    output_type = 'result_t'
                    output_object = 'res'
                    n_out = 'N_OUTPUTS'
                elif i==len(layer_list) and is_conv2d and layer_list[i-1]['class_name']=='BatchNormalization':
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    out_height = 'OUT_HEIGHT_{}'.format(i)
                    out_width = 'OUT_WIDTH_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                elif(i==len(layer_list)-1 and is_dense and layer_list[i-1]['class_name']=='BatchNormalization' and layer_list[i]['class_name'] in activation_layers):
                    output_type = 'result_t'
                    output_object = 'layer{}_out'.format(i)
                    n_out = 'N_OUTPUTS' 
                elif layer_list[i-1]['class_name']=='Dense' or layer_list[i-1]['class_name']=='BinaryDense' or layer_list[i-1]['class_name']=='TernaryDense' or (layer_list[i-1]['class_name']=='BatchNormalization' and is_dense) or (layer_list[i-1]['class_name'] in activation_layers and is_dense):
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    n_out = 'N_LAYER_{}'.format(i)
                elif layer_list[i-1]['class_name']=='Conv1D':
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    y_out = 'Y_OUTPUTS_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                elif layer_list[i-1]['class_name']=='Conv2D' or (is_conv2d and layer_list[i-1]['class_name']=='BatchNormalization'):
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    out_height = 'OUT_HEIGHT_{}'.format(i)
                    out_width = 'OUT_WIDTH_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                #Currently assumes end with dense

                if( i!=len(layer_list) ):
                    if layer_list[i-1]['class_name']=='Dense' or layer_list[i-1]['class_name']=='BinaryDense' or layer_list[i-1]['class_name']=='TernaryDense' or (layer_list[i-1]['class_name']=='BatchNormalization' and is_dense) or (layer_list[i-1]['class_name'] in activation_layers and is_dense):
                        newline += '    {} layer{}_out[{}];\n'.format(output_type,i,n_out)
                    elif layer_list[i-1]['class_name']=='Conv1D' or 'Pooling1D' in layer_list[i-1]['class_name']:
                        newline += '    {} layer{}_out[{}*{}];\n'.format(output_type,i,y_out,n_filt)
                    elif layer_list[i-1]['class_name']=='Conv2D' or 'Pooling2D' in layer_list[i-1]['class_name']:
                        newline += '    {} layer{}_out[{}*{}*{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                    elif layer_list[i-1]['class_name']=='BatchNormalization' and is_conv2d:
                        if i!= 1: newline += '    {} layer{}_out[{}*{}*{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                        else: newline += '    {} layer{}_out[{}*{}*{}];\n'.format(output_type,i,in_height,in_width,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=layer{}_out complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=layer{}_out depth=1\n'.format(i)

                #github Issue 53
                #Compute Dense layer
                #if layer_list[i-1]['activation'] == "linear" and layer_list[i-1]['class_name']=='Dense':
                #    newline += '    nnet::compute_layer<{}, {}, config{}>({}, {}, w{}, b{});\n'.format(input_type, output_type, i, input_object, output_object, i, i)
                #elif layer_list[i-1]['class_name']=='Dense':
                if layer_list[i-1]['class_name']=='Dense' or layer_list[i-1]['class_name']=='BinaryDense' or layer_list[i-1]['class_name']=='TernaryDense':
                    newline += '    {} logits{}[{}];\n'.format(output_type,i,n_out)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} depth=1\n'.format(i)
                    
                    if layer_list[i-1]['n_part']==1 or yamlConfig["IOType"]=="io_serial":
                        # Use one layer if there's only 1 partition, or if we're using serial mode
                        newline += '    nnet::compute_layer<{}, {}, config{}>({}, logits{}, w{}, b{});\n'.format(input_type, output_type, i, input_object, i, i, i, i)
                    else:
                        # initialize arrays for sublayer outputs
                        newline += '    compute_layer{}({}, logits{});\n'.format(i, input_object, i)
                        sublayerline = 'void compute_layer{}({} {}[{}], {} logits{}[{}]) {{\n'.format(i,input_type, input_object, n_in, output_type, i, n_out)
                        sublayerline_h = 'void compute_layer{}({} {}[{}], {} logits{}[{}]);\n'.format(i,input_type, input_object, n_in, output_type, i, n_out)
                        sublayerlines_h.append(sublayerline_h)
                        for i_part in range(0, layer_list[i-1]['n_part']):
                            n_subout = layer_list[i-1]['n_subout'][i_part]
                            sublayerline += '    {} logits{}_{}[{}];\n'.format(output_type,i,i_part,n_subout)                        
                            if yamlConfig["IOType"] == "io_parallel": sublayerline += '    #pragma HLS ARRAY_PARTITION variable=logits{}_{} complete dim=0\n'.format(i,i_part)
                            if yamlConfig["IOType"] == "io_serial":   sublayerline += '    #pragma HLS STREAM variable=logits{}_{} depth=1\n'.format(i,i_part)

                        # initialize arrays for merged partial outputs 
                        for i_part in range(1, layer_list[i-1]['n_part']-1):
                            n_mergeout = sum([layer_list[i-1]['n_subout'][kk] for kk in range(0, i_part+1)])
                            sublayerline += '    {} logits{}_0to{}[{}];\n'.format(output_type,i,i_part,n_mergeout)                        
                            if yamlConfig["IOType"] == "io_parallel": sublayerline += '    #pragma HLS ARRAY_PARTITION variable=logits{}_0to{} complete dim=0\n'.format(i,i_part)
                            if yamlConfig["IOType"] == "io_serial":   sublayerline += '    #pragma HLS STREAM variable=logits{}_0to{} depth=1\n'.format(i,i_part)
                        # compute sublayer outputs
                        for i_part in range(0, layer_list[i-1]['n_part']):
                            sublayerline += '    nnet::compute_layer<{}, {}, config{}_{}>({}, logits{}_{}, w{}_{}, b{}_{});\n'.format(input_type, output_type, i, i_part, input_object, i, i_part, i, i_part, i, i_part)   

                        # merge sublayer outputs
                        for i_part in range(0, layer_list[i-1]['n_part']-1):
                            n_subout = layer_list[i-1]['n_subout'][i_part+1]
                            n_mergeout = sum([layer_list[i-1]['n_subout'][kk] for kk in range(0, i_part+1)])
                            if layer_list[i-1]['n_part']==2:
                                sublayerline += '    nnet::merge<{}, {}, {}>(logits{}_{}, logits{}_{}, logits{});\n'.format(output_type, n_mergeout, n_subout, i, i_part, i, i_part+1, i)
                            elif i_part==0: 
                                sublayerline += '    nnet::merge<{}, {}, {}>(logits{}_{}, logits{}_{}, logits{}_0to{});\n'.format(output_type, n_mergeout, n_subout, i, i_part, i, i_part+1, i, i_part+1)
                            elif i_part==layer_list[i-1]['n_part']-2:
                                sublayerline += '    nnet::merge<{}, {}, {}>(logits{}_0to{}, logits{}_{}, logits{});\n'.format(output_type, n_mergeout, n_subout, i, i_part, i, i_part+1, i)
                            else:
                                sublayerline += '    nnet::merge<{}, {}, {}>(logits{}_0to{}, logits{}_{}, logits{}_0to{});\n'.format(output_type, n_mergeout, n_subout, i, i_part, i, i_part+1, i, i_part+1)
                        sublayerline += '}\n'
                        sublayerlines.append(sublayerline)
                    
                elif layer_list[i-1]['class_name']=='Conv1D':
                    if i>1 and layer_list[i-2]['class_name']=='Conv1D':
                        newline += '    {} conv_layer{}_in[{}][{}];\n'.format(input_type,i,y_in,n_chan)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv_layer{}_in complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv_layer{}_in depth=1\n'.format(i)
                        newline += '    nnet::unflatten<{}, {}, {}>({}, conv_layer{}_in);\n'.format(input_type, y_in, n_chan, input_object, i)                              
                        newline += '    {} conv_layer{}_out[{}][{}];\n'.format(output_type,i,y_out,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_1d<{}, {}, config{}>(conv_layer{}_in, conv_layer{}_out, w{}, b{});\n'.format(input_type, output_type, i, i, i, i, i, i)  
                    else:                        
                        newline += '    {} conv_layer{}_out[{}][{}];\n'.format(output_type,i,y_out,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_1d<{}, {}, config{}>({}, conv_layer{}_out, w{}, b{});\n'.format(input_type, output_type, i, input_object, i, i, i, i)
                    newline += '    {} logits{}[{}*{}];\n'.format(output_type,i,y_out,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} complete depth=1\n'.format(i)
                    newline += '    nnet::flatten<{}, {}, {}>(conv_layer{}_out, logits{});\n'.format(input_type, y_out, n_filt, i, i)
                elif layer_list[i-1]['class_name']=='Conv2D':
                    if i>1 and (layer_list[i-2]['class_name']=='Conv2D' or layer_list[i-2]['class_name']=='BatchNormalization'):
                        newline += '    {} conv2d_layer{}_in[{}][{}][{}];\n'.format(input_type,i,in_height,in_width,n_chan)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv2d_layer{}_in complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv2d_layer{}_in depth=1\n'.format(i)
                        newline += '    nnet::unflatten<{}, {}, {}, {}>({}, conv2d_layer{}_in);\n'.format(input_type, in_height, in_width, n_chan, input_object, i)                              
                        newline += '    {} conv2d_layer{}_out[{}][{}][{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv2d_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv2d_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_2d<{}, {}, config{}>(conv2d_layer{}_in, conv2d_layer{}_out, w{}, b{});\n'.format(input_type, output_type, i, i, i, i, i, i)  
                    else:                        
                        newline += '    {} conv2d_layer{}_out[{}][{}][{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv2d_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv2d_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_2d<{}, {}, config{}>({}, conv2d_layer{}_out, w{}, b{});\n'.format(input_type, output_type, i, input_object, i, i, i, i)
                    newline += '    {} logits{}[{}*{}*{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} complete depth=1\n'.format(i)
                    newline += '    nnet::flatten<{}, {}, {}, {}>(conv2d_layer{}_out, logits{});\n'.format(output_type, out_height, out_width, n_filt, i, i)
                elif layer_list[i-1]['class_name'] == 'BatchNormalization' and is_dense:
                    newline += '    nnet::normalize<{}, {}, config{}>({}, {}, scale{}, beta{}, mean{});\n'.format(input_type, output_type, i, input_object, output_object, i, i, i)
                elif i==1 and layer_list[i-1]['class_name'] == 'BatchNormalization' and is_conv2d:
                    newline += '    {} logits{}[{}*{}*{}];\n'.format(output_type,i,in_height,in_width,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} complete depth=1\n'.format(i)
                    newline += '    nnet::flatten<{}, {}, {}, {}>({}, logits{});\n'.format(input_type, in_height, in_width, n_filt, input_object, i)
                    newline += '    nnet::normalize<{}, {}, config{}>(logits{}, {}, scale{}, beta{}, mean{});\n'.format(output_type, output_type, i, i, output_object, i, i, i)
                elif layer_list[i-1]['class_name'] == 'BatchNormalization' and is_conv2d:
                    newline += '    nnet::normalize<{}, {}, config{}>({}, {}, scale{}, beta{}, mean{});\n'.format(input_type, output_type, i, input_object, output_object, i, i, i)
                elif 'Pooling' in layer_list[i-1]['class_name']:
                    info = layer_list[i-1]['class_name'].split('Pooling')
                    d = int(info[1].split('D')[0]) # n dimensions
                    if d == 1:
                        newline += '    nnet::pooling1d<{}, config{}>({}, {});\n'.format(input_type, i, input_object, output_object)
                    elif d == 2:
                        # Unflatten if needed: if the last layer is activation or batchnorm
                        unflatten = layer_list[i-2]['class_name'] in activation_layers
                        unflatten |= 'activation' in list(layer_list[i-2].keys())
                        unflatten |= layer_list[i-2]['class_name'] == 'BatchNormalization'
                        if unflatten:
                            # Add the unflatten layer
                            inshape = ''.join('[{0}]'.format(dim) for dim in [in_height, in_width, n_filt])
                            newline += '    {} pool2d_layer{}_in{};\n'.format(input_type, i, inshape)
                            if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=pool2d_layer{}_in complete dim=0\n'.format(i)
                            if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=pool2d_layer{}_in depth=1\n'.format(i)
                            newline += '    nnet::unflatten<{}, {}, {}, {}>({}, pool2d_layer{}_in);\n'.format(input_type, in_height, in_width, n_filt, input_object, i)                              
                            outshape = ''.join('[{0}]'.format(dim) for dim in [out_height, out_width, n_filt])
                            newline += '    {} pool2d_layer{}_out{};\n'.format(input_type, i, outshape)
                            if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=pool2d_layer{}_out complete dim=0\n'.format(i)
                            if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=pool2d_layer{}_out depth=1\n'.format(i)
                            # Do the pooling layer
                            newline += '    nnet::pooling2d<{}, config{i}>(pool2d_layer{i}_in, pool2d_layer{i}_out);\n'.format(input_type, i=i)
                        else:
                            newline += '    nnet::pooling2d<{}, config{i}>({}, {});\n'.format(input_type, i, input_object, output_object)
                        # Flatten the pooling output
                        newline += '    nnet::flatten<{}, {}, {}, {}>(pool2d_layer{}_out, layer{}_out);\n'.format(input_type, out_height, out_width, n_filt, i, i)
                        
                #Activations
                if layer_list[i-1]['class_name'] in activation_layers or 'activation' in list(layer_list[i-1].keys()):
                    if layer_list[i-1]['class_name'] not in activation_layers:
                        act_input_type = output_type
                        act_input_object = "logits" + str(i)
                    else:
                        act_input_type = input_type
                        act_input_object = input_object
                    
                    activation_name = layer_list[i-1]['activation']+'_config'+str(i)
                    activation_param = layer_list[i-1].get('activ_param')
                    if layer_list[i-1]['activation'] == "relu":
                        newline += '    nnet::relu<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "LeakyReLU":
                        newline += '    nnet::leaky_relu<{}, {}, {}>({}, {}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, activation_param, output_object)
                    elif layer_list[i-1]['activation'] == "ThresholdedReLU":
                        newline += '    nnet::thresholded_relu<{}, {}, {}>({}, {}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, activation_param, output_object)
                    elif layer_list[i-1]['activation'].lower() == "elu":
                        if activation_param:
                            newline += '    nnet::elu<{}, {}, {}>({}, {}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, activation_param, output_object)
                        else:
                            newline += '    nnet::elu<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "selu":
                        newline += '    nnet::selu<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "PReLU":
                        newline += '    nnet::prelu<{}, {}, {}>({}, a{}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, i, output_object)
                    elif layer_list[i-1]['activation'] == "softmax":
                        newline += '    nnet::softmax<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "sigmoid":
                        newline += '    nnet::sigmoid<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "hard_sigmoid":
                        newline += '    nnet::hard_sigmoid<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "tanh":
                        newline += '    nnet::tanh<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "linear": 
                        #github Issue 53
                        newline += '    nnet::linear<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "softsign":
                        newline += '    nnet::softsign<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "softplus":
                        newline += '    nnet::softplus<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object)
                    elif layer_list[i-1]['activation'] == "binary_tanh":	
                        newline += '    nnet::binary_tanh<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object) 
                    elif layer_list[i-1]['activation'] == "ternary_tanh":	
                        newline += '    nnet::ternary_tanh<{}, {}, {}>({}, {});\n'.format(act_input_type, output_type, activation_name, act_input_object, output_object) 
                    else:
                        raise Exception('ERROR: MISSING ACTIVATION')

                newline += '\n'



        #Just copy line
        else:
            newline = line

        fout.write(newline)

    f.close()
    fout.close()

def write_project_header(model):
    #######################
    ## myproject.h
    #######################

    filedir = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(filedir,'../hls-template/firmware/myproject.h'),'r')
    fout = open('{}/firmware/{}.h'.format(model.get_output_dir(), model.get_project_name()),'w')

    model_inputs = model.get_input_variables()
    model_outputs = model.get_output_variables()

    indent = '    '

    for line in f.readlines():

        if 'MYPROJECT' in line:
            newline = line.replace('MYPROJECT',format(model.get_project_name().upper()))
        elif 'void myproject(' in line:
            newline = 'void {}(\n'.format(model.get_project_name())
        elif '//hls-fpga-machine-learning insert header' in line:
            inputs_str = ', '.join([i.definition_cpp() for i in model_inputs])
            outputs_str = ', '.join([o.definition_cpp() for o in model_outputs])
            insize_str = ', '.join(['unsigned short &const_size_in_{}'.format(i) for i in range(1, len(model_inputs) + 1)])
            outsize_str = ', '.join(['unsigned short &const_size_out_{}'.format(o) for o in range(1, len(model_outputs) + 1)])

            newline = ''
            newline += indent + inputs_str + ',\n'
            newline += indent + outputs_str + ',\n'
            newline += indent + insize_str + ',\n'
            newline += indent + outsize_str + '\n'
        else:
            newline = line
        fout.write(newline)

    f.close()
    fout.close()

def write_parameters(model):
    filedir = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(filedir,'../hls-template/firmware/parameters.h'),'r')
    fout = open('{}/firmware/parameters.h'.format(model.get_output_dir()),'w')

    for line in f.readlines():

        #Insert numbers
        if '//hls-fpga-machine-learning insert numbers' in line:
            newline = line
            newline += 'typedef {precision} accum_default_t;\n'.format(precision=model.get_default_precision())
            newline += 'typedef {precision} weight_default_t;\n'.format(precision=model.get_default_precision())
            newline += 'typedef {precision} bias_default_t;\n'.format(precision=model.get_default_precision())
            newline += 'typedef {precision} input_t;\n'.format(precision=model.get_default_precision())
            newline += 'typedef {precision} result_t;\n'.format(precision=model.get_default_precision())
            newline += 'typedef {precision} beta_default_t;\n'.format(precision=model.get_default_precision())
            newline += 'typedef {precision} mean_default_t;\n'.format(precision=model.get_default_precision())
            newline += 'typedef {precision} scale_default_t;\n'.format(precision=model.get_default_precision())

            newline += '\n'

            numbers = OrderedDict.fromkeys([layer.get_numbers_cpp() for layer in model.get_layers()])
            newline += ''.join(numbers)

        elif '//hls-fpga-machine-learning insert layer-precision' in line:
            newline = line
            for layer in model.get_layers():
                newline += layer.precision_cpp() + '\n'

        elif "//hls-fpga-machine-learning insert layer-config" in line:
            newline = line
            for layer in model.get_layers():
                config = layer.config_cpp()
                if config:
                    newline += config + '\n'
        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()

def write_weights(model):
    for layer in model.get_layers():
        for weights in layer.get_weights():
            print_array_to_cpp(weights.name, weights.data, model.get_output_dir())

def write_test_bench(model):
    ###################
    ## test bench
    ###################

    filedir = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(filedir,'../hls-template/myproject_test.cpp'),'r')
    fout = open('{}/{}_test.cpp'.format(model.get_output_dir(), model.get_project_name()),'w')

    for line in f.readlines():

        #Insert numbers
        if 'myproject' in line:
            newline = line.replace('myproject', model.get_project_name())
        elif '//hls-fpga-machine-learning insert data' in line:
            newline = line
            for inp in model.get_input_variables():
                input_str = '  ' + inp.definition_cpp() + ' = {};\n'
                default_val = ','.join(str(i) for i in [0] * inp.size())
                newline += input_str.format('{' + default_val + '}')
            for out in model.get_output_variables():
                output_str = '  ' + out.definition_cpp() + ' = {};\n'
                default_val = ','.join(str(o) for o in [0] * out.size())
                newline += output_str.format('{' + default_val + '}')
        elif '//hls-fpga-machine-learning insert top-level-function' in line:
            newline = line

            size_str = '  unsigned short {},{};\n'
            input_size_vars = ','.join(['size_in{}'.format(i) for i in range(1, len(model.get_input_variables()) + 1)])
            output_size_vars = ','.join(['size_out{}'.format(o) for o in range(1, len(model.get_output_variables()) + 1)])
            newline += size_str.format(input_size_vars, output_size_vars)

            input_vars = ','.join([i.name for i in model.get_input_variables()])
            output_vars = ','.join([o.name for o in model.get_output_variables()])
            top_level = '  {}({},{},{},{});\n'.format(model.get_project_name(), input_vars, output_vars, input_size_vars, output_size_vars)
            newline += top_level
        elif '//hls-fpga-machine-learning insert output' in line:
            newline = line
            for out in model.get_output_variables():
                newline += '  for(int i = 0; i < {}; i++) {{\n'.format(out.size_cpp())
                newline += '    std::cout << {}[i] << " ";\n'.format(out.name)
                newline += '  }\n'
                newline += '  std::cout << std::endl;\n'
        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()

def write_build_script(model):
    ###################
    # build_prj.tcl
    ###################

    filedir = os.path.dirname(os.path.abspath(__file__))
    nnetdir = os.path.abspath(os.path.join(filedir, "../nnet_utils"))
    relpath = os.path.relpath(nnetdir, start=model.get_output_dir())

    f = open(os.path.join(filedir,'../hls-template/build_prj.tcl'),'r')
    fout = open('{}/build_prj.tcl'.format(model.get_output_dir()),'w')

    for line in f.readlines():

        line = line.replace('myproject',model.get_project_name())
        line = line.replace('nnet_utils', relpath)

        if 'set_part {xc7vx690tffg1927-2}' in line:
            line = 'set_part {{{}}}\n'.format(model.get_config_value('XilinxPart'))
        elif 'create_clock -period 5 -name default' in line:
            line = 'create_clock -period {} -name default\n'.format(model.get_config_value('ClockPeriod'))

        fout.write(line)
    f.close()
    fout.close()

def write_tar(model):
    ###################
    # Tarball output
    ###################

    with tarfile.open(model.get_output_dir() + '.tar.gz', mode='w:gz') as archive:
        archive.add(model.get_output_dir(), recursive=True)

def write_hls(model):
    write_project_cpp(model)
    write_project_header(model)
    write_weights(model)
    write_parameters(model)
    write_test_bench(model)
    write_build_script(model)
    write_tar(model)
