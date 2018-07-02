import tarfile
import yaml
from shutil import copyfile
import numpy as np
import os

def hls_writer(layer_list, yamlConfig):

    filedir = os.path.dirname(os.path.abspath(__file__))

    ###################
    ## myproject.cpp
    ###################

    f = open(os.path.join(filedir,'../hls-template/firmware/myproject.cpp'),'r')
    fout = open('{}/firmware/{}.cpp'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

    for line in f.readlines():
        #Add headers to weights and biases
        if 'myproject' in line:
            newline = line.replace('myproject',yamlConfig['ProjectName'])
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='Conv1D':
            newline = line.replace('input_t data[N_INPUTS]','input_t data[Y_INPUTS_1][N_CHAN_1]')
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='Conv2D':
            newline = line.replace('input_t data[N_INPUTS]','input_t data[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1]')
        elif 'const_size_in   = N_INPUTS' in line and layer_list[0]['class_name']=='Conv1D':
            newline = line.replace('const_size_in   = N_INPUTS','const_size_in   = Y_INPUTS_1*N_CHAN_1')
        elif 'const_size_in   = N_INPUTS' in line and layer_list[0]['class_name']=='Conv2D':
            newline = line.replace('const_size_in   = N_INPUTS','const_size_in   = IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1')
        elif '//hls-fpga-machine-learning insert weights' in line:
            newline = line
            for i in range(1,len(layer_list)+1):
                newline += '#include "weights/w{}.h"\n'.format(i)
                newline += '#include "weights/b{}.h"\n'.format(i)

        #Add input/output type
        elif '//hls-fpga-machine-learning insert IO' in line:
            newline = line
            if yamlConfig["IOType"] == "io_parallel":
                newline += '    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 \n'
                newline += '    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 \n'
                newline += '    #pragma HLS INTERFACE ap_vld port=data,res \n'
                newline += '    #pragma HLS PIPELINE \n'
            if yamlConfig["IOType"] == "io_serial":
                newline += '    #pragma HLS INTERFACE axis port=data,res \n'
                newline += '    #pragma HLS DATAFLOW \n'

        #Add layers
        elif '//hls-fpga-machine-learning insert layers' in line:
            newline = line + '\n'
            for i in range(1,len(layer_list)+1):
                
                #Input to compute_layer

                #First layer and dense
                if(i==1 and layer_list[i-1]['class_name']=='Dense'):
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
                #Layer is Dense
                elif layer_list[i-1]['class_name']=='Dense':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    n_in = 'N_LAYER_{}'.format(i-1)
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
                elif layer_list[i-1]['class_name']=='Conv1D':
                    input_type = 'layer{}_t'.format(i-1)
                    input_object = 'layer{}_out'.format(i-1)
                    in_height = 'IN_HEIGHT_{}'.format(i)
                    in_width = 'IN_WIDTH_{}'.format(i)
                    n_chan = 'N_CHAN_{}'.format(i)
                #Currently doesn't allow all combinations


                #Outputs of compute_layer and activation 
                if(i==len(layer_list) and layer_list[i-1]['class_name']=='Dense'):
                    output_type = 'result_t'
                    output_object = 'res'
                    n_out = 'N_OUTPUTS'
                elif layer_list[i-1]['class_name']=='Dense':
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    n_out = 'N_LAYER_{}'.format(i)
                elif layer_list[i-1]['class_name']=='Conv1D':
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    y_out = 'Y_OUTPUTS_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                elif layer_list[i-1]['class_name']=='Conv2D':
                    output_type = 'layer{}_t'.format(i)
                    output_object = 'layer{}_out'.format(i)
                    out_height = 'OUT_HEIGHT_{}'.format(i)
                    out_width = 'OUT_WIDTH_{}'.format(i)
                    n_filt = 'N_FILT_{}'.format(i)
                #Currently assumes end with dense

                if(i!=len(layer_list)):
                    if layer_list[i-1]['class_name']=='Dense':
                        newline += '    {} layer{}_out[{}];\n'.format(output_type,i,n_out)
                    elif layer_list[i-1]['class_name']=='Conv1D':
                        newline += '    {} layer{}_out[{}*{}];\n'.format(output_type,i,y_out,n_filt)
                    elif layer_list[i-1]['class_name']=='Conv2D':
                        newline += '    {} layer{}_out[{}*{}*{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=layer{}_out complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=layer{}_out depth=1\n'.format(i)

                #github Issue 53
                #Compute Dense layer
                #if layer_list[i-1]['activation'] == "linear" and layer_list[i-1]['class_name']=='Dense':
                #    newline += '    nnet::compute_layer<{}, {}, config{}>({}, {}, w{}, b{});\n'.format(input_type, output_type, i, input_object, output_object, i, i)
                #elif layer_list[i-1]['class_name']=='Dense':
                if layer_list[i-1]['class_name']=='Dense':
                    newline += '    {} logits{}[{}];\n'.format(output_type,i,n_out)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} depth=1\n'.format(i)
                    newline += '    nnet::compute_layer<{}, {}, config{}>({}, logits{}, w{}, b{});\n'.format(input_type, output_type, i, input_object, i, i, i, i)
                elif layer_list[i-1]['class_name']=='Conv1D':
                    if i>1 and layer_list[i-2]['class_name']=='Conv1D':
                        newline += '    {} conv_layer{}_in[{}][{}];\n'.format(input_type,i,y_in,n_chan)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv_layer{}_in complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv_layer{}_in depth=1\n'.format(i)
                        newline += '    nnet::unflatten<{}, {}, {}>({}, conv_layer{}_in);\n'.format(input_type, y_in, n_chan, input_object, i)                              
                        newline += '    {} conv_layer{}_out[{}][{}];\n'.format(output_type,i,y_out,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_1d<{}, {}, config{}>(conv_layer{}_in, conv_layer{}_out, w{}, b{});\n'.format(input_type, input_type, i, i, i, i, i, i)  
                    else:                        
                        newline += '    {} conv_layer{}_out[{}][{}];\n'.format(output_type,i,y_out,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_1d<{}, {}, config{}>({}, conv_layer{}_out, w{}, b{});\n'.format(input_type, input_type, i, input_object, i, i, i, i)
                    newline += '    {} logits{}[{}*{}];\n'.format(output_type,i,y_out,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} complete depth=1\n'.format(i)
                    newline += '    nnet::flatten<{}, {}, {}>(conv_layer{}_out, logits{});\n'.format(input_type, y_out, n_filt, i, i)
                elif layer_list[i-1]['class_name']=='Conv2D':
                    if i>1 and layer_list[i-2]['class_name']=='Conv2D':
                        newline += '    {} conv2d_layer{}_in[{}][{}][{}];\n'.format(input_type,i,in_height,in_width,n_chan)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv2d_layer{}_in complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv2d_layer{}_in depth=1\n'.format(i)
                        newline += '    nnet::unflatten<{}, {}, {}, {}>({}, conv2d_layer{}_in);\n'.format(input_type, in_height, in_width, n_chan, input_object, i)                              
                        newline += '    {} conv2d_layer{}_out[{}][{}][{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv2d_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv2d_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_2d<{}, {}, config{}>(conv2d_layer{}_in, conv2d_layer{}_out, w{}, b{});\n'.format(input_type, input_type, i, i, i, i, i, i)  
                    else:                        
                        newline += '    {} conv2d_layer{}_out[{}][{}][{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                        if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=conv2d_layer{}_out complete dim=0\n'.format(i)
                        if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=conv2d_layer{}_out depth=1\n'.format(i)
                        newline += '    nnet::conv_2d<{}, {}, config{}>({}, conv2d_layer{}_out, w{}, b{});\n'.format(input_type, input_type, i, input_object, i, i, i, i)
                    newline += '    {} logits{}[{}*{}*{}];\n'.format(output_type,i,out_height,out_width,n_filt)
                    if yamlConfig["IOType"] == "io_parallel": newline += '    #pragma HLS ARRAY_PARTITION variable=logits{} complete dim=0\n'.format(i)
                    if yamlConfig["IOType"] == "io_serial":   newline += '    #pragma HLS STREAM variable=logits{} complete depth=1\n'.format(i)
                    newline += '    nnet::flatten<{}, {}, {}, {}>(conv2d_layer{}_out, logits{});\n'.format(input_type, out_height, out_width, n_filt, i, i)

                
                #Activations
                activation_name = layer_list[i-1]['activation']+'_config'+str(i)
                if layer_list[i-1]['activation'] == "relu":
                    newline += '    nnet::relu<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, activation_name, i, output_object)
                elif layer_list[i-1]['activation'] =="softmax":
                    newline += '    nnet::softmax<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, activation_name, i, output_object)
                elif layer_list[i-1]['activation'] =="sigmoid":
                    newline += '    nnet::sigmoid<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, activation_name, i, output_object)
                elif layer_list[i-1]['activation'] =="tanh":
                    newline += '    nnet::tanh<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, activation_name, i, output_object)
                elif layer_list[i-1]['activation'] =="linear": 
                    #github Issue 53
                    newline += '    nnet::linear<{}, {}, {}>(logits{}, {});\n'.format(output_type, output_type, activation_name, i, output_object)
                else:
                    raise Exception('ERROR: MISSING ACTIVATION')

                newline += '\n'

        #Just copy line
        else: 
            newline = line
        fout.write(newline)
    f.close()
    fout.close()

    ###################
    ## parameters.h
    ###################

    f = open(os.path.join(filedir,'../hls-template/firmware/parameters.h'),'r')
    fout = open('{}/firmware/parameters.h'.format(yamlConfig['OutputDir']),'w')

    dense_config_template = """struct config{index} : nnet::layer_config {{
        static const unsigned n_in = {n_in};
        static const unsigned n_out = {n_out};
        static const unsigned io_type = nnet::{iotype};
        static const unsigned reuse_factor = {reuse};
        static const unsigned n_zeros = {nzeros};
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        }};\n"""

    conv_config_template = """struct config{index} : nnet::conv_config {{
        static const unsigned pad_left = {pad_left};
        static const unsigned pad_right = {pad_right};
        static const unsigned y_in = {y_in};
        static const unsigned n_chan = {n_chan};
        static const unsigned y_filt = {y_filt};
        static const unsigned n_filt = {n_filt};
        static const unsigned stride = {stride};
        static const unsigned y_out = {y_out};
        static const unsigned reuse_factor = {reuse};
        static const unsigned n_zeros = {nzeros};
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        }};\n"""

    conv2d_config_template = """struct config{index} : nnet::conv2d_config {{
        static const unsigned pad_top = {pad_top};
        static const unsigned pad_bottom = {pad_bottom};
        static const unsigned pad_left = {pad_left};
        static const unsigned pad_right = {pad_right};
        static const unsigned in_height = {in_height};
        static const unsigned in_width = {in_width};
        static const unsigned n_chan = {n_chan};
        static const unsigned filt_height = {filt_height};
        static const unsigned filt_width = {filt_width};
        static const unsigned n_filt = {n_filt};
        static const unsigned stride_height = {stride_height};
        static const unsigned stride_width = {stride_width};
        static const unsigned out_height = {out_height};
        static const unsigned out_width = {out_width};
        static const unsigned reuse_factor = {reuse};
        static const unsigned n_zeros = {nzeros};
        static const bool store_weights_in_bram = false;
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
            newline += 'typedef {precision} accum_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline += 'typedef {precision} weight_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline += 'typedef {precision} bias_default_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline += 'typedef {precision} input_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            newline += 'typedef {precision} result_t;\n'.format(precision=yamlConfig["DefaultPrecision"])
            for i in range(1,len(layer_list)+1):

                if i==1 and layer_list[i-1]['class_name']=='Dense':
                    newline += '#define N_INPUTS {}\n'.format(layer_list[i-1]['n_in'])
                    newline += '#define N_LAYER_1 {}\n'.format(layer_list[i-1]['n_out'])
                elif i==len(layer_list) and layer_list[i-1]['class_name']=='Dense':
                    newline += '#define N_OUTPUTS {}\n'.format(layer_list[i-1]['n_out'])
                elif layer_list[i-1]['class_name']=='Dense':
                    newline += '#define N_LAYER_{} {}\n'.format(i, layer_list[i-1]['n_out'])    
                elif layer_list[i-1]['class_name']=='Conv1D':
                    newline += '#define Y_INPUTS_{} {}\n'.format(i, layer_list[i-1]['y_in'])
                    newline += '#define N_CHAN_{} {}\n'.format(i, layer_list[i-1]['n_chan'])
                    newline += '#define Y_OUTPUTS_{} {}\n'.format(i, layer_list[i-1]['y_out'])
                    newline += '#define N_FILT_{} {}\n'.format(i, layer_list[i-1]['n_filt'])
                elif layer_list[i-1]['class_name']=='Conv2D':
                    newline += '#define IN_HEIGHT_{} {}\n'.format(i, layer_list[i-1]['in_height'])
                    newline += '#define IN_WIDTH_{} {}\n'.format(i, layer_list[i-1]['in_width'])
                    newline += '#define N_CHAN_{} {}\n'.format(i, layer_list[i-1]['n_chan'])
                    newline += '#define OUT_HEIGHT_{} {}\n'.format(i, layer_list[i-1]['out_height'])
                    newline += '#define OUT_WIDTH_{} {}\n'.format(i, layer_list[i-1]['out_width'])
                    newline += '#define N_FILT_{} {}\n'.format(i, layer_list[i-1]['n_filt'])

                    
        elif '//hls-fpga-machine-learning insert layer-precision' in line:
            newline = line
            for i in range(1,len(layer_list)):
            #    if layer_list[i-1]['class_name']=='Dense':
            #        newline += 'typedef {precision} layer{index}_t;\n'.format(precision=yamlConfig["DefaultPrecision"], index=i)
                newline += 'typedef {precision} layer{index}_t;\n'.format(precision=yamlConfig["DefaultPrecision"][:-1]+", AP_TRN_ZERO, AP_SAT>", index=i)

        elif "//hls-fpga-machine-learning insert layer-config" in line:
            newline = line
            for i in range(1,len(layer_list)+1):
                if i==1 and layer_list[i-1]['class_name']=='Dense':
                    layer_in_name = "N_INPUTS"
                    layer_out_name = "N_LAYER_1"
                elif i==len(layer_list) and layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv1D':
                    layer_in_name = "Y_OUTPUTS_%i*N_FILT_%i" % (i-1, i-1)
                    layer_out_name = "N_OUTPUTS"
                elif i==len(layer_list) and layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv2D':
                    layer_in_name = "OUT_HEIGHT_%i*OUT_WIDTH_%i*N_FILT_%i" % (i-1, i-1, i-1)
                    layer_out_name = "N_OUTPUTS"
                elif layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv1D':
                    layer_in_name = "Y_OUTPUTS_%i*N_FILT_%i" % (i-1, i-1)
                    layer_out_name = "N_LAYER_%i" % (i)   
                elif layer_list[i-1]['class_name']=='Dense' and layer_list[i-2]['class_name']=='Conv2D':
                    layer_in_name = "OUT_HEIGHT_%i*OUT_WIDTH_%i*N_FILT_%i" % (i-1, i-1, i-1)
                    layer_out_name = "N_LAYER_%i" % (i)   
                elif i==len(layer_list) and layer_list[i-1]['class_name']=='Dense':
                    layer_in_name = "N_LAYER_%i" % (i-1)
                    layer_out_name = "N_OUTPUTS"               
                elif layer_list[i-1]['class_name']=='Dense':
                    layer_in_name = "N_LAYER_%i" % (i-1)
                    layer_out_name = "N_LAYER_%i" % (i)    
                elif layer_list[i-1]['class_name']=='Conv1D':
                    layer_y_in_name = "Y_INPUTS_%i" % (i)
                    layer_n_chan_name = "N_CHAN_%i" % (i)
                    layer_y_out_name = "Y_OUTPUTS_%i" % (i)
                    layer_n_filt_name = "N_FILT_%i" % (i)
                elif layer_list[i-1]['class_name']=='Conv2D':
                    layer_in_height_name = "IN_HEIGHT_%i" % (i)
                    layer_in_width_name = "IN_WIDTH_%i" % (i)
                    layer_n_chan_name = "N_CHAN_%i" % (i)
                    layer_out_height_name = "OUT_HEIGHT_%i" % (i)
                    layer_out_width_name = "OUT_WIDTH_%i" % (i)
                    layer_n_filt_name = "N_FILT_%i" % (i)
                if layer_list[i-1]['class_name']=='Dense':
                    newline += dense_config_template.format(index=str(i), 
                                                            n_in=layer_in_name, 
                                                            n_out=layer_out_name,
                                                            iotype=yamlConfig["IOType"],
                                                            reuse=yamlConfig["ReuseFactor"],
                                                            nzeros=layer_list[i-1]['weights_n_zeros'])

                    newline += activ_config_template.format(type=layer_list[i-1]['activation'],
                                                                    index=str(i), 
                                                                    n_in=layer_out_name,
                                                                    iotype=yamlConfig["IOType"]) 
                elif layer_list[i-1]['class_name']=='Conv1D':
                    newline += conv_config_template.format(index=str(i), 
                                                            pad_left=layer_list[i-1]['pad_left'], 
                                                            pad_right=layer_list[i-1]['pad_right'],
                                                            y_in=layer_y_in_name,
                                                            n_chan=layer_n_chan_name,
                                                            y_out=layer_y_out_name,
                                                            n_filt=layer_n_filt_name,
                                                            y_filt=layer_list[i-1]['y_filt'],
                                                            stride=layer_list[i-1]['stride'],
                                                            iotype=yamlConfig["IOType"],
                                                            reuse=yamlConfig["ReuseFactor"],
                                                            nzeros=layer_list[i-1]['weights_n_zeros'])

                    newline += activ_config_template.format(type=layer_list[i-1]['activation'],
                                                                    index=str(i), 
                                                                    n_in='%s*%s'%(layer_y_out_name,layer_n_filt_name),
                                                                    iotype=yamlConfig["IOType"]) 

                elif layer_list[i-1]['class_name']=='Conv2D':
                    newline += conv2d_config_template.format(index=str(i), 
                                                            pad_top=layer_list[i-1]['pad_top'], 
                                                            pad_bottom=layer_list[i-1]['pad_bottom'],
                                                            pad_left=layer_list[i-1]['pad_left'], 
                                                            pad_right=layer_list[i-1]['pad_right'],
                                                            in_height=layer_in_height_name,
                                                            in_width=layer_in_width_name,
                                                            n_chan=layer_n_chan_name,
                                                            out_height=layer_out_height_name,
                                                            out_width=layer_out_width_name,
                                                            n_filt=layer_n_filt_name,
                                                            filt_height=layer_list[i-1]['filt_height'],
                                                            filt_width=layer_list[i-1]['filt_width'],
                                                            stride_height=layer_list[i-1]['stride_height'],
                                                            stride_width=layer_list[i-1]['stride_width'],
                                                            iotype=yamlConfig["IOType"],
                                                            reuse=yamlConfig["ReuseFactor"],
                                                            nzeros=layer_list[i-1]['weights_n_zeros'])

                    newline += activ_config_template.format(type=layer_list[i-1]['activation'],
                                                                    index=str(i), 
                                                                    n_in='%s*%s*%s'%(layer_out_height_name,layer_out_width_name,layer_n_filt_name),
                                                                    iotype=yamlConfig["IOType"]) 

        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()


    ###################
    ## test bench
    ###################

    f = open(os.path.join(filedir,'../hls-template/myproject_test.cpp'),'r')
    fout = open('{}/{}_test.cpp'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

    for line in f.readlines():

        #Insert numbers
        if 'myproject' in line:
            newline = line.replace('myproject',yamlConfig['ProjectName'])
        elif '//hls-fpga-machine-learning insert data' in line and layer_list[0]['class_name']=='Dense':
            newline = line
            newline += '  input_t  data_str[N_INPUTS] = {'
            for i in range(0,layer_list[0]['n_in']-1):
                newline += '0,'
            newline += '0};\n'
        elif '//hls-fpga-machine-learning insert data' in line and layer_list[0]['class_name']=='Conv1D':
            newline = line
            newline += '  input_t  data_str[Y_INPUTS_1][N_CHAN_1] = {'
            for i in range(0,layer_list[0]['y_in']*layer_list[0]['n_chan']-1):
                newline += '0,'
            newline += '0};\n'
        elif '//hls-fpga-machine-learning insert data' in line and layer_list[0]['class_name']=='Conv2D':
            newline = line
            newline += '  input_t  data_str[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1] = {'
            for i in range(0,layer_list[0]['in_height']*layer_list[0]['in_width']*layer_list[0]['n_chan']-1):
                newline += '0,'
            newline += '0};\n'
        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()


    #######################
    ## myproject.h
    #######################

    f = open(os.path.join(filedir,'../hls-template/firmware/myproject.h'),'r')
    fout = open('{}/firmware/{}.h'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

    for line in f.readlines():

        if 'MYPROJECT' in line:
            newline = line.replace('MYPROJECT',format(yamlConfig['ProjectName'].upper()))
        elif 'void myproject(' in line:
            newline = 'void {}(\n'.format(yamlConfig['ProjectName'])
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='Conv1D':
            newline = line.replace('input_t data[N_INPUTS]','input_t data[Y_INPUTS_1][N_CHAN_1]')
        elif 'input_t data[N_INPUTS]' in line and layer_list[0]['class_name']=='Conv2D':
            newline = line.replace('input_t data[N_INPUTS]','input_t data[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1]')
        else:
            newline = line
        fout.write(newline)

    f.close()
    fout.close()


    #######################
    ## build_prj.tcl
    #######################

    nnetdir = os.path.abspath(os.path.join(filedir, "../nnet_utils"))
    relpath = os.path.relpath(nnetdir, start=yamlConfig['OutputDir'])

    f = open(os.path.join(filedir,'../hls-template/build_prj.tcl'),'r')
    fout = open('{}/build_prj.tcl'.format(yamlConfig['OutputDir']),'w')

    for line in f.readlines():

        line = line.replace('myproject',yamlConfig['ProjectName'])
        line = line.replace('nnet_utils', relpath)

        if 'set_part {xc7vx690tffg1927-2}' in line:
            line = 'set_part {{{}}}\n'.format(yamlConfig['XilinxPart'])
        elif 'create_clock -period 5 -name default' in line:
            line = 'create_clock -period {} -name default\n'.format(yamlConfig['ClockPeriod'])

        fout.write(line)
    f.close()
    fout.close()


    ###################
    # Tarball output
    ###################
    with tarfile.open(yamlConfig['OutputDir'] + '.tar.gz', mode='w:gz') as archive:
        archive.add(yamlConfig['OutputDir'], recursive=True)



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
    zero_ctr = 0
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

    #hls doesn't like 3d arrays... unrolling to 1d
    if len(a.shape)>=3: 
        f.write("[{}]".format(np.prod(a.shape)))
    else:
        for x in a.shape:
            f.write("[{}]".format(x))
    f.write(" = {")
    
    #fill c++ array.  
    #not including internal brackets for multidimensional case
    i=0
    for x in np.nditer(a, order='C'):
        if i==0:
            f.write("{}".format(x))
        else:
            f.write(", {}".format(x))
        i=i+1
    f.write("};\n")
    f.close()

    return zero_ctr

