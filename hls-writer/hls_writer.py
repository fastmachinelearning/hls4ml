import tarfile
import yaml
from shutil import copyfile

def hls_writer(layer_list, yamlConfig):
    
    ###################
    ## myproject.cpp
    ###################
    
    f = open('../hls-template/firmware/myproject.cpp','r')
    fout = open('{}/firmware/{}.cpp'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

    for line in f.readlines():
        #Add headers to weights and biases
        if 'myproject' in line:
            newline = line.replace('myproject',yamlConfig['ProjectName'])
        elif '//hls-fpga-machine-learning insert weights' in line:
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
                                                           nzeros=layer_list[i-1]['weights_n_zeros'])

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
    fout = open('{}/{}_test.cpp'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

    for line in f.readlines():

        #Insert numbers
        if 'myproject' in line:
            newline = line.replace('myproject',yamlConfig['ProjectName'])
        elif '//hls-fpga-machine-learning insert data' in line:
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
    ## myproject.h
    #######################
    f = open('../hls-template/firmware/myproject.h','r')
    fout = open('{}/firmware/{}.h'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

    for line in f.readlines():

        if 'MYPROJECT' in line:
            newline = line.replace('MYPROJECT',format(yamlConfig['ProjectName'].upper()))
        elif 'void myproject(' in line:
            newline = 'void {}(\n'.format(yamlConfig['ProjectName'])
        else:
            newline = line
        fout.write(newline)

    f.close()
    fout.close()


    #######################
    ## build_prj.tcl
    #######################
    f = open('../hls-template/build_prj.tcl','r')
    fout = open('{}/build_prj.tcl'.format(yamlConfig['OutputDir']),'w')

    for line in f.readlines():

        #Insert numbers
        if 'myproject' in line:
            newline = line.replace('myproject',yamlConfig['ProjectName'])
        elif 'set_part {xc7vx690tffg1927-2}' in line:
            newline = 'set_part {{{}}}\n'.format(yamlConfig['XilinxPart'])
        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()


    ###################
    # Tarball output
    ###################
    with tarfile.open(yamlConfig['OutputDir'] + '.tar.gz', mode='w:gz') as archive:
        archive.add(yamlConfig['OutputDir'], recursive=True)

