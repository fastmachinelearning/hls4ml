from __future__ import print_function
import tarfile
import yaml
from shutil import copyfile, copytree, rmtree
import numpy as np
import os
import re
import glob
from collections import OrderedDict

from hls4ml.writer.writers import Writer

class QuartusWriter(Writer):

    def next_pow2(self, x):
        return 1<<(x-1).bit_length()

    def get_max_reuse_factor(self, model):
        max_rf = 0
        for layer in model.get_layers():
            rf = int(layer.reuse_factor)
            if(rf > max_rf):
                max_rf = rf
        return max_rf

    def print_array_to_cpp(self, var, layer, odir):
        #######################################
        ## Print weight array to C++
        #######################################
        h_file = open("{}/firmware/weights/{}.h".format(odir,var.name),"w")

        #meta data
        h_file.write("//Numpy array shape {}\n".format(var.shape))
        h_file.write("//Min {:.12f}\n".format(np.min(var.min)))
        h_file.write("//Max {:.12f}\n".format(np.max(var.max)))
        h_file.write("//Number of zeros {}\n".format(var.nzeros))
        h_file.write("\n")

        h_file.write("#ifndef {}_H_\n".format(var.name.upper()))
        h_file.write("#define {}_H_\n".format(var.name.upper()))
        h_file.write("\n")

        rf = int(layer.reuse_factor)
        weight_header = '#ifdef __INTELFPGA_COMPILER__\n'
        if(rf == 1 or var.name[0] == 'b' or layer.get_attr('n_in')*layer.get_attr('n_out') <= 2048 or (var.name[0] == 'w' and (layer.binary_check() or layer.ternary_check()))):
            weight_header += 'hls_init_on_powerup\n'
        else:
            block_factor = (layer.get_attr('n_in')*layer.get_attr('n_out'))/rf
            nbanks = int((pow(2, np.ceil(np.log(block_factor)/np.log(2))))/2)
            var_width = int(np.ceil(int(re.findall('\d+',var.type.precision)[0])/8))
            bwidth = self.next_pow2(var_width)
            weight_header += 'hls_bankwidth({bwidth})\nhls_numbanks({nbanks})\nhls_max_replicates(1)\nhls_memory_impl("BLOCK_RAM")\n'.format(bwidth=bwidth, nbanks=nbanks)
        weight_header += '#endif\n'
        weight_header += 'static const '
        h_file.write(weight_header + var.definition_cpp() + " = {")

        #fill c++ array.
        #not including internal brackets for multidimensional case
        sep = ''
        for x in var:
            h_file.write(sep + x)
            sep = ", "
        h_file.write("};\n")
        h_file.write("\n#endif\n")
        h_file.close()

    def write_project_dir(self, model):
        if not os.path.isdir("{}/firmware/weights".format(model.config.get_output_dir())):
            os.makedirs("{}/firmware/weights".format(model.config.get_output_dir()))

    def write_project_cpp(self, model):
        ###################
        ## myproject.cpp
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/quartus/firmware/myproject.cpp'),'r')
        fout = open('{}/firmware/{}.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        indent = '    '

        for line in f.readlines():
            #Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '//hls-fpga-machine-learning insert cpragmas' in line:

                newline = line
                newline += 'hls_max_concurrency(0)\n'
                newline += 'hls_component_ii({})\n'.format(self.get_max_reuse_factor(model))
                clock_mhz = 1000/(model.config.get_config_value('ClockPeriod'))
                newline += 'hls_scheduler_target_fmax_mhz({})\n'.format(np.ceil(clock_mhz).astype(np.int))

            elif '//hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join(['inputdat ' + i.definition_cpp_name() for i in model_inputs])

                newline = ''
                newline += indent + inputs_str + '\n'

            elif '//hls-fpga-machine-learning insert weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += '#include "weights/{}.h"\n'.format(w.name)

            elif '//hls-fpga-machine-learning insert test weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += '#include "weights/{}_test.h"\n'.format(w.name)

            elif '//hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                inputs = model.get_input_variables()
                outputs = model.get_output_variables()
                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in inputs and var not in outputs:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                newline += '    ' + def_cpp + ' hls_register;\n'
                        if var in inputs:
                            var.name += '.data'
                        if var in outputs:
                            name = var.definition_cpp_name()
                            newline += '    ' + 'hls_register outputdat ' + name + ';\n'
                            var.name += '.data'
                    if layer.get_attr('activation') == 'tanh':
                        layer.set_attr('activation') == 'dense_tanh'
                    func = layer.function_cpp()
                    if func:
                        for line in func:
                            newline += '    ' + line + '\n'
                        newline += '\n'

                for inp in model.get_input_variables():
                    inp.name = inp.name.replace('.data','')
                for out in model.get_output_variables():
                    out.name = out.name.replace('.data','')
                    name = out.definition_cpp_name()
                    newline += indent + 'return ' + name + ';\n'
            #Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_project_header(self, model):
        #######################
        ## myproject.h
        #######################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/quartus/firmware/myproject.h'),'r')
        fout = open('{}/firmware/{}.h'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        indent = '    '

        for line in f.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT',format(model.config.get_project_name().upper()))
            elif '//hls-fpga-machine-learning insert cpragmas' in line:
                newline = line
                newline += 'hls_max_concurrency(0)\n'
                newline += 'hls_component_ii({})\n'.format(self.get_max_reuse_factor(model))
                clock_mhz = 1000/(model.config.get_config_value('ClockPeriod'))
                newline += 'hls_scheduler_target_fmax_mhz({})\n'.format(np.ceil(clock_mhz).astype(np.int))
            elif 'component outputdat myproject(' in line:
                newline = 'component outputdat {}(\n'.format(model.config.get_project_name())
            elif '//Input Parameters' in line:
                for input in model.get_input_variables():
                    newline = ''
                    newline += indent + input.definition_cpp_type() + ' data' + '[' + input.size_cpp() + ']' + ';\n'
            elif '//Output Parameters' in line:
                for out in model.get_output_variables():
                    newline = ''
                    newline += indent + out.definition_cpp_type() + ' data' + '[' + out.size_cpp() + ']' + ';\n'
            elif '//hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join(['inputdat ' + i.definition_cpp_name() for i in model_inputs])

                newline = ''
                newline += indent + inputs_str + '\n'
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_defines(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/quartus/firmware/defines.h'),'r')
        fout = open('{}/firmware/defines.h'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():

            #Insert numbers
            if '//hls-fpga-machine-learning insert numbers' in line:
                newline = line
                numbers = OrderedDict.fromkeys([layer.get_numbers_cpp() for layer in model.get_layers()])
                newline += ''.join(numbers)

            elif '//hls-fpga-machine-learning insert layer-precision' in line:
                newline = line
                all_precision = OrderedDict()
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    all_precision.update(layer_precision)
                    dummy_layer = layer
                for used_type in all_precision.values():
                    newline += dummy_layer.var_definition_cpp(used_type)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_parameters(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/quartus/firmware/parameters.h'),'r')
        fout = open('{}/firmware/parameters.h'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():

            if '//hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(set(sum((layer.include_list for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include

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

    def write_weights(self, model):
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(weights, layer, model.config.get_output_dir())


    def write_test_bench(self, model):
        ###################
        ## test bench
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists('{}/tb_data/'.format(model.config.get_output_dir())):
            os.mkdir('{}/tb_data/'.format(model.config.get_output_dir()))

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == "dat":
                copyfile(input_data, '{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(input_data,'{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))

        if output_predictions:
            if output_predictions[-3:] == "dat":
                copyfile(output_predictions, '{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(output_predictions,'{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))

        f = open(os.path.join(filedir,'../templates/quartus/myproject_test.cpp'),'r')
        fout = open('{}/{}_test.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            #Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert data' in line:
                newline = line
                newline += '      std::vector<float>::const_iterator in_begin = in.cbegin();\n'
                newline += '      std::vector<float>::const_iterator in_end;\n'
                for inp in model.get_input_variables():
                    newline += '      in_end = in_begin + ({});\n'.format(inp.size_cpp())
                    newline += '      std::copy(in_begin, in_end, {});\n'.format(inp.cppname+'[e].data')
                    newline += '      in_begin = in_end;\n'
            elif '//hls-fpga-machine-learning insert component-io' in line:
                newline = line
                for inp in model.get_input_variables():
                    newline += indent + 'inputdat ' + inp.definition_cpp_name() + '[num_iterations];\n'
                for out in model.get_output_variables():
                    # brace-init zeros the array out because we use std=c++0x
                    newline += indent + 'outputdat ' + out.definition_cpp_name() + '[num_iterations];\n'
            elif '//hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inp in model.get_input_variables():
                    newline += '    ' + 'inputdat ' + inp.definition_cpp_name() + '[num_iterations];\n'
                for out in model.get_output_variables():
                    newline += '    ' + 'outputdat ' + out.definition_cpp_name() + '[num_iterations];\n'
            elif '//hls-fpga-machine-learning insert top-level-function' in line:
                newline = line

                input_vars = ','.join([i.cppname for i in model.get_input_variables()])
                output_vars = ','.join([o.cppname for o in model.get_output_variables()])

                top_level = indent + 'ihc_hls_enqueue(&{}, {}, {});\n'.format(output_vars+'[e]', model.config.get_project_name(), input_vars+'[e]')
                newline += top_level
                newline += '    ' + '}\n'
                newline += '    ' + 'ihc_hls_component_run_all({});\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert second-top-level-function' in line:
                newline = line

                input_vars = ','.join([i.cppname for i in model.get_input_variables()])
                output_vars = ','.join([o.cppname for o in model.get_output_variables()])

                newline += indent + 'std::fill_n({}, {}, 0.);\n'.format(inp.cppname+'[i].data', inp.size_cpp())

                top_level = indent + 'ihc_hls_enqueue(&{}, {}, {});\n'.format(output_vars+'[i]', model.config.get_project_name(), input_vars+'[i]')
                newline += top_level
                newline += '    ' + '}\n'
                newline += '    ' + 'ihc_hls_component_run_all({});\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for out in model.get_output_variables():
                    newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(out.size_cpp())
                    newline += indent + '  std::cout << pr[j][i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'
            elif '//hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                for out in model.get_output_variables():
                    newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(out.size_cpp())
                    newline += indent + '  fout << {}[j].data[i] << " ";\n'.format(out.cppname)
                    newline += indent + '}\n'
                    newline += indent + 'fout << std::endl;\n'
            elif '//hls-fpga-machine-learning insert output' in line or '//hls-fpga-machine-learning insert quantized' in line:
                newline = line
                for out in model.get_output_variables():
                    newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(out.size_cpp())
                    newline += indent + '  std::cout << {}[j].data[i] << " ";\n'.format(out.cppname)
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_bridge(self, model):
        ###################
        # c++-python bridge
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/vivado/myproject_bridge.cpp'),'r')
        fout = open('{}/{}_bridge.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        indent = '    '

        for line in f.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))
            elif '//hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join(['{type} {name}[{shape}]'.format(type=dtype, name=i.cppname, shape=i.size_cpp()) for i in model_inputs])
                outputs_str = ', '.join(['{type} {name}[{shape}]'.format(type=dtype, name=o.cppname, shape=o.size_cpp()) for o in model_outputs])
                insize_str = ', '.join(['unsigned short &const_size_in_{}'.format(i) for i in range(1, len(model_inputs) + 1)])
                outsize_str = ', '.join(['unsigned short &const_size_out_{}'.format(o) for o in range(1, len(model_outputs) + 1)])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + ',\n'
                newline += indent + insize_str + ',\n'
                newline += indent + outsize_str + '\n'

            elif '//hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for i in model_inputs:
                    newline += indent + 'inputdat {name}_ap;\n'.format(name=i.cppname)
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap.data);\n'.format(dtype, i.type.name, i.size_cpp(), i.cppname, i.cppname)
                newline += '\n'

                for o in model_outputs:
                    newline += indent + 'outputdat {name}_ap;\n'.format(name=o.cppname)

                input_vars = ','.join([i.cppname + '_ap' for i in model.get_input_variables()])
                output_vars = ','.join([o.cppname + '_ap' for o in model.get_output_variables()])
                top_level = indent + '{} = {}({});\n'.format(output_vars, model.config.get_project_name(), input_vars)
                newline += top_level

                newline += '\n'

                for o in model_outputs:
                    newline += indent + 'nnet::convert_data_back<{}, {}, {}>({}_ap.data, {});\n'.format(o.type.name, dtype, o.size_cpp(), o.cppname, o.cppname)
            elif '//hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    if layer.function_cpp() and model.config.trace_output and model.config.get_layer_config_value(layer, 'Trace', False):
                            vars = layer.get_variables()
                            for var in vars:
                                newline += indent + 'nnet::trace_outputs->insert(std::pair<std::string, void *>("{}", (void *) malloc({} * element_size)));\n'.format(layer.name, var.size_cpp())

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_build_script(self, model):
        ###################
        # Makefile
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/quartus/Makefile'),'r')
        fout = open('{}/Makefile'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():

            line = line.replace('myproject',model.config.get_project_name())

            if 'DEVICE :=' in line:
                line = 'DEVICE := {}\n'.format(model.config.get_config_value('FPGAPart'))

            fout.write(line)
        f.close()
        fout.close()

        ###################
        # build_lib.sh
        ###################

        f = open(os.path.join(filedir,'../templates/quartus/build_lib.sh'),'r')
        fout = open('{}/build_lib.sh'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())

            fout.write(line)
        f.close()
        fout.close()

    def write_nnet_utils(self, model):
        ###################
        ## nnet_utils
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir,'../templates/quartus/firmware/nnet_utils/')
        dstpath = '{}/firmware/nnet_utils/'.format(model.config.get_output_dir())

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        ###################
        ## ac_types
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir,'../templates/quartus/ac_types/')
        dstpath = '{}/firmware/ac_types/'.format(model.config.get_output_dir())

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

    def write_activation_tables(self, model):

        ###################
        ## activation_tables
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        #srcpath = os.path.join(filedir,'../templates/quartus/firmware/nnet_utils/activation_tables/')
        dstpath = '{}/firmware/nnet_utils/activation_tables/'.format(model.config.get_output_dir())

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        ###################
        ## elu_table
        ###################
        for layer in model.get_layers():
            if(layer.get_attr('activation') == 'elu'):
                table_size = layer.get_attr('table_size')
            else:
                table_size = 1024

        table_name = 'elu_table'
        h_file = open("{}/{}.tb".format(dstpath, table_name),"w")

        #meta data
        h_file.write("#ifndef {}_H_\n".format(table_name.upper()))
        h_file.write("#define {}_H_\n".format(table_name.upper()))
        h_file.write("\n")

        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += 'static const typename CONFIG_T::table_t {}[{}] = {{'.format(table_name, table_size)

        h_file.write(table_header)

        sep = ''
        for i in range(table_size):
            in_val = -8.0*i/float(table_size);
            real_val = np.exp(in_val) - 1.;
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write("};\n")
        h_file.write("\n#endif\n")
        h_file.close()

        ###################
        ## sigmoid_table
        ###################
        for layer in model.get_layers():
            if(layer.get_attr('activation') == 'sigmoid'):
                table_size = layer.get_attr('table_size')
            else:
                table_size = 1024

        table_name = 'sigmoid_table'
        h_file = open("{}/{}.tb".format(dstpath, table_name),"w")

        #meta data
        h_file.write("#ifndef {}_H_\n".format(table_name.upper()))
        h_file.write("#define {}_H_\n".format(table_name.upper()))
        h_file.write("\n")

        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += 'static const typename CONFIG_T::table_t {}[{}] = {{'.format(table_name, table_size)

        h_file.write(table_header)

        sep = ''
        for i in range(table_size):
            in_val = 2*8.0*(i-float(table_size)/2.0)/float(table_size)
            real_val = 1.0 / (1 + np.exp(-in_val))
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write("};\n")
        h_file.write("\n#endif\n")
        h_file.close()

        ###################
        ## tanh_table
        ###################
        for layer in model.get_layers():
            if(layer.get_attr('activation') == 'dense_tanh'):
                table_size = layer.get_attr('table_size')
            else:
                table_size = 1024

        table_name = 'tanh_table'
        h_file = open("{}/{}.tb".format(dstpath, table_name),"w")

        #meta data
        h_file.write("#ifndef {}_H_\n".format(table_name.upper()))
        h_file.write("#define {}_H_\n".format(table_name.upper()))
        h_file.write("\n")

        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += 'static const typename CONFIG_T::table_t {}[{}] = {{'.format(table_name, table_size)

        h_file.write(table_header)

        sep = ''
        for i in range(table_size):
            in_val = 2*4.0*(i-float(table_size)/2.0)/float(table_size)
            real_val = np.tanh(in_val)
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write("};\n")
        h_file.write("\n#endif\n")
        h_file.close()

        ###################
        ## softplus_table
        ###################
        for layer in model.get_layers():
            if(layer.get_attr('activation') == 'softplus'):
                table_size = layer.get_attr('table_size')
            else:
                table_size = 1024

        table_name = 'softplus_table'
        h_file = open("{}/{}.tb".format(dstpath, table_name),"w")

        #meta data
        h_file.write("#ifndef {}_H_\n".format(table_name.upper()))
        h_file.write("#define {}_H_\n".format(table_name.upper()))
        h_file.write("\n")

        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += 'static const typename CONFIG_T::table_t {}[{}] = {{'.format(table_name, table_size)

        h_file.write(table_header)

        sep = ''
        for i in range(table_size):
            in_val = 2*8.0*(i-float(table_size)/2.0)/float(table_size)
            real_val = np.log(np.exp(in_val) + 1.)
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write("};\n")
        h_file.write("\n#endif\n")
        h_file.close()

        ###################
        ## softsign_table
        ###################
        for layer in model.get_layers():
            if(layer.get_attr('activation') == 'softsign'):
                table_size = layer.get_attr('table_size')
            else:
                table_size = 1024

        table_name = 'softsign_table'
        h_file = open("{}/{}.tb".format(dstpath, table_name),"w")

        #meta data
        h_file.write("#ifndef {}_H_\n".format(table_name.upper()))
        h_file.write("#define {}_H_\n".format(table_name.upper()))
        h_file.write("\n")

        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += 'static const typename CONFIG_T::table_t {}[{}] = {{'.format(table_name, table_size)

        h_file.write(table_header)

        sep = ''
        for i in range(table_size):
            in_val = 2*8.0*(i-float(table_size)/2.0)/float(table_size);
            real_val = in_val / (np.fabs(in_val) + 1.);
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write("};\n")
        h_file.write("\n#endif\n")
        h_file.close()

        ###################
        ## selu_table
        ###################
        for layer in model.get_layers():
            if(layer.get_attr('activation') == 'selu'):
                table_size = layer.get_attr('table_size')
            else:
                table_size = 1024

        table_name = 'selu_table'
        h_file = open("{}/{}.tb".format(dstpath, table_name),"w")

        #meta data
        h_file.write("#ifndef {}_H_\n".format(table_name.upper()))
        h_file.write("#define {}_H_\n".format(table_name.upper()))
        h_file.write("\n")

        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += 'static const typename CONFIG_T::table_t {}[{}] = {{'.format(table_name, table_size)

        h_file.write(table_header)

        sep = ''
        for i in range(table_size):
            in_val = -8.0*i/float(table_size);
            real_val = 1.0507009873554804934193349852946 * (1.6732632423543772848170429916717 * (np.exp(in_val) - 1.));
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write("};\n")
        h_file.write("\n#endif\n")
        h_file.close()

        ###################
        ## exp_table
        ###################
        for layer in model.get_layers():
            if(layer.get_attr('activation') == 'softmax'):
                table_size = layer.get_attr('table_size')
            else:
                table_size = 1024

        table_name = 'exp_table'
        h_file = open("{}/{}.tb".format(dstpath, table_name),"w")

        #meta data
        h_file.write("#ifndef {}_H_\n".format(table_name.upper()))
        h_file.write("#define {}_H_\n".format(table_name.upper()))
        h_file.write("\n")

        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += 'static const typename CONFIG_T::table_t {}[{}] = {{'.format(table_name, table_size)

        h_file.write(table_header)

        sep = ''
        for i in range(table_size):
            in_val = 2*8.0*(i-float(table_size)/2.0)/float(table_size);
            real_val = np.exp(in_val);
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write("};\n")
        h_file.write("\n#endif\n")
        h_file.close()

        ###################
        ## invert_table
        ###################
        for layer in model.get_layers():
            if(layer.get_attr('activation') == 'softmax'):
                table_size = layer.get_attr('table_size')
            else:
                table_size = 1024

        table_name = 'invert_table'
        h_file = open("{}/{}.tb".format(dstpath, table_name),"w")

        #meta data
        h_file.write("#ifndef {}_H_\n".format(table_name.upper()))
        h_file.write("#define {}_H_\n".format(table_name.upper()))
        h_file.write("\n")

        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += 'static const typename CONFIG_T::table_t {}[{}] = {{'.format(table_name, table_size)

        h_file.write(table_header)

        sep = ''
        for i in range(table_size):
            real_val = 0
            in_val = 64.0*i/float(table_size);
            if (in_val > 0.0):
                real_val = 1.0/in_val;
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write("};\n")
        h_file.write("\n#endif\n")
        h_file.close()


    def write_tar(self, model):
        ###################
        # Tarball output
        ###################

        with tarfile.open(model.config.get_output_dir() + '.tar.gz', mode='w:gz') as archive:
            archive.add(model.config.get_output_dir(), recursive=True)

    def write_hls(self, model):
        print('Writing HLS project')
        self.write_project_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        self.write_weights(model)
        self.write_defines(model)
        self.write_parameters(model)
        self.write_test_bench(model)
        self.write_bridge(model)
        self.write_build_script(model)
        self.write_nnet_utils(model)
        self.write_activation_tables(model)
        self.write_tar(model)
        print('Done')
