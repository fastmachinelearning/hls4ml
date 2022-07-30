from __future__ import print_function
import tarfile
import yaml
from shutil import copyfile, copytree, rmtree
import numpy as np
import os
import glob
from collections import OrderedDict

from hls4ml.writer.writers import Writer
from hls4ml.backends import get_backend
from hls4ml.utils.fixed_point_utils import FixedPointEmulator, ceil_log2, uint_to_binary

config_filename = 'hls4ml_config.yml'

class QuartusWriter(Writer):

    def next_pow2(self, x):
        return 1 << (x - 1).bit_length()

    def get_max_reuse_factor(self, model):
        max_rf = 0
        for layer in model.get_layers():
            rf = int(layer.get_attr('reuse_factor'))
            if (rf > max_rf):
                max_rf = rf
        return max_rf

    def print_array_to_cpp(self, var, layer, odir):
        #######################################
        ## Print weight array to C++
        #######################################
        h_file = open("{}/firmware/weights/{}.h".format(odir, var.name), "w")

        # meta data
        h_file.write("//Numpy array shape {}\n".format(var.shape))
        h_file.write("//Min {:.12f}\n".format(np.min(var.min)))
        h_file.write("//Max {:.12f}\n".format(np.max(var.max)))
        h_file.write("//Number of zeros {}\n".format(var.nzeros))
        h_file.write("\n")

        h_file.write("#ifndef {}_H_\n".format(var.name.upper()))
        h_file.write("#define {}_H_\n".format(var.name.upper()))
        h_file.write("\n")

        rf = int(layer.get_attr('reuse_factor'))
        weight_header = '#ifdef __INTELFPGA_COMPILER__\n'
        if (rf == 1 or var.name[0] == 'b' or layer.get_attr('n_in') * layer.get_attr('n_out') <= 2048
                or (var.name[0] == 'w' and var.type.precision.width < 3)):
            weight_header += 'hls_init_on_powerup\n'
        else:
            block_factor = (layer.get_attr('n_in') * layer.get_attr('n_out')) / rf
            nbanks = int(2 ** np.ceil(np.log2(block_factor)) / 2)
            var_width = int(np.ceil(var.type.precision.width / 8))
            bwidth = self.next_pow2(var_width)
            weight_header += 'hls_bankwidth({bwidth})\nhls_numbanks({nbanks})\nhls_max_replicates(1)\nhls_memory_impl("BLOCK_RAM")\n'.format(
                bwidth=bwidth, nbanks=nbanks)
        weight_header += '#endif\n'
        weight_header += 'static const '
        h_file.write(weight_header + var.definition_cpp() + " = {")

        # fill c++ array.
        # not including internal brackets for multidimensional case
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
        f = open(os.path.join(filedir, '../templates/quartus/firmware/myproject.cpp'), 'r')
        fout = open('{}/firmware/{}.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        io_type = model.config.get_config_value('IOType')
        indent = '   '

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            
            # Intel HLS 'streams' need to be passed by reference to top-level entity or declared as global variables
            # Streams cannot be declared inside a function
            # Therefore, layer connections (inputs/outputs) are declared here
            elif '//hls-fpga-machine-learning insert inter-task streams' in line:
                newline = line
                if io_type == 'io_stream':
                    for layer in model.get_layers():
                        vars = layer.get_variables()
                        for var in vars:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                newline += def_cpp + ';\n'

            # Instantiate GCC top-level function, to be used during GCC compilation / hls4ml.predict()
            elif '//hls-fpga-machine-learning instantiate GCC top-level' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += 'void myproject(\n'
                    newline += indent+'stream_in<{}> &input_stream,\n'.format(model_inputs[0].type.name)
                    newline += indent+'stream_out<{}> &output_stream\n'.format(model_outputs[0].type.name)
                    newline += ') {\n'
                if io_type == 'io_parallel':
                    newline = 'output_data myproject(\n'
                    newline+=indent+'input_data inputs\n'
                    newline+=') {\n'

            # Instantiate HLS top-level function, to be used during HLS synthesis
            elif '//hls-fpga-machine-learning instantiate HLS top-level' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += 'component void myproject(\n'
                    newline += indent+'stream_in<{}> &input_stream,\n'.format(model_inputs[0].type.name)
                    newline += indent+'stream_out<{}> &output_stream\n'.format(model_outputs[0].type.name)
                    newline += ') {\n'
                if io_type == 'io_parallel':
                    newline += 'component output_data myproject(\n'
                    newline += indent+'input_data inputs\n'
                    newline += ') {\n'
        
            # Insert HLS pragmas such as maximum frequency, initiation interval etc.
            elif '//hls-fpga-machine-learning insert cpragmas' in line:
                newline = line
                newline += 'hls_max_concurrency(0)\n'
                newline += 'hls_component_ii({})\n'.format(self.get_max_reuse_factor(model))
                clock_mhz = 1000 / (model.config.get_config_value('ClockPeriod'))
                newline += 'hls_scheduler_target_fmax_mhz({})\n'.format(np.ceil(clock_mhz).astype(np.int))

            # In io_parallel, an output (struct) is returned from the top-level function
            # Therefore, it needs to be initialised before returning
            # In io_stream, the input is of type 'stream_in' and output is of type 'stream_out'
            # However, individual layers accept the type 'stream' 
            # Therefore, data is first read from 'stream_in', written to 'stream' and propagated through network 
            elif '//hls-fpga-machine-learning initialize input/output' in line:
                if io_type == 'io_stream':
                    newline = line
                    newline += indent + f'for (size_t i = 0; i < {model_inputs[0].size_cpp()} / {model_inputs[0].type.name}::size; i++) {{\n'
                    newline += indent + f'  {model_inputs[0].type.name} tmp = input_stream.read();\n'
                    newline += indent + f'  {model_inputs[0].name}.write(tmp);\n'
                    newline += indent + f'}}\n'
                else:
                    newline = line
                    newline += indent+'hls_register output_data outputs;\n'
            
            # Insert weights
            elif '//hls-fpga-machine-learning insert weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += '#include "weights/{}.h"\n'.format(w.name)
            
            # Insert test weights
            elif '//hls-fpga-machine-learning insert test weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += '#include "weights/{}_test.h"\n'.format(w.name)

            # Neural net instantiation
            elif '//hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                for layer in model.get_layers():
                    if io_type != 'io_stream':
                        vars = layer.get_variables()
                        for var in vars:
                            if var not in model_inputs and var not in model_outputs:
                                def_cpp = var.definition_cpp()
                                if def_cpp is not None:
                                    newline += '    ' + def_cpp + ';\n'
                    func = layer.get_attr('function_cpp', None)
                    if func:
                        newline += '    ' + func + '\n'
                        if model.config.trace_output and layer.get_attr('Trace', False):
                            newline += '#ifndef HLS_SYNTHESIS\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(var.type.name, var.name, layer.name, var.size_cpp())
                            newline += '#endif\n'
                        newline += '\n'
            
            # In io_parallel, a return is required; for more details see myproject.cpp & myproject.h
            elif '//hls-fpga-machine-learning return' in line:
                if io_type == 'io_stream':
                    newline = line
                    newline += indent + f'for (size_t i = 0; i < {model_outputs[0].size_cpp()} / {model_outputs[0].type.name}::size; i++) {{\n'
                    newline += indent + f'  {model_outputs[0].type.name} tmp = {model_outputs[0].name}.read();\n'
                    newline += indent + f'  output_stream.write(tmp);\n'
                    newline += indent + f'}}\n'
                    newline += '}\n'
                else:
                    newline = line
                    newline += indent+'return outputs;\n'
                    newline += '}\n'

            # Just copy line
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
        f = open(os.path.join(filedir, '../templates/quartus/firmware/myproject.h'), 'r')
        fout = open('{}/firmware/{}.h'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        # io_parallel and io_stream instantiate the top-level function differently
        io_type = model.config.get_config_value('IOType')
        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            
            elif 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            
            elif '//hls-fpga-machine-learning instantiate GCC top-level' in line:
                newline = line
                # For io_stream, input and output are passed by reference; see myproject.h & myproject.cpp for more details
                if io_type == 'io_stream':
                    newline += 'void myproject(\n'
                    newline += indent+'stream_in<{}> &input_stream,\n'.format(model_inputs[0].type.name)
                    newline += indent+'stream_out<{}> &output_stream\n'.format(model_outputs[0].type.name)
                    newline += ');\n'
                # In io_parallel, a struct is returned; see myproject.h & myproject.cpp for more details
                else:
                    newline += 'output_data myproject(\n'
                    newline += indent+'input_data inputs\n'
                    newline += ');\n'

            # Similar to GCC instantiation, but with the keyword 'component'
            elif '//hls-fpga-machine-learning instantiate HLS top-level' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += 'component void myproject(\n'
                    newline += indent+'stream_in<{}> &input_stream,\n'.format(model_inputs[0].type.name)
                    newline += indent+'stream_out<{}> &output_stream\n'.format(model_outputs[0].type.name)
                    newline += ');\n'
                else:
                    newline += 'component output_data myproject(\n'
                    newline += indent+'input_data inputs\n'
                    newline += ');\n'
        
            elif '//hls-fpga-machine-learning insert cpragmas' in line:
                newline = line
                newline += 'hls_max_concurrency(0)\n'
                newline += 'hls_component_ii({})\n'.format(self.get_max_reuse_factor(model))
                clock_mhz = 1000 / (model.config.get_config_value('ClockPeriod'))
                newline += 'hls_scheduler_target_fmax_mhz({})\n'.format(np.ceil(clock_mhz).astype(np.int))
            
            # For io_stream, no inputs/outputs are instantiated, as they are passed by reference
            # For io_parallel, input/output structs are required 
            elif '//hls-fpga-machine-learning insert inputs' in line:
                newline = line
                if io_type!='io_stream':
                    newline += 'struct input_data { \n'
                    for inp in model_inputs:
                        newline += indent + inp.definition_cpp() + ';\n'
                    newline+='};\n'
            elif '//hls-fpga-machine-learning insert outputs' in line:
                newline = line
                if io_type!='io_stream':
                    newline += 'struct output_data { \n'
                    for out in model_outputs:
                        newline += indent + out.definition_cpp() + ';\n'
                    newline += '};\n'
            # Simply copy line, if no inserts are required
            else:
                newline = line
            
            fout.write(newline)

        f.close()
        fout.close()

    def write_defines(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/quartus/firmware/defines.h'), 'r')
        fout = open('{}/firmware/defines.h'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():

            # Insert numbers
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
                for used_type in all_precision.values():
                    newline += used_type.definition_cpp()
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_parameters(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/quartus/firmware/parameters.h'), 'r')
        fout = open('{}/firmware/parameters.h'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():

            if '//hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(
                        set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include

            elif "//hls-fpga-machine-learning insert layer-config" in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.get_attr('config_cpp', None)
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

    def write_testbench_parallel(self, model):
        if len(model.get_output_variables()) != 1:
            print("WARNING:  The testbench only supports one output variable. Leaving empty testbench")
            return

        outvar = model.get_output_variables()[0]
        invar = model.get_input_variables()[0]

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists('{}/tb_data/'.format(model.config.get_output_dir())):
            os.mkdir('{}/tb_data/'.format(model.config.get_output_dir()))

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == "dat":
                copyfile(input_data, '{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(input_data,
                                     '{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))

        if output_predictions:
            if output_predictions[-3:] == "dat":
                copyfile(output_predictions,
                         '{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(output_predictions,
                                     '{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))
        
        f = open(os.path.join(filedir, '../templates/quartus/myproject_test_parallel.cpp'), 'r') 
        fout = open('{}/{}_test.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')
        
        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert data' in line:
                newline = line
                newline += '      std::vector<float>::const_iterator in_begin = in.cbegin();\n'
                newline += '      std::vector<float>::const_iterator in_end;\n'
                newline += '      inputs.emplace_back();\n'
                for inp in model.get_input_variables():
                    newline += f'      in_end = in_begin + ({inp.size_cpp()});\n'
                    newline += f'      std::copy(in_begin, in_end, inputs.back().{inp.member_name});\n'
                    newline += '      in_begin = in_end;\n'
                newline += '      outputs.emplace_back();\n'
            elif '//hls-fpga-machine-learning insert zero' in line:
                newline = line
                newline += indent + 'for(int i = 0; i < num_iterations; i++) {\n'
                for inp in model.get_input_variables():
                    newline += indent + f'  inputs.emplace_back();\n'
                    newline += indent + f'  outputs.emplace_back();\n'
                    newline += indent + f'  std::fill_n(inputs[i].{inp.member_name}, {inp.size_cpp()}, 0.0);\n'
                newline += indent + '}\n'

            elif '//hls-fpga-machine-learning insert top-level-function' in line:
                newline = line

                newline += indent + 'for(int i = 0; i < num_iterations; i++) {\n'
                newline += indent + f'  ihc_hls_enqueue(&outputs[i], {model.config.get_project_name()}, inputs[i]);\n'
                newline += indent + '}\n'
            elif 'hls-fpga-machine-learning insert run' in line:
                newline = line
                newline += '    ' + 'ihc_hls_component_run_all({});\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert predictions' in line:
                newline = line
                newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(outvar.size_cpp())
                newline += indent + '  std::cout << predictions[j][i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'std::cout << std::endl;\n'
            elif '//hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(outvar.size_cpp())
                newline += indent + '  fout << outputs[j].{}[i] << " ";\n'.format(outvar.member_name)
                newline += indent + '}\n'
                newline += indent + 'fout << std::endl;\n'
            elif '//hls-fpga-machine-learning insert output' in line or '//hls-fpga-machine-learning insert quantized' in line:
                newline = line
                newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(outvar.size_cpp())
                newline += indent + '  std::cout << outputs[j].{}[i] << " ";\n'.format(outvar.member_name)
                newline += indent + '}\n'
                newline += indent + 'std::cout << std::endl;\n'
            else:
                newline = line
            
            fout.write(newline)
        
        f.close()
        fout.close()

    def write_testbench_stream(self, model):
        if len(model.get_output_variables()) != 1:
            print("WARNING:  The testbench only supports one output variable. Leaving empty testbench")
            return

        outvar = model.get_output_variables()[0]
        invar = model.get_input_variables()[0]

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists('{}/tb_data/'.format(model.config.get_output_dir())):
            os.mkdir('{}/tb_data/'.format(model.config.get_output_dir()))

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == "dat":
                copyfile(input_data, '{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(input_data,
                                     '{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))

        if output_predictions:
            if output_predictions[-3:] == "dat":
                copyfile(output_predictions,
                         '{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(output_predictions,
                                     '{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))
        
        f = open(os.path.join(filedir, '../templates/quartus/myproject_test_stream.cpp'), 'r')
        fout = open('{}/{}_test.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        if len(model.get_input_variables()) > 1 or len(model.get_output_variables()) > 1:
                raise Exception('Quartus io_stream supports exactly one input/output per model')
            
        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            
            elif '//hls-fpga-machine learning instantiate inputs and outputs' in line:
                newline = line
                newline += indent + 'stream_in<{}> inputs;\n'.format(invar.type.name)
                newline += indent + 'stream_out<{}> outputs;\n'.format(outvar.type.name)

            # TODO - This is one-input specific (are multiple model inputs needed at all?)
            elif '//hls-fpga-machine-learning insert data' in line:
                newline = line
                newline += indent + f'float vals[{invar.size_cpp()}]; \n'
                newline += indent + f'for (int j = 0 ; j < {invar.size_cpp()} ; j++) {{\n'
                newline += indent + f'  vals[j] = in[j]; \n'
                newline += indent + f'}}'
                newline += indent + f'nnet::convert_data<float, {invar.type.name}, {invar.size_cpp()}>(vals, inputs);\n'
            
            elif '//hls-fpga-machine-learning insert zero' in line:
                newline = line
                newline += indent + f'float vals[{invar.size_cpp()}]; \n'
                newline += indent + f'for (int j = 0 ; j < {invar.size_cpp()} ; j++) {{'
                newline += indent + f'  vals[j] = 0.0; \n'
                newline += indent + f'}}'
                newline += indent + f'nnet::convert_data<float, {invar.type.name}, {invar.size_cpp()}>(vals, inputs);\n'

            elif '//hls-fpga-machine-learning insert top-level-function' in line:
                newline = line
                newline += indent + f'ihc_hls_enqueue_noret(&{model.config.get_project_name()}, inputs, outputs); \n'
            
            elif 'hls-fpga-machine-learning insert run' in line:
                newline = line
                newline += indent + 'ihc_hls_component_run_all({});\n'.format(model.config.get_project_name())
            
            elif '//hls-fpga-machine-learning convert output' in line:
                newline = line
                newline += indent + 'float res[{}];\n'.format(outvar.size_cpp())
                newline += indent + 'nnet::convert_data_back<{}, float, {}>(outputs, res);\n'.format(outvar.type.name,
                                                                                                    outvar.size_cpp())

            elif '//hls-fpga-machine-learning insert tb-output' in line:
                newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(outvar.size_cpp())
                newline += indent + '  fout << res[i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'fout << std::endl;\n'

            elif '//hls-fpga-machine-learning print predictions' in line:
                newline = line
                newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(outvar.size_cpp())
                newline += indent + '  std::cout << predictions[iteration][i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'std::cout << std::endl;\n'
            
            elif '//hls-fpga-machine-learning print output' in line:
                newline = line
                newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(outvar.size_cpp())
                newline += indent + '  std::cout << res[i] << " "; \n'
                newline += indent + '} \n'
                newline += indent + 'std::cout << std::endl; \n'
            else:
                newline = line
            
            fout.write(newline)

        f.close()
        fout.close()

    def write_test_bench(self, model):
        ###################
        ## Test Bench
        ###################
        # TODO - This function only works with one model input (NOT one data point - it works as expected with multiple data points)
        io_type = model.config.get_config_value('IOType')
        if io_type == 'io_parallel':
            self.write_testbench_parallel(model)
        elif io_type == 'io_stream':
             self.write_testbench_stream(model)
        
    def write_bridge(self, model):
        ###################
        # C++-python bridge
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/quartus/myproject_bridge.cpp'), 'r')
        fout = open('{}/{}_bridge.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        io_type = model.config.get_config_value('IOType')
        indent = '    '

        for line in f.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
           
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))
            
            elif '//hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()                
                if io_type == 'io_stream':
                    inputs_str = ', '.join(
                        ['{type} {name}[{shape}]'.format(type=dtype, name=i.name, shape=i.size_cpp()) for i in
                        model_inputs])
                    outputs_str = ', '.join(
                        ['{type} {name}[{shape}]'.format(type=dtype, name=o.name, shape=o.size_cpp()) for o in
                        model_outputs])
                else:
                    inputs_str = ', '.join(
                        ['{type} {name}[{shape}]'.format(type=dtype, name=i.member_name, shape=i.size_cpp()) for i in
                        model_inputs])
                    outputs_str = ', '.join(
                        ['{type} {name}[{shape}]'.format(type=dtype, name=o.member_name, shape=o.size_cpp()) for o in
                        model_outputs])
                
                insize_str = ', '.join(
                    ['unsigned short &const_size_in_{}'.format(i) for i in range(1, len(model_inputs) + 1)])
                outsize_str = ', '.join(
                    ['unsigned short &const_size_out_{}'.format(o) for o in range(1, len(model_outputs) + 1)])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + ',\n'
                newline += indent + insize_str + ',\n'
                newline += indent + outsize_str + '\n'

            elif '//hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                if io_type == 'io_stream':
                    if len(model_inputs) > 1 or len(model_outputs) > 1:
                        raise Exception('io_stream Quartus supports exactly one input/output')
                    i = model_inputs[0]
                    o = model_outputs[0]

                    # Initialise stream object and store input data (C-array) to a 'stream' object
                    newline = indent + 'stream_in<{}> inputs;\n'.format(model_inputs[0].type.name)
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}, inputs);\n'.format(dtype, 
                                                                                            i.type.name,
                                                                                            i.size_cpp(),
                                                                                            i.name,
                                                                                        )
                    
                    # Initialise stream output
                    newline += '\n'
                    newline += indent + 'stream_out<{}> outputs;\n'.format(model_outputs[0].type.name)                    
                    
                    # Execute top-level function
                    top_level = indent + '{}(inputs, outputs);\n'.format(model.config.get_project_name())
                    newline += top_level
                    newline += '\n'

                    # Store data from 'stream' output to C-array, to be then returned and handled in Python
                    newline += indent + 'nnet::convert_data_back<{}, {}, {}>(outputs, {});\n'.format(o.type.name,
                                                                                                dtype,
                                                                                                o.size_cpp(),
                                                                                                o.name
                                                                                            )
                
                else:
                    # Convert input data from C-array to HLS type
                    newline = ''
                    newline += indent + 'input_data inputs_ap;\n'
                    for i in model_inputs:
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}, inputs_ap.{});\n'.format(dtype, i.type.name,
                                                                                                            i.size_cpp(),
                                                                                                            i.member_name,
                                                                                                            i.member_name)
                    newline += '\n'

                    # Initialise HLS output
                    newline += indent + 'output_data outputs_ap;\n'
                    
                    # Execute top-level function
                    top_level = indent + 'outputs_ap = {}(inputs_ap);\n'.format(model.config.get_project_name())
                    newline += top_level
                    newline += '\n'

                    # Convert HLS outputs back to C-array
                    for o in model_outputs:
                        newline += indent + 'nnet::convert_data_back<{}, {}, {}>(outputs_ap.{}, {});\n'.format(o.type.name,
                                                                                                                dtype,
                                                                                                                o.size_cpp(),
                                                                                                                o.member_name,
                                                                                                                o.member_name)
            elif '//hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp')
                    if func and model.config.trace_output and layer.get_attr('Trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += indent + 'nnet::trace_outputs->insert(std::pair<std::string, void *>("{}", (void *) malloc({} * element_size)));\n'.format(
                                layer.name, var.size_cpp())

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
        f = open(os.path.join(filedir, '../templates/quartus/Makefile'), 'r')
        fout = open('{}/Makefile'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():

            line = line.replace('myproject', model.config.get_project_name())

            if 'DEVICE   :=' in line:
                line = 'DEVICE   := {}\n'.format(model.config.get_config_value('Part'))

            fout.write(line)
        f.close()
        fout.close()

        ###################
        # build_lib.sh
        ###################

        f = open(os.path.join(filedir, '../templates/quartus/build_lib.sh'), 'r')
        fout = open('{}/build_lib.sh'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            line = line.replace('mystamp', model.config.get_config_value('Stamp'))

            fout.write(line)
        f.close()
        fout.close()

    def write_nnet_utils(self, model):
        ###################
        ## nnet_utils
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/quartus/firmware/nnet_utils/')
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

        srcpath = os.path.join(filedir, '../templates/quartus/ac_types/')
        dstpath = '{}/firmware/ac_types/'.format(model.config.get_output_dir())

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        ###################
        ## custom source
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = get_backend('Quartus').get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = '{}/firmware/{}'.format(model.config.get_output_dir(), dst)
            copyfile(srcpath, dstpath)

    def __get_table_size(self, model, activation):
        for layer in model.get_layers():
            if layer.get_attr('activation') == activation and layer.get_attr('table_size') is not None:
                return layer.get_attr('table_size')
        return 1024

    def __get_table_header(self, table_name, table_size):
        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += 'static const typename CONFIG_T::table_t {}[{}] = {{'.format(table_name, table_size)
        return table_header

    def __write_elu_table(self, model, path):
        table_name = 'elu_table'
        table_size = self.__get_table_size(model, 'elu')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = -8.0 * i / float(table_size)
            real_val = np.exp(in_val) - 1.
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_sigmoid_table(self, model, path):
        MAX_VALUE = 8
        MIN_VALUE = 0

        table_name = 'sigmoid_table'
        table_size = self.__get_table_size(model, 'sigmoid')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = i * (MAX_VALUE - MIN_VALUE) / float(table_size) + (MAX_VALUE - MIN_VALUE) / (
                        float(table_size) * 2) + MIN_VALUE
            real_val = 1.0 / (1 + np.exp(-in_val))
            if (real_val >= 0.5):
                h_file.write(sep + str(real_val))
                sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_tanh_table(self, model, path):
        MAX_VALUE = 4
        MIN_VALUE = 0

        table_name = 'tanh_table'
        table_size = self.__get_table_size(model, 'dense_tanh')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = i * (MAX_VALUE - MIN_VALUE) / float(table_size) + (MAX_VALUE - MIN_VALUE) / (
                        float(table_size) * 2) + MIN_VALUE
            real_val = np.tanh(in_val)
            if (real_val >= 0):
                h_file.write(sep + str(real_val))
                sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_softplus_table(self, model, path):
        table_name = 'softplus_table'
        table_size = self.__get_table_size(model, 'softplus')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = 2 * 8.0 * (i - float(table_size) / 2.0) / float(table_size)
            real_val = np.log(np.exp(in_val) + 1.)
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_softsign_table(self, model, path):
        table_name = 'softsign_table'
        table_size = self.__get_table_size(model, 'softsign')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = 2 * 8.0 * (i - float(table_size) / 2.0) / float(table_size)
            real_val = in_val / (np.fabs(in_val) + 1.)
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_selu_table(self, model, path):
        table_name = 'selu_table'
        table_size = self.__get_table_size(model, 'selu')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = -8.0 * i / float(table_size)
            real_val = 1.0507009873554804934193349852946 * (1.6732632423543772848170429916717 * (np.exp(in_val) - 1.))
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_exp_table(self, model, path):
        table_name = 'exp_table'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        # Default fixed point precision
        # 6 bits for integer part, 10 bits for decimal - total, 16
        fp_bits = 16
        fp_integer = 6
        fp_signed = True

        # Exp table should use the same precision as exp_table, as seen in Vivado code
        # init_exp_table<data_T, CONFIG_T>(exp_table);
        for layer in model.get_layers():
            if layer.name == 'softmax':
                ac_type = layer.get_input_variable().type
                if ac_type is not None:
                    try:
                        fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                        fp_integer = ac_type.precision.integer
                        fp_signed = ac_type.precision.signed
                    except:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            f.set_msb_bits(uint_to_binary(i, N))
            real_val = f.exp_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_invert_table(self, model, path):
        table_name = 'invert_table'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        # Default fixed point precision, in case values from layer attributes cannot be extracted
        # 8 bits for integer part, 10 bits for decimal - total, 18
        fp_bits = 18
        fp_integer = 8
        fp_signed = True

        # Invert table should use the same precision as exp_table, as seen in Vivado code
        # init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        for layer in model.get_layers():
            if layer.name == 'softmax':
                ac_type = layer.get_attr('exp_table_t')
                if ac_type is not None:
                    try:
                        fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                        fp_integer = ac_type.precision.integer
                        fp_signed = ac_type.precision.signed
                    except:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            f.set_msb_bits(uint_to_binary(i, N))
            real_val = f.inv_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_exp_table_latency(self, model, path):
        table_name = 'exp_table_latency'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        # Default fixed point precision
        # 6 bits for integer part, 10 bits for decimal - total, 16
        fp_bits = 16
        fp_integer = 6
        fp_signed = True

        # Exp table should use the same precision as exp_table, as seen in Vivado code
        # init_exp_table<data_T, CONFIG_T>(exp_table);
        for layer in model.get_layers():
            if layer.name == 'softmax':
                ac_type = layer.get_input_variable().type
                if ac_type is not None:
                    try:
                        fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                        fp_integer = ac_type.precision.integer
                        fp_signed = ac_type.precision.signed
                    except:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            f.set_msb_bits(uint_to_binary(i, N))
            real_val = f.exp_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_invert_table_latency(self, model, path):
        table_name = 'invert_table_latency'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        # Default fixed point precision, in case values from layer attributes cannot be extracted
        # 8 bits for integer part, 10 bits for decimal - total, 18
        fp_bits = 18
        fp_integer = 8
        fp_signed = True

        # Invert table should use the same precision as exp_table, as seen in Vivado code
        # init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        for layer in model.get_layers():
            if layer.name == 'softmax':
                ac_type = layer.get_attr('exp_table_t')
                if ac_type is not None:
                    try:
                        fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                        fp_integer = ac_type.precision.integer
                        fp_signed = ac_type.precision.signed
                    except:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            f.set_msb_bits(uint_to_binary(i, N))
            real_val = f.inv_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_exp_table_legacy(self, model, path):
        table_name = 'exp_table_legacy'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = 2 * 8.0 * (i - float(table_size) / 2.0) / float(table_size)
            real_val = np.exp(in_val)
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_invert_table_legacy(self, model, path):
        table_name = 'invert_table_legacy'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open('{}/{}.tb'.format(path, table_name), 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            real_val = 0
            in_val = 64.0 * i / float(table_size)
            if (in_val > 0.0):
                real_val = 1.0 / in_val
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def write_activation_tables(self, model):
        # Output path
        dstpath = '{}/firmware/nnet_utils/activation_tables'.format(model.config.get_output_dir())
        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        # Tables
        # TODO - Only write tables needed by model, not all of them
        self.__write_elu_table(model, dstpath)
        self.__write_sigmoid_table(model, dstpath)
        self.__write_tanh_table(model, dstpath)
        self.__write_softplus_table(model, dstpath)
        self.__write_softsign_table(model, dstpath)
        self.__write_selu_table(model, dstpath)
        self.__write_exp_table(model, dstpath)
        self.__write_invert_table(model, dstpath)
        self.__write_exp_table_latency(model, dstpath)
        self.__write_invert_table_latency(model, dstpath)
        self.__write_exp_table_legacy(model, dstpath)
        self.__write_invert_table_legacy(model, dstpath)

    def write_yml(self, model):
        ###################
        # YAML config file
        ###################

        def keras_model_representer(dumper, keras_model):
            model_path = model.config.get_output_dir() + '/keras_model.h5'
            keras_model.save(model_path)
            return dumper.represent_scalar(u'!keras_model', model_path)

        try:
            from tensorflow.keras import Model as KerasModel
            yaml.add_multi_representer(KerasModel, keras_model_representer)
        except:
            pass

        with open(model.config.get_output_dir() + '/' + config_filename, 'w') as file:
            yaml.dump(model.config.config, file)

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
        self.write_yml(model)
        self.write_tar(model)
        print('Done')
