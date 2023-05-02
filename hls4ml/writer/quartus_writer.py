import glob
import os
import tarfile
from collections import OrderedDict
from shutil import copyfile, copytree, rmtree

import numpy as np
import yaml

from hls4ml.backends import get_backend
from hls4ml.model.layers import Conv1D, Conv2D, Conv2DBatchnorm, Dense
from hls4ml.utils.fixed_point_utils import FixedPointEmulator, ceil_log2, uint_to_binary
from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class QuartusWriter(Writer):
    def next_pow2(self, x):
        return 1 << (x - 1).bit_length()

    def __make_dat_file(self, original_path, project_path):
        """
        Convert other input/output data types into a dat file, which is
        a text file with the falttened matrix printed out. Note that ' ' is
        assumed to be the delimiter.
        """

        # Take in data from current supported data files
        if original_path[-3:] == "npy":
            data = np.load(original_path)
        else:
            raise Exception("Unsupported input/output data files.")

        # Faltten data, just keep first dimension
        data = data.reshape(data.shape[0], -1)

        def print_data(f):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(data[i][j]) + " ")
                f.write("\n")

        # Print out in dat file
        with open(project_path, "w") as f:
            print_data(f)

    def get_max_reuse_factor(self, model):
        max_rf = 0
        for layer in model.get_layers():
            rf = int(layer.get_attr('reuse_factor'))
            if rf > max_rf:
                max_rf = rf
        return max_rf

    def print_array_to_cpp(self, var, layer, odir):
        """Write a weights array to C++ header files.

        Args:
            var (WeightVariable): Weight to write
            layer (Layer): Instance of the layer to which the weights belong
            odir (str): Output directory
        """
        h_file = open(f"{odir}/firmware/weights/{var.name}.h", "w")

        # meta data
        h_file.write(f"//Numpy array shape {var.shape}\n")
        h_file.write(f"//Min {np.min(var.min):.12f}\n")
        h_file.write(f"//Max {np.max(var.max):.12f}\n")
        h_file.write(f"//Number of zeros {var.nzeros}\n")
        h_file.write("\n")

        h_file.write(f"#ifndef {var.name.upper()}_H_\n")
        h_file.write(f"#define {var.name.upper()}_H_\n")
        h_file.write("\n")

        rf = int(layer.get_attr('reuse_factor', 1))
        weight_header = '#ifdef __INTELFPGA_COMPILER__\n'

        weight_size = 0
        if isinstance(layer, (Conv2D, Conv2DBatchnorm)):
            weight_size = (
                layer.get_attr('impl_filt_height')
                * layer.get_attr('impl_filt_width')
                * layer.get_attr('n_filt')
                * layer.get_attr('n_chan')
            )
        elif isinstance(layer, (Conv1D)):
            weight_size = layer.get_attr('impl_filt_width') * layer.get_attr('n_filt') * layer.get_attr('n_chan')
        elif isinstance(layer, (Dense)):
            weight_size = layer.get_attr('n_in') * layer.get_attr('n_out')

        if rf == 1 or var.name[0] == 'b' or weight_size <= 2048 or (var.name[0] == 'w' and var.type.precision.width < 3):
            weight_header += 'hls_init_on_powerup\n'
        else:
            block_factor = (layer.get_attr('n_in') * layer.get_attr('n_out')) / rf
            nbanks = int(2 ** np.ceil(np.log2(block_factor)) / 2)
            var_width = int(np.ceil(var.type.precision.width / 8))
            bwidth = self.next_pow2(var_width)
            weight_header += (
                f'hls_bankwidth({bwidth})\nhls_numbanks({nbanks})\nhls_max_replicates(1)\nhls_memory_impl("BLOCK_RAM")\n'
            )
        weight_header += '#endif\n'
        if var.storage.lower() == 'bram':
            weight_header += 'static '
        else:
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
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f"{model.config.get_output_dir()}/firmware/weights"):
            os.makedirs(f"{model.config.get_output_dir()}/firmware/weights")

    def write_project_cpp(self, model):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        project_name = model.config.get_project_name()

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/quartus/firmware/myproject.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{project_name}.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        io_type = model.config.get_config_value('IOType')
        indent = '   '
        brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', project_name)

            # Intel HLS 'streams' need to be passed by reference to top-level entity or declared as global variables
            # Streams cannot be declared inside a function
            # Therefore, layer connections (inputs/outputs) are declared here
            elif '// hls-fpga-machine-learning insert inter-task streams' in line:
                newline = line
                if io_type == 'io_stream':
                    for layer in model.get_layers():
                        vars = layer.get_variables()
                        for var in vars:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                newline += def_cpp + ';\n'

            # Instantiate GCC top-level function, to be used during GCC compilation / hls4ml.predict()
            elif '// hls-fpga-machine-learning instantiate GCC top-level' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += f'void {project_name}(\n'
                    for inp in model_inputs:
                        newline += indent + f'stream_in<{inp.type.name}> &{inp.name}_stream,\n'
                    for out in model_outputs:
                        newline += indent + f'stream_out<{out.type.name}> &{out.name}_stream'
                    if model_brams:
                        newline += ',\n' + brams_str
                    newline += '\n) {\n'
                if io_type == 'io_parallel':
                    newline = f'output_data {project_name}(\n'
                    newline += indent + 'input_data inputs'
                    if model_brams:
                        newline += ',\n' + brams_str
                    newline += '\n) {\n'

            # Instantiate HLS top-level function, to be used during HLS synthesis
            elif '// hls-fpga-machine-learning instantiate HLS top-level' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += f'component void {project_name}(\n'
                    for inp in model_inputs:
                        newline += indent + f'stream_in<{inp.type.name}> &{inp.name}_stream,\n'
                    for out in model_outputs:
                        newline += indent + f'stream_out<{out.type.name}> &{out.name}_stream'
                    if model_brams:
                        newline += ',\n' + brams_str
                    newline += '\n) {\n'
                if io_type == 'io_parallel':
                    newline += f'component output_data {project_name}(\n'
                    newline += indent + 'input_data inputs'
                    if model_brams:
                        newline += ',\n' + brams_str
                    newline += '\n) {\n'

            # Insert HLS pragmas such as maximum frequency, initiation interval etc.
            elif '// hls-fpga-machine-learning insert cpragmas' in line:
                newline = line
                if io_type == 'io_parallel':
                    newline += 'hls_max_concurrency(0)\n'
                    newline += f'hls_component_ii({self.get_max_reuse_factor(model)})\n'
                clock_mhz = 1000 / (model.config.get_config_value('ClockPeriod'))
                newline += f'hls_scheduler_target_fmax_mhz({np.ceil(clock_mhz).astype(int)})\n'

            # In io_parallel, an output (struct) is returned from the top-level function
            # Therefore, it needs to be initialised before returning
            # In io_stream, the input is of type 'stream_in' and output is of type 'stream_out'
            # However, individual layers accept the type 'stream'
            # Therefore, data is first read from 'stream_in', written to 'stream' and propagated through network
            elif '// hls-fpga-machine-learning initialize input/output' in line:
                if io_type == 'io_stream':
                    newline = line
                    for inp in model_inputs:
                        newline += indent + f'for (size_t i = 0; i < {inp.size_cpp()} / {inp.type.name}::size; i++) {{\n'
                        newline += indent + f'  {inp.type.name} tmp = {inp.name}_stream.read();\n'
                        newline += indent + f'  {inp.name}.write(tmp);\n'
                        newline += indent + '}\n'
                else:
                    newline = line
                    newline += indent + 'hls_register output_data outputs;\n'

            # Insert weights
            elif '// hls-fpga-machine-learning insert weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w not in model_brams:
                            newline += f'#include "weights/{w.name}.h"\n'

            # Insert test weights
            elif '// hls-fpga-machine-learning insert test weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += f'#include "weights/{w.name}_test.h"\n'

            # Neural net instantiation
            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                model_inputs = model.get_input_variables()
                model_outputs = model.get_output_variables()
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
                        if model.config.trace_output and layer.get_attr('trace', False):
                            newline += '#ifndef HLS_SYNTHESIS\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                    var.type.name, var.name, layer.name, var.size_cpp()
                                )
                            newline += '#endif\n'
                        newline += '\n'

            # In io_parallel, a return is required; for more details see myproject.cpp & myproject.h
            elif '// hls-fpga-machine-learning return' in line:
                if io_type == 'io_stream':
                    newline = line
                    for out in model_outputs:
                        newline += indent + f'for (size_t i = 0; i < {out.size_cpp()} / {out.type.name}::size; i++) {{\n'
                        newline += indent + f'  {out.type.name} tmp = {out.name}.read();\n'
                        newline += indent + f'  {out.name}_stream.write(tmp);\n'
                        newline += indent + '}\n'
                        newline += '}\n'
                else:
                    newline = line
                    newline += indent + 'return outputs;\n'
                    newline += '}\n'

            # Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_project_header(self, model):
        """Write the main architecture header file (myproject.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        project_name = model.config.get_project_name()

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/quartus/firmware/myproject.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{project_name}.h', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        # io_parallel and io_stream instantiate the top-level function differently
        io_type = model.config.get_config_value('IOType')
        indent = '    '
        brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(project_name.upper()))

            elif 'myproject' in line:
                newline = line.replace('myproject', project_name)

            elif '// hls-fpga-machine-learning instantiate GCC top-level' in line:
                newline = line
                # For io_stream, input and output are passed by reference; see myproject.h & myproject.cpp for more details

                if io_type == 'io_stream':
                    newline += f'void {project_name}(\n'
                    for inp in model_inputs:
                        newline += indent + f'stream_in<{inp.type.name}> &{inp.name}_stream,\n'
                    for out in model_outputs:
                        newline += indent + f'stream_out<{out.type.name}> &{out.name}_stream'
                    if model_brams:
                        newline += ',\n' + brams_str
                    newline += '\n);\n'
                # In io_parallel, a struct is returned; see myproject.h & myproject.cpp for more details
                else:
                    newline += f'output_data {project_name}(\n'
                    newline += indent + 'input_data inputs'
                    if model_brams:
                        newline += ',\n' + brams_str
                    newline += '\n);\n'

            # Similar to GCC instantiation, but with the keyword 'component'
            elif '// hls-fpga-machine-learning instantiate HLS top-level' in line:
                newline = line
                if io_type == 'io_stream':
                    newline += f'component void {project_name}(\n'
                    for inp in model_inputs:
                        newline += indent + f'stream_in<{inp.type.name}> &{inp.name}_stream,\n'
                    for out in model_outputs:
                        newline += indent + f'stream_out<{out.type.name}> &{out.name}_stream'
                    if model_brams:
                        newline += ',\n' + brams_str
                    newline += '\n);\n'
                else:
                    newline += f'component output_data {project_name}(\n'
                    newline += indent + 'input_data inputs'
                    if model_brams:
                        newline += ',\n' + brams_str
                    newline += '\n);\n'

            elif '// hls-fpga-machine-learning insert cpragmas' in line:
                newline = line
                if io_type == 'io_parallel':
                    newline += 'hls_max_concurrency(0)\n'
                    newline += f'hls_component_ii({self.get_max_reuse_factor(model)})\n'
                clock_mhz = 1000 / (model.config.get_config_value('ClockPeriod'))
                newline += f'hls_scheduler_target_fmax_mhz({np.ceil(clock_mhz).astype(int)})\n'

            # For io_stream, no inputs/outputs are instantiated, as they are passed by reference
            # For io_parallel, input/output structs are required
            elif '// hls-fpga-machine-learning insert inputs' in line:
                newline = line
                if io_type != 'io_stream':
                    newline += 'struct input_data { \n'
                    for inp in model_inputs:
                        newline += indent + inp.definition_cpp() + ';\n'
                    newline += '};\n'
            elif '// hls-fpga-machine-learning insert outputs' in line:
                newline = line
                if io_type != 'io_stream':
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
        """Write the C++ type definitions file (defines.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/quartus/firmware/defines.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/defines.h', 'w')

        for line in f.readlines():

            # Insert numbers
            if '// hls-fpga-machine-learning insert numbers' in line:
                newline = line

                defines_list = []
                for layer in model.get_layers():
                    defines = ''
                    for k, v in layer.get_output_variable().get_shape():
                        defines += f'#define {k} {v}\n'

                    defines_list.append(defines)

                newline += ''.join(defines_list)

            elif '// hls-fpga-machine-learning insert layer-precision' in line:
                newline = line
                all_precision = OrderedDict()
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    for type_name, type_var in layer_precision.items():
                        # Ensure that layer's types doesn't override existing types
                        # This can happen in case of InplaceVariable types
                        if type_name not in all_precision:
                            all_precision[type_name] = type_var
                for used_type in all_precision.values():
                    newline += used_type.definition_cpp()

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_parameters(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/quartus/firmware/parameters.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/parameters.h', 'w')

        for line in f.readlines():

            if '// hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include

            elif "// hls-fpga-machine-learning insert layer-config" in line:
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
        """Write the weights into header files

        Args:
            model (ModelGraph): the hls4ml model.
        """
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(weights, layer, model.config.get_output_dir())

    def write_testbench_parallel(self, model):
        """Write the testbench file for io_parallel (myproject_test.cpp and input/output .dat files)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if len(model.get_output_variables()) != 1:
            print("WARNING:  The testbench only supports one output variable. Leaving empty testbench")
            return

        outvar = model.get_output_variables()[0]

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists(f'{model.config.get_output_dir()}/tb_data/'):
            os.mkdir(f'{model.config.get_output_dir()}/tb_data/')

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == "dat":
                copyfile(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')
            else:
                self.__make_dat_file(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')

        if output_predictions:
            if output_predictions[-3:] == "dat":
                copyfile(output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat')
            else:
                self.__make_dat_file(
                    output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat'
                )

        f = open(os.path.join(filedir, '../templates/quartus/myproject_test_parallel.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp', 'w')

        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'
            elif '// hls-fpga-machine-learning insert data' in line:
                newline = line
                newline += '      std::vector<float>::const_iterator in_begin = in.cbegin();\n'
                newline += '      std::vector<float>::const_iterator in_end;\n'
                newline += '      inputs.emplace_back();\n'
                for inp in model.get_input_variables():
                    newline += f'      in_end = in_begin + ({inp.size_cpp()});\n'
                    newline += f'      std::copy(in_begin, in_end, inputs.back().{inp.member_name});\n'
                    newline += '      in_begin = in_end;\n'
                newline += '      outputs.emplace_back();\n'
            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                newline += indent + 'for(int i = 0; i < num_iterations; i++) {\n'
                for inp in model.get_input_variables():
                    newline += indent + '  inputs.emplace_back();\n'
                    newline += indent + '  outputs.emplace_back();\n'
                    newline += indent + f'  std::fill_n(inputs[i].{inp.member_name}, {inp.size_cpp()}, 0.0);\n'
                newline += indent + '}\n'

            elif '// hls-fpga-machine-learning insert top-level-function' in line:
                newline = line
                newline += indent + 'for(int i = 0; i < num_iterations; i++) {\n'
                newline += indent + f'  ihc_hls_enqueue(&outputs[i], {model.config.get_project_name()}, inputs[i]'
                if model_brams:
                    bram_vars = ','.join([b.name for b in model_brams])
                    newline += f', {bram_vars});\n'
                else:
                    newline += ');\n'
                newline += indent + '}\n'
            elif 'hls-fpga-machine-learning insert run' in line:
                newline = line
                newline += '    ' + f'ihc_hls_component_run_all({model.config.get_project_name()});\n'
            elif '// hls-fpga-machine-learning insert predictions' in line:
                newline = line
                newline += indent + f'for(int i = 0; i < {outvar.size_cpp()}; i++) {{\n'
                newline += indent + '  std::cout << predictions[j][i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'std::cout << std::endl;\n'
            elif '// hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                newline += indent + f'for(int i = 0; i < {outvar.size_cpp()}; i++) {{\n'
                newline += indent + f'  fout << outputs[j].{outvar.member_name}[i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'fout << std::endl;\n'
            elif (
                '// hls-fpga-machine-learning insert output' in line
                or '// hls-fpga-machine-learning insert quantized' in line
            ):
                newline = line
                newline += indent + f'for(int i = 0; i < {outvar.size_cpp()}; i++) {{\n'
                newline += indent + f'  std::cout << outputs[j].{outvar.member_name}[i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'std::cout << std::endl;\n'
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_testbench_stream(self, model):
        """Write the testbench file for io_stream (myproject_test.cpp and input/output .dat files)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if len(model.get_output_variables()) != 1:
            print("WARNING:  The testbench only supports one output variable. Leaving empty testbench")
            return

        outvar = model.get_output_variables()[0]

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists(f'{model.config.get_output_dir()}/tb_data/'):
            os.mkdir(f'{model.config.get_output_dir()}/tb_data/')

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == "dat":
                copyfile(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')
            else:
                self.__make_dat_file(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')

        if output_predictions:
            if output_predictions[-3:] == "dat":
                copyfile(output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat')
            else:
                self.__make_dat_file(
                    output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat'
                )

        f = open(os.path.join(filedir, '../templates/quartus/myproject_test_stream.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp', 'w')

        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

            elif '// hls-fpga-machine learning instantiate inputs and outputs' in line:
                newline = line
                for inp in model_inputs:
                    newline += indent + f'stream_in<{inp.type.name}> {inp.name}_input;\n'
                for out in model_outputs:
                    newline += indent + f'stream_out<{out.type.name}> {out.name}_output;\n'

            # TODO - This is one-input specific (are multiple model inputs needed at all?)
            elif '// hls-fpga-machine-learning insert data' in line:
                newline = line
                c = 0
                for inp in model_inputs:
                    newline += indent + f'float vals_{c}[{inp.size_cpp()}]; \n'
                    newline += indent + f'for (int j = 0 ; j < {inp.size_cpp()} ; j++) {{\n'
                    newline += indent + indent + f'vals_{c}[j] = in[j]; \n'
                    newline += indent + '}\n'
                    newline += (
                        indent
                        + f'nnet::convert_data<float, {inp.type.name}, {inp.size_cpp()}>(vals_{c}, {inp.name}_input);\n'
                    )
                    c += 1

            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                c = 0
                for inp in model_inputs:
                    newline += indent + f'float vals_{c}[{inp.size_cpp()}]; \n'
                    newline += indent + f'for (int j = 0 ; j < {inp.size_cpp()} ; j++) {{\n'
                    newline += indent + indent + f'vals_{c}[j] = 0.0; \n'
                    newline += indent + '}\n'
                    newline += (
                        indent
                        + f'nnet::convert_data<float, {inp.type.name}, {inp.size_cpp()}>(vals_{c}, {inp.name}_input);\n'
                    )
                    c += 1

            elif '// hls-fpga-machine-learning insert top-level-function' in line:
                newline = line
                input_params = ', '.join([f'{i.name}_input' for i in model_inputs])
                output_params = ', '.join([f'{o.name}_output' for o in model_outputs])
                newline += (
                    indent + f'ihc_hls_enqueue_noret(&{model.config.get_project_name()}, {input_params}, {output_params}'
                )
                if model_brams:
                    bram_vars = ','.join([b.name for b in model_brams])
                    newline += f', {bram_vars});\n'
                else:
                    newline += ');\n'

            elif 'hls-fpga-machine-learning insert run' in line:
                newline = line
                newline += indent + f'ihc_hls_component_run_all({model.config.get_project_name()});\n'

            elif '// hls-fpga-machine-learning convert output' in line:
                newline = line
                newline += indent + f'float res[{outvar.size_cpp()}];\n'
                newline += indent + 'nnet::convert_data_back<{}, float, {}>({}_output, res);\n'.format(
                    outvar.type.name, outvar.size_cpp(), outvar.name
                )

            elif '// hls-fpga-machine-learning insert tb-output' in line:
                newline += indent + f'for(int i = 0; i < {outvar.size_cpp()}; i++) {{\n'
                newline += indent + '  fout << res[i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'fout << std::endl;\n'

            elif '// hls-fpga-machine-learning print predictions' in line:
                newline = line
                newline += indent + f'for(int i = 0; i < {outvar.size_cpp()}; i++) {{\n'
                newline += indent + '  std::cout << predictions[iteration][i] << " ";\n'
                newline += indent + '}\n'
                newline += indent + 'std::cout << std::endl;\n'

            elif '// hls-fpga-machine-learning print output' in line:
                newline = line
                newline += indent + f'for(int i = 0; i < {outvar.size_cpp()}; i++) {{\n'
                newline += indent + '  std::cout << res[i] << " "; \n'
                newline += indent + '} \n'
                newline += indent + 'std::cout << std::endl; \n'
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_test_bench(self, model):
        """Write the testbench

        Args:
            model (ModelGraph): the hls4ml model.
        """
        # TODO - This function only works with one model input
        # (NOT one data point - it works as expected with multiple data points)
        io_type = model.config.get_config_value('IOType')
        if io_type == 'io_parallel':
            self.write_testbench_parallel(model)
        elif io_type == 'io_stream':
            self.write_testbench_stream(model)

    def write_bridge(self, model):
        """Write the Python-C++ bridge (myproject_bridge.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/quartus/myproject_bridge.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        io_type = model.config.get_config_value('IOType')
        indent = '    '

        for line in f.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))

            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                if io_type == 'io_stream':
                    inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                    outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])
                else:
                    inputs_str = ', '.join([f'{dtype} {i.member_name}[{i.size_cpp()}]' for i in model_inputs])
                    outputs_str = ', '.join([f'{dtype} {o.member_name}[{o.size_cpp()}]' for o in model_outputs])

                insize_str = ', '.join([f'unsigned short &const_size_in_{i}' for i in range(1, len(model_inputs) + 1)])
                outsize_str = ', '.join([f'unsigned short &const_size_out_{o}' for o in range(1, len(model_outputs) + 1)])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + ',\n'
                newline += indent + insize_str + ',\n'
                newline += indent + outsize_str + '\n'

            elif '// hls-fpga-machine-learning insert wrapper' in line:
                bram_params = ''
                if model_brams:
                    bram_params = ', ' + ','.join([b.name for b in model_brams])

                dtype = line.split('#', 1)[1].strip()
                if io_type == 'io_stream':
                    newline = ''
                    for i in model_inputs:
                        # Initialise stream object and store input data (C-array) to a 'stream' object
                        newline += indent + f'stream_in<{i.type.name}> {i.name}_input;\n'
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_input);\n'.format(
                            dtype, i.type.name, i.size_cpp(), i.name, i.name
                        )

                    # Initialise stream output
                    for o in model_outputs:
                        newline += '\n'
                        newline += indent + f'stream_out<{o.type.name}> {o.name}_output;\n'

                    # Execute top-level function
                    input_params = ', '.join([f'{i.name}_input' for i in model_inputs])
                    output_params = ', '.join([f'{o.name}_output' for o in model_outputs])

                    top_level = (
                        indent + f'{model.config.get_project_name()}({input_params}, {output_params}{bram_params});\n'
                    )
                    newline += top_level
                    newline += '\n'

                    # Store data from 'stream' output to C-array, to be then returned and handled in Python
                    for o in model_outputs:
                        newline += indent + 'nnet::convert_data_back<{}, {}, {}>({}_output, {});\n'.format(
                            o.type.name, dtype, o.size_cpp(), o.name, o.name
                        )

                else:
                    # Convert input data from C-array to HLS type
                    newline = ''
                    newline += indent + 'input_data inputs_ap;\n'
                    for i in model_inputs:
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}, inputs_ap.{});\n'.format(
                            dtype, i.type.name, i.size_cpp(), i.member_name, i.member_name
                        )
                    newline += '\n'

                    # Initialise HLS output
                    newline += indent + 'output_data outputs_ap;\n'

                    # Execute top-level function
                    top_level = indent + f'outputs_ap = {model.config.get_project_name()}(inputs_ap{bram_params});\n'
                    newline += top_level
                    newline += '\n'

                    # Convert HLS outputs back to C-array
                    for o in model_outputs:
                        newline += indent + 'nnet::convert_data_back<{}, {}, {}>(outputs_ap.{}, {});\n'.format(
                            o.type.name, dtype, o.size_cpp(), o.member_name, o.member_name
                        )
            elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp')
                    if func and model.config.trace_output and layer.get_attr('trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += (
                                indent
                                + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                            )

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_build_script(self, model):
        """Write the build scripts (Makefile, build_lib.sh)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # Makefile
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/quartus/Makefile'))
        fout = open(f'{model.config.get_output_dir()}/Makefile', 'w')

        for line in f.readlines():

            line = line.replace('myproject', model.config.get_project_name())

            if 'DEVICE   :=' in line:
                line = 'DEVICE   := {}\n'.format(model.config.get_config_value('Part'))

            fout.write(line)
        f.close()
        fout.close()

        # build_lib.sh
        f = open(os.path.join(filedir, '../templates/quartus/build_lib.sh'))
        fout = open(f'{model.config.get_output_dir()}/build_lib.sh', 'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            line = line.replace('mystamp', model.config.get_config_value('Stamp'))

            fout.write(line)
        f.close()
        fout.close()

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/quartus/firmware/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        # ac_types
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/quartus/ac_types/')
        dstpath = f'{model.config.get_output_dir()}/firmware/ac_types/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        # custom source
        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = get_backend('Quartus').get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = f'{model.config.get_output_dir()}/firmware/{dst}'
            copyfile(srcpath, dstpath)

    def __get_table_size(self, model, activation):
        for layer in model.get_layers():
            if (
                layer.get_attr('activation') == activation or layer.get_attr('recurrent_activation') == activation
            ) and layer.get_attr('table_size') is not None:
                return int(layer.get_attr('table_size'))
        return 1024

    def __get_table_header(self, table_name, table_size):
        table_header = '#ifdef __INTELFPGA_COMPILER__\n'
        table_header += 'hls_init_on_powerup\n'
        table_header += '#endif\n'
        table_header += f'static const typename CONFIG_T::table_t {table_name}[{table_size}] = {{'
        return table_header

    def __write_elu_table(self, model, path):
        table_name = 'elu_table'
        table_size = self.__get_table_size(model, 'elu')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = -8.0 * i / float(table_size)
            real_val = np.exp(in_val) - 1.0
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_sigmoid_table(self, model, path):
        MAX_VALUE = 8
        MIN_VALUE = 0

        table_name = 'sigmoid_table'
        table_size = self.__get_table_size(model, 'sigmoid')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(int(table_size)):
            in_val = (
                i * (MAX_VALUE - MIN_VALUE) / float(table_size)
                + (MAX_VALUE - MIN_VALUE) / (float(table_size) * 2)
                + MIN_VALUE
            )
            real_val = 1.0 / (1 + np.exp(-in_val))
            if real_val >= 0.5:
                h_file.write(sep + str(real_val))
                sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_tanh_table(self, model, path):
        MAX_VALUE = 4
        MIN_VALUE = 0

        table_name = 'tanh_table'
        table_size = self.__get_table_size(model, 'tanh')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = (
                i * (MAX_VALUE - MIN_VALUE) / float(table_size)
                + (MAX_VALUE - MIN_VALUE) / (float(table_size) * 2)
                + MIN_VALUE
            )
            real_val = np.tanh(in_val)
            if real_val >= 0:
                h_file.write(sep + str(real_val))
                sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_softplus_table(self, model, path):
        table_name = 'softplus_table'
        table_size = self.__get_table_size(model, 'softplus')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = 2 * 8.0 * (i - float(table_size) / 2.0) / float(table_size)
            real_val = np.log(np.exp(in_val) + 1.0)
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_softsign_table(self, model, path):
        MAX_VALUE = 8
        MIN_VALUE = 0
        table_name = 'softsign_table'
        table_size = self.__get_table_size(model, 'softsign')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):

            in_val = (
                i * (MAX_VALUE - MIN_VALUE) / float(table_size)
                + (MAX_VALUE - MIN_VALUE) / (float(table_size) * 2)
                + MIN_VALUE
            )

            real_val = in_val / (np.fabs(in_val) + 1.0)
            if real_val >= 0:
                h_file.write(sep + str(real_val))
                sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_selu_table(self, model, path):
        table_name = 'selu_table'
        table_size = self.__get_table_size(model, 'selu')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = -8.0 * i / float(table_size)
            real_val = 1.0507009873554804934193349852946 * (1.6732632423543772848170429916717 * (np.exp(in_val) - 1.0))
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_exp_table(self, model, path):
        table_name = 'exp_table'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open(f'{path}/{table_name}.tb', 'w')
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
                    except Exception:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass
                    if fp_signed is False:
                        raise Exception('Softmax types need to be signed')

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            b = uint_to_binary(i, N)
            if i == 0:
                b.insert(0, 0)
            else:
                b.insert(0, 1)
            f.set_msb_bits(b)
            real_val = f.exp_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_invert_table(self, model, path):
        table_name = 'invert_table'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open(f'{path}/{table_name}.tb', 'w')
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
                    except Exception:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass
                    if fp_signed is False:
                        raise Exception('Softmax types need to be signed')

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            b = uint_to_binary(i, N)
            b.insert(0, 0)
            f.set_msb_bits(b)
            real_val = f.inv_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_exp_table_latency(self, model, path):
        table_name = 'exp_table_latency'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open(f'{path}/{table_name}.tb', 'w')
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
                    except Exception:
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

        h_file = open(f'{path}/{table_name}.tb', 'w')
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
                    except Exception:
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

        h_file = open(f'{path}/{table_name}.tb', 'w')
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

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            real_val = 0
            in_val = 64.0 * i / float(table_size)
            if in_val > 0.0:
                real_val = 1.0 / in_val
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def write_activation_tables(self, model):
        """Write the lookup tables for activation functions

        Args:
            model (ModelGraph): the hls4ml model.
        """
        # Output path
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/activation_tables'
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
        """Write the config to the YAML file

        Args:
            model (ModelGraph): the hls4ml model.
        """

        def keras_model_representer(dumper, keras_model):
            model_path = model.config.get_output_dir() + '/keras_model.h5'
            keras_model.save(model_path)
            return dumper.represent_scalar('!keras_model', model_path)

        try:
            from tensorflow.keras import Model as KerasModel

            yaml.add_multi_representer(KerasModel, keras_model_representer)
        except Exception:
            pass

        with open(model.config.get_output_dir() + '/' + config_filename, 'w') as file:
            yaml.dump(model.config.config, file)

    def write_tar(self, model):
        """Write the generated project as a .tar.gz archive

        Args:
            model (ModelGraph): the hls4ml model.
        """

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
