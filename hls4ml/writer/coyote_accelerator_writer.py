import os
import stat
import glob
import numpy as np
from pathlib import Path
from shutil import copyfile, copytree, move

from hls4ml.writer.vitis_writer import VitisWriter

class CoyoteAcceleratorWriter(VitisWriter):
    def __init__(self):
        super().__init__()

    def write_coyote(self, model):
        """
        Copies the Coyote repository to the project folder

        Args:
            model (ModelGraph): the hls4ml model
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath = os.path.join(filedir, '../contrib/Coyote/')
        dstpath = f'{model.config.get_output_dir()}/Coyote'
        copytree(srcpath, dstpath)

    def restructure_dir(self, model):  
        """
        Simply moves around some files; these files were generated from the Vitis backend
        For a cleaner integration with the rest of the Coyote library, these are
        moved to the src/ folder

        Args:
            model (ModelGraph): the hls4ml model
        """      
        srcpath = f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp'
        dstpath = f'{model.config.get_output_dir()}/src/{model.config.get_project_name()}_bridge.cpp'
        move(srcpath, dstpath)

        srcpath = f'{model.config.get_output_dir()}/firmware'
        dstpath = f'{model.config.get_output_dir()}/src/hls/model_wrapper/firmware'
        move(srcpath, dstpath)

    def write_project_cpp(self, model):
        """
        Write the main architecture source file (myproject.cpp)
        Very similar to VivadoWriter, but with a different generation for I/O.
        Since the myproject.cpp is no longer the top-level file (but model_wrapper is),
        no need to specify interfaces. Additionally, inlining can cause issues here
        when integrated with the model_wrapper, so it's disabled.

        Args:
            model (ModelGraph): the hls4ml model
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/vivado/firmware/myproject.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str
                if len(model_brams) > 0:
                    newline += ',\n' + brams_str
                newline += '\n'

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                if model.config.get_writer_config()['WriteWeightsTxt']:

                    newline += '#ifndef __SYNTHESIS__\n'
                    newline += '    static bool loaded_weights = false;\n'
                    newline += '    if (!loaded_weights) {\n'

                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            if w.weight_class == 'CompressedWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.nonzeros, w.name, w.name
                                    )
                                )
                            elif w.weight_class == 'ExponentWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.data_length, w.name, w.name
                                    )
                                )
                            else:
                                newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                    w.type.name, w.data_length, w.name, w.name
                                )

                    newline += '        loaded_weights = true;'
                    newline += '    }\n'
                    newline += '#endif'

            # Add input/output type
            elif '// hls-fpga-machine-learning insert IO' in line:
                newline = line
                newline += indent + '#pragma HLS INLINE OFF\n'
    
                pipeline_style = model.config.pipeline_style
                pipeline_ii = model.config.pipeline_ii
                pipeline_pragma = indent + f'#pragma HLS {pipeline_style.upper()}'
                if pipeline_style == 'pipeline' and pipeline_ii is not None:
                    pipeline_pragma += f' II={pipeline_ii}\n'
                else:
                    pipeline_pragma += '\n'
                newline += pipeline_pragma

            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in model_inputs and var not in model_outputs:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                newline += '    ' + def_cpp + ';\n'
                                if var.pragma:
                                    newline += '    ' + self._make_array_pragma(var) + '\n\n'
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func:
                        if not isinstance(func, (list, set)):
                            func = [func]
                        if len(func) == 1:
                            newline += '    ' + func[0] + ' // ' + layer.name + '\n'
                        else:
                            newline += '    // ' + layer.name + '\n'
                            for line in func:
                                newline += '    ' + line + '\n'
                        if model.config.trace_output and layer.get_attr('trace', False):
                            vars = layer.get_variables()
                            newline += '#ifndef __SYNTHESIS__\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                    var.type.name, var.name, layer.name, var.size_cpp()
                                )
                            newline += '#endif\n'
                        newline += '\n'

            # Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_nnet_utils_overrides(self, model):
        """
        Writes the HLS templates, both from Vitis and from Coyote
        
        Args:
            model (ModelGraph): the hls4ml model
        """
        filedir = os.path.dirname(os.path.abspath(__file__))

        # Vitis HLS overwrites, as done in VitisWriter
        srcpath = os.path.join(filedir, '../templates/vitis/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'
        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]
        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        # Coyote accelerator-specific overvwrites
        srcpath = os.path.join(filedir, '../templates/coyote_accelerator/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'
        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]
        for h in headers:
            copyfile(srcpath + h, dstpath + h)

    def write_build_script(self, model):
        """
        Generate the following build scripts:
            - build_lib.sh --- used for software emulation (with gcc) of the model
            - CMakeLists.txt --- for synthesizing the hardware with Coyote and the corresponding software library

        Args:
            model (ModelGraph): the hls4ml model
        """
        filedir = Path(__file__).parent
        
        # build_lib.sh
        build_lib_src = (filedir / '../templates/coyote_accelerator/build_lib.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))
                dst.write(line)

        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

        # CMakeLists.txt
        cmake_src = os.path.join(filedir, '../templates/coyote_accelerator/CMakeLists.txt')
        cmake_dst = f'{model.config.get_output_dir()}/CMakeLists.txt'
        with open(cmake_src) as src, open(cmake_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                dst.write(line)

    def write_model_wrapper(self, model):
        """
        Generate the model_wrapper and vfpga_top
        
        model_wrapper encapsulates the hls4ml model kernel as well as AXI-to-data
        and data-to-AXI converters. More details on the model_wrapper and these 
        converters can be found in model_wrapper.hpp.

        vfpga_top.svh is a simple SystemVerilog header that is needed to synthesize
        any Coyote project; see vfpga_top.svh and the Coyote examples for more details

        Args:
            model (ModelGraph): the hls4ml model
        """
        filedir = Path(__file__).parent

        if not os.path.isdir(f'{model.config.get_output_dir()}/src/hls/model_wrapper'):
            os.makedirs(f'{model.config.get_output_dir()}/src/hls/model_wrapper')
        
        # model_wrapper.h
        srcpath = (filedir / '../templates/coyote_accelerator/model_wrapper.hpp').resolve()
        dstpath = f'{model.config.get_output_dir()}/src/hls/model_wrapper/model_wrapper.hpp'
        copyfile(srcpath, dstpath)

        # model_wrapper.cpp
        f = open(os.path.join(filedir, '../templates/coyote_accelerator/model_wrapper.cpp'))
        fout = open(f'{model.config.get_output_dir()}/src/hls/model_wrapper/model_wrapper.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        if len(model_inputs) > 1 or len(model_outputs) > 1:
            raise RuntimeError('CoyoteAccelerator backend currently only supports one input and one output')

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert data' in line:
                newline = ''
                io_type = model.config.get_config_value('IOType')

                for inp in model_inputs:
                    newline += indent + inp.definition_cpp() + ';\n'      
                    newline += indent + self._make_array_pragma(inp) + '\n\n'
                    
                for out in model_outputs:
                    newline += indent + out.definition_cpp() + ';\n'
                    newline += indent + self._make_array_pragma(out) + '\n\n'
                    
            elif '// hls-fpga-machine-learning insert top-level function' in line:
                newline = ''

                for inp in model_inputs:
                    newline += indent + f'nnet::axi_stream_to_data<{inp.type.name}, float, {inp.size_cpp()}, COYOTE_AXI_STREAM_BITS, 8 * sizeof(float)>(data_in, {inp.name});\n'
                
                input_vars = ','.join([i.name for i in model_inputs])
                output_vars = ','.join([o.name for o in model_outputs])
                all_vars = ','.join(filter(None, [input_vars, output_vars]))
                top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'
                newline += top_level

                for out in model_outputs:
                    newline += indent + f'nnet::data_to_axi_stream<{out.type.name}, float, {out.size_cpp()}, COYOTE_AXI_STREAM_BITS, 8 * sizeof(float)>({out.name}, data_out);\n'

            else:
                newline = line
            
            fout.write(newline)
        
        f.close()
        fout.close()

        # vfpga_top.svh
        srcpath = (filedir / '../templates/coyote_accelerator/vfpga_top.svh').resolve()
        dstpath = f'{model.config.get_output_dir()}/src/vfpga_top.svh'
        copyfile(srcpath, dstpath)

        # init_ip.tcl for any additional IPs that may be needed for the model (e.g., ILA for debugging) --- UNUSED FOR NOW
        # srcpath = (filedir / '../templates/coyote_accelerator/init_ip.tcl').resolve()
        # dstpath = f'{model.config.get_output_dir()}/src/init_ip.tcl'

        copyfile(srcpath, dstpath)

    def write_host_code(self, model):
        """
        Generates the host code, namely myproject_host.cpp and host_libs.hpp
        host_libs.hpp implements the "glue" logic which interacts with the Coyote 
        software library. myproject_host.cpp is a stand-alone program that can be 
        compiled and used to run model inference on an FPGA, with inputs from tb_data.

        Args:
            model (ModelGraph): the hls4ml model
        """
        filedir = Path(__file__).parent

        if not os.path.isdir(f'{model.config.get_output_dir()}/src/'):
            os.makedirs(f'{model.config.get_output_dir()}/src/')

        # myproject_host.cpp
        f = open(os.path.join(filedir, '../templates/coyote_accelerator/myproject_host.cpp'))
        fout = open(f'{model.config.get_output_dir()}/src/{model.config.get_project_name()}_host.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        if len(model_inputs) > 1 or len(model_outputs) > 1:
            raise RuntimeError('CoyoteAccelerator backend currently only supports one input and one output')

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            if '// hls-fpga-machine-learning insert I/O size' in line:
                newline = ''
                for inp in model_inputs:
                    newline += indent + f'constexpr const unsigned int in_size = {inp.size_cpp()};\n'
                for out in model_outputs:
                    newline += indent + f'constexpr const unsigned int out_size = {out.size_cpp()};\n'
            
            else:
                newline = line
            
            fout.write(newline)

        f.close()
        fout.close()

        # host_libs.hpp
        srcpath = os.path.join(filedir, '../templates/coyote_accelerator/host_libs.hpp')
        dstpath = f'{model.config.get_output_dir()}/src/host_libs.hpp'
        copyfile(srcpath, dstpath)

        # host_libs.cpp
        srcpath = os.path.join(filedir, '../templates/coyote_accelerator/host_libs.cpp')
        dstpath = f'{model.config.get_output_dir()}/src/host_libs.cpp'
        copyfile(srcpath, dstpath)

    def __make_dat_file(self, original_path, project_path):
        """
        Convert other input/output data types into a dat file, which is
        a text file with the falttened matrix printed out. Note that ' ' is
        assumed to be the delimiter.

        TODO: These seemed to be shared between many hls4ml writers; perhaps
        these should be moved to some utility class
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

    def write_test_bench(self, model):
        """
        Generates the HLS testbench; very similar to the testbench in Vivado/Vitis backends
        For differences, refer to the myproject_test.cpp file.

        Args:
            model (ModelGraph): the hls4ml model
        """
        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists(f'{model.config.get_output_dir()}/tb_data/'):
            os.mkdir(f'{model.config.get_output_dir()}/tb_data/')

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == 'dat':
                copyfile(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')
            else:
                self.__make_dat_file(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')

        if output_predictions:
            if output_predictions[-3:] == 'dat':
                copyfile(output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat')
            else:
                self.__make_dat_file(
                    output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat'
                )
        
        f = open(os.path.join(filedir, '../templates/coyote_accelerator/myproject_test.cpp'))
        fout = open(f'{model.config.get_output_dir()}/src/{model.config.get_project_name()}_test.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        if len(model_inputs) > 1 or len(model_outputs) > 1:
            raise RuntimeError('CoyoteAccelerator backend currently only supports one input and one output')

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert data' in line:
                newline = line
                offset = 0
                for inp in model_inputs:
                    newline += indent + f'float {inp.name}[{inp.size_cpp()}];\n'
                    newline += indent + f'nnet::copy_data<float, float, {offset}, {inp.size_cpp()}>(in, {inp.name});\n'
                    newline += indent + 'hls::stream<axi_s> data_in;\n'
                    newline += indent + f'nnet::data_to_axi_stream<float, float, {inp.size_cpp()}, COYOTE_AXI_STREAM_BITS, 8 * sizeof(float)>({inp.name}, data_in);\n'
                    offset += inp.size()
                for out in model_outputs:
                    newline += indent + f'float {out.name}[{out.size_cpp()}];\n'
                    newline += indent + 'hls::stream<axi_s> data_out;\n'

            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inp in model_inputs:
                    newline += indent + f'float {inp.name}[{inp.size_cpp()}];\n'
                    newline += indent + f'nnet::fill_zero<float, {inp.size_cpp()}>({inp.name});\n'
                    newline += indent + 'hls::stream<axi_s> data_in;\n'
                    newline += indent + f'nnet::data_to_axi_stream<float, float, {inp.size_cpp()}, COYOTE_AXI_STREAM_BITS, 8 * sizeof(float)>({inp.name}, data_in);\n'

                for out in model_outputs:
                    newline += indent + f'float {out.name}[{out.size_cpp()}];\n'
                    newline += indent + 'hls::stream<axi_s> data_out;\n'

            elif '// hls-fpga-machine-learning insert top-level-function' in line:
                newline = line
                newline += indent + 'model_wrapper(data_in, data_out);\n'
                newline += indent + f'nnet::axi_stream_to_data<float, float, {out.size_cpp()}, COYOTE_AXI_STREAM_BITS, 8 * sizeof(float)>(data_out, {out.name});\n'

            elif '// hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for out in model_outputs:
                    newline += indent + f'for(int i = 0; i < {out.size_cpp()}; i++) {{\n'
                    newline += indent + '  std::cout << pr[i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'

            elif '// hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                for out in model_outputs:
                    newline += indent + f'nnet::print_result<float, {out.size_cpp()}>({out.name}, fout);\n'

            elif (
                '// hls-fpga-machine-learning insert output' in line
                or '// hls-fpga-machine-learning insert quantized' in line
            ):
                newline = line
                for out in model_outputs:
                    newline += indent + f'nnet::print_result<float, {out.size_cpp()}>({out.name}, std::cout, true);\n'

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_hls(self, model):    
        """
        Write the HLS project. Most of the functionality inherited from VitisWriter;
        some additional functionality added for Coyote specifically.

        Args:
            model (ModelGraph): the hls4ml model
        """
        # General hls4ml write proces, inherited from Vitis Writer
        self.write_project_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        self.write_weights(model)
        self.write_defines(model)
        self.write_parameters(model)
        self.write_bridge(model)
        self.write_nnet_utils(model)
        self.write_nnet_utils_overrides(model)
        self.write_generated_code(model)
        
        # Coyote-specific writes, implemented in this file
        self.write_coyote(model)
        self.write_model_wrapper(model)
        self.write_host_code(model)
        self.write_test_bench(model)
        self.write_build_script(model)
        self.restructure_dir(model)
        self.write_yml(model)
        
        print('Done')
