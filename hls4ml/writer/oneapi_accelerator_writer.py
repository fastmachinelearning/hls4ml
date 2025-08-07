import os
from shutil import copyfile

from hls4ml.utils.string_utils import convert_to_pascal_case
from hls4ml.writer.oneapi_writer import OneAPIWriter

config_filename = 'hls4ml_config.yml'


class OneAPIAcceleratorWriter(OneAPIWriter):

    def write_project_cpp(self, model):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        project_name = model.config.get_project_name()

        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/oneapi/firmware/myproject.cpp')) as f,
            open(f'{model.config.get_output_dir()}/src/firmware/{project_name}.cpp', 'w') as fout,
        ):
            model_inputs = model.get_input_variables()
            model_outputs = model.get_output_variables()
            model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

            if len(model_brams) != 0:
                raise NotImplementedError("Weights on the interface is currently not supported")

            io_type = model.config.get_config_value('IOType')
            indent = '    '

            for line in f.readlines():
                # Add headers to weights and biases
                if 'myproject' in line:
                    newline = line.replace('myproject', project_name)
                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                # oneAPI pipes need to be declared and passed as template parameters
                elif '// hls-fpga-machine-learning insert inter-task pipes' in line:
                    newline = line
                    if io_type == 'io_stream':
                        for layer in model.get_layers():
                            vars = layer.get_variables()
                            for var in vars:
                                if var not in model_inputs and var not in model_outputs:
                                    newline += var.declare_cpp()

                # Read in inputs
                elif '// hls-fpga-machine-learning read in' in line:
                    newline = line
                    if io_type == 'io_parallel':
                        restartable_kernel_loop = f"bool keep_going = true;\n\n" f"{indent}while (keep_going) {{\n"
                        newline += indent + restartable_kernel_loop
                        for inp in model_inputs:
                            newline += indent * 2 + f'auto {inp.name}_beat = {inp.pipe_name}::read();\n'
                            newline += indent * 2 + f'auto {inp.name} = {inp.name}_beat.data;\n'
                    # for streaming we don't need to read it in

                # Insert weights
                elif '// hls-fpga-machine-learning insert weights' in line:
                    newline = line
                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            if w not in model_brams:
                                newline += f'#include "weights/{w.name}.h"\n'

                # Insert task sequences
                elif '// hls-fpga-machine-learning declare task sequences' in line:
                    if io_type == 'io_stream':  # only need this for io_stream
                        newline = line
                        for layer in model.get_layers():
                            ts = layer.get_attr('tast_sequence_cpp')
                            if ts:
                                newline += '    ' + ts + '\n'
                    else:
                        newline = indent + line

                # Neural net instantiation
                elif '// hls-fpga-machine-learning insert layers' in line:
                    if io_type == 'io_parallel':
                        newline = indent + line + '\n'
                    else:
                        newline = line + '\n'
                    for layer in model.get_layers():
                        if io_type != 'io_stream':
                            vars = layer.get_variables()
                            for var in vars:
                                if var not in model_inputs:
                                    def_cpp = var.definition_cpp()
                                    if def_cpp is not None:
                                        newline += indent * 2 + def_cpp + ';\n'
                        func = (
                            layer.get_attr('function_cpp')
                            if io_type == 'io_parallel'
                            else layer.get_attr('stream_function_cpp')
                        )
                        if func:
                            newline += (indent * 2 if io_type == 'io_parallel' else indent) + func + '\n'
                            if model.config.trace_output and layer.get_attr('trace', False):
                                newline += '#ifndef HLS_SYNTHESIS\n'
                                for var in vars:
                                    newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                        var.type.name, var.name, layer.name, var.size_cpp()
                                    )
                                newline += '#endif\n'

                # Write the output
                elif '// hls-fpga-machine-learning return' in line:
                    newline = line
                    if io_type == 'io_parallel':
                        newline = indent + newline
                        for out in model_outputs:
                            out_beat = f"{out.name}_beat"
                            newline += (
                                indent * 2 + f'typename nnet::ExtractPipeType<{out.pipe_name}>::value_type {out_beat};\n'
                            )
                            newline += indent * 2 + f'{out_beat}.data = {out.name};\n'
                            newline += indent * 2 + f'{out.pipe_name}::write({out_beat});\n'
                        newline += indent * 2 + '// stops the kernel when the last input seen.\n'
                        newline += indent * 2 + f'keep_going = !{model_inputs[0].name}_beat.eop;\n'
                        newline += f"{indent}}}\n"
                    # don't need to add anything in io_stream

                # Just copy line
                else:
                    newline = line

                fout.write(newline)

    def write_project_header(self, model):
        """Write the main architecture header file (myproject.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        project_name = model.config.get_project_name()

        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/oneapi_accelerator/firmware/myproject.h')) as f,
            open(f'{model.config.get_output_dir()}/src/firmware/{project_name}.h', 'w') as fout,
        ):
            model_inputs = model.get_input_variables()
            model_outputs = model.get_output_variables()
            # model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

            # io_parallel and io_stream instantiate the top-level function differently (io_stream not yet supported)
            # io_type = model.config.get_config_value('IOType')
            # indent = '    '
            # brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

            for line in f.readlines():
                if 'MYPROJECT' in line:
                    newline = line.replace('MYPROJECT', format(project_name.upper()))

                elif 'myproject' in line:
                    newline = line.replace('myproject', project_name)

                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                # Declarations for the inputs. May need modification when io_stream is supported
                elif '// hls-fpga-machine-learning insert inputs' in line:
                    newline = line
                    for inp in model_inputs:
                        newline += inp.declare_cpp()

                # and declareations for the outputs
                elif '// hls-fpga-machine-learning insert outputs' in line:
                    newline = line
                    for out in model_outputs:
                        newline += out.declare_cpp()

                # Simply copy line, if no inserts are required
                else:
                    newline = line

                fout.write(newline)

    def write_test_bench(self, model):
        """Write the testbench

        Args:
            model (ModelGraph): the hls4ml model.
        """
        # TODO - This function only works with one model input
        # (NOT one data point - it works as expected with multiple data points)

        # copy the exception handler
        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath = os.path.join(filedir, '../templates/oneapi/exception_handler.hpp')
        dstpath = f'{model.config.get_output_dir()}/src/exception_handler.hpp'
        copyfile(srcpath, dstpath)

        project_name = model.config.get_project_name()
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        if len(model_brams) != 0:
            raise NotImplementedError("Weights on the interface is currently not supported")

        if len(model_inputs) != 1 or len(model_outputs) != 1:
            print("The testbench supports only single input arrays and single output arrays.")
            print("Please modify it before using it.")

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

        with (
            open(os.path.join(filedir, '../templates/oneapi_accelerator/myproject_test.cpp')) as f,
            open(f'{model.config.get_output_dir()}/src/{project_name}_test.cpp', 'w') as fout,
        ):
            for line in f.readlines():
                indent = ' ' * (len(line) - len(line.lstrip(' ')))

                if 'myproject' in line:
                    newline = line.replace('myproject', project_name)
                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                elif '// hls-fpga-machine-learning insert bram' in line:
                    newline = line
                    for bram in model_brams:
                        newline += f'#include \"firmware/weights/{bram.name}.h\"\n'
                elif '// hls-fpga-machine-learning insert runtime contant' in line:
                    newline = line
                    insert_constant_lines = (
                        f'{indent}const size_t kInputSz = {model_inputs[0].size_cpp()} * num_iterations;\n'
                        f'{indent}const size_t kOutputSz = {model_outputs[0].size_cpp()} * num_iterations;\n'
                        f'{indent}const size_t kInputLayerSize = {model_inputs[0].size_cpp()};\n'
                        f'{indent}const size_t kOutLayerSize = {model_outputs[0].size_cpp()};\n'
                    )
                    newline += insert_constant_lines
                elif '// hls-fpga-machine-learning insert zero' in line:
                    newline = line
                    inp = model_inputs[0]
                    insert_zero_lines = (
                        f'{indent}for (int j = 0 ; j < kInputSz; j++)\n'
                        f'{indent}    vals[j] = 0.0;\n'
                        f'{indent}q.single_task(nnet::DMA_convert_data<float, {inp.pipe_name}>{{vals, num_iterations}});\n'
                    )
                    newline += insert_zero_lines
                elif '// hls-fpga-machine-learning insert data' in line:
                    newline = line
                    inp = model_inputs[0]
                    insert_data_lines = (
                        f'{indent}for (int i = 0; i < num_iterations; i++)\n'
                        f'{indent}    for (int j = 0 ; j < kInputLayerSize; j++)\n'
                        f'{indent}        vals[i * kInputLayerSize + j] = inputs[i][j]; \n'
                        f'{indent}q.single_task(nnet::DMA_convert_data<float, {inp.pipe_name}>{{vals, num_iterations}});\n'
                    )
                    newline += insert_data_lines
                elif '// hls-fpga-machine-learning convert output' in line:
                    newline = line
                    out = model_outputs[0]
                    newline += f'{indent}q.single_task(nnet::DMA_convert_data_back<{out.pipe_name}, float>'
                    newline += '{outputs, num_iterations}).wait();\n'
                else:
                    newline = line

                fout.write(newline)

    def write_build_script(self, model):
        """Write the build scripts (Makefile, build_lib.sh)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # Makefile
        filedir = os.path.dirname(os.path.abspath(__file__))
        device = model.config.get_config_value('Part')
        period = model.config.get_config_value('ClockPeriod')
        hyper = model.config.get_config_value('HyperoptHandshake')
        with (
            open(os.path.join(filedir, '../templates/oneapi/CMakeLists.txt')) as f,
            open(f'{model.config.get_output_dir()}/CMakeLists.txt', 'w') as fout,
        ):
            for line in f.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

                if 'set(FPGA_DEVICE' in line:
                    line = f'    set(FPGA_DEVICE "{device}")\n'

                if model.config.get_config_value('UseOneAPIBSP'):
                    if 'hls-fpga-machine-learning insert oneapi_bsp_cmake_flag' in line:
                        line = 'set(BSP_FLAG "-DIS_BSP")'

                if 'set(USER_FPGA_FLAGS' in line:
                    line += f'set(USER_FPGA_FLAGS -Xsclock={period}ns; ${{USER_FPGA_FLAGS}})\n'
                    if not hyper:
                        line += 'set(USER_FPGA_FLAGS -Xsoptimize=latency; ${USER_FPGA_FLAGS})\n'

                fout.write(line)
