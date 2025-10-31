import glob
import stat
import tarfile
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile

import numpy as np
import yaml

from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class LiberoWriter(Writer):
    def print_array_to_cpp(self, var, odir, namespace=None, write_txt_file=True):
        """Write a weights array to C++ header files.

        Args:
            var (WeightVariable): Weight to write
            odir (str): Output directory
            namespace (str, optional): Writes a namespace for the weights to avoid clashes with global variables.
            write_txt_file (bool, optional): Write txt files in addition to .h files. Defaults to True.
        """

        h_file = open(f'{odir}/firmware/weights/{var.name}.h', 'w')
        if write_txt_file:
            txt_file = open(f'{odir}/firmware/weights/{var.name}.txt', 'w')

        # meta data
        h_file.write(f'//Numpy array shape {var.shape}\n')
        h_file.write(f'//Min {np.min(var.min):.12f}\n')
        h_file.write(f'//Max {np.max(var.max):.12f}\n')
        h_file.write(f'//Number of zeros {var.nzeros}\n')
        h_file.write('\n')

        h_file.write(f'#ifndef {var.name.upper()}_H_\n')
        h_file.write(f'#define {var.name.upper()}_H_\n')
        h_file.write('\n')

        if namespace is not None:
            h_file.write(f'namespace {namespace} {{\n\n')

        if write_txt_file:
            h_file.write('#ifndef __SYNTHESIS__\n')
            h_file.write(var.definition_cpp() + ';\n')
            h_file.write('#else\n')

        h_file.write(var.definition_cpp() + ' = {')

        # fill c++ array.
        # not including internal brackets for multidimensional case
        sep = ''
        for x in var:
            h_file.write(sep + x)
            if write_txt_file:
                txt_file.write(sep + x)
            sep = ', '
        h_file.write('};\n\n')

        if write_txt_file:
            h_file.write('#endif\n')
            txt_file.close()

        if namespace is not None:
            h_file.write('}\n\n')

        h_file.write('\n#endif\n')
        h_file.close()

    def write_project_dir(self, model):
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        out_path = Path(f'{model.config.get_output_dir()}/firmware/weights')
        out_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_array_pragma(variable, is_argument=False):
        """
        Layers in ModelGraph can specify output array partitioning through the `pragma` attribute.
        If `pragma` is a string: options are 'partition' or 'stream'.
        If `pragma` is a tuple: (mode, type, factor) where mode is 'partition', type is
        'complete', 'cyclic', or 'block', and factor is an integer only used when the type is not 'complete'.
        """

        config = variable.pragma
        if type(config) is tuple:
            mode = config[0]
            if mode == 'partition':
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        arg_name = 'argument' if is_argument else 'variable'

        if mode == 'partition':
            if typ == 'complete':
                template = '#pragma HLS memory partition {arg_name}({name}) type({type}) dim({dim})'
            else:
                template = '#pragma HLS memory partition {arg_name}({name}) type({type}) factor({factor}) dim({dim})'

            return template.format(mode=mode.upper(), name=variable.name, type=typ, factor=factor, dim=0, arg_name=arg_name)

        elif mode == 'stream':
            # TODO update for streaming IO
            return f'#pragma HLS STREAM {arg_name}={variable.name} depth={depth}'

    def write_project_cpp(self, model):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = Path(__file__).parent
        prj_name = model.config.get_project_name()
        prj_cpp_src = (filedir / '../templates/libero/firmware/myproject.cpp').resolve()
        prj_cpp_dst = Path(f'{model.config.get_output_dir()}/firmware/{prj_name}.cpp').resolve()

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']
        prj_name = prj_name

        indent = '    '

        with open(prj_cpp_src) as src, open(prj_cpp_dst, 'w') as dst:
            for line in src.readlines():
                # Add headers to weights and biases
                if 'myproject' in line:
                    newline = line.replace('myproject', prj_name)

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

                        newline += '        loaded_weights = true;\n'
                        newline += '    }\n'
                        newline += '#endif'

                # Add input/output type
                elif '// hls-fpga-machine-learning insert IO' in line:
                    newline = ''
                    all_inputs = [i.name for i in model_inputs]
                    all_outputs = [o.name for o in model_outputs]
                    all_brams = [b.name for b in model_brams]
                    io_type = model.config.get_config_value('IOType')

                    pipeline_style = model.config.pipeline_style
                    pipeline_ii = model.config.pipeline_ii
                    pipeline_pragma = indent + f'#pragma HLS function {pipeline_style}'
                    if pipeline_style == 'pipeline' and pipeline_ii is not None:
                        pipeline_pragma += f' II({pipeline_ii})\n'
                    else:
                        pipeline_pragma += '\n'

                    if io_type == 'io_parallel':
                        # TODO Expose interface in a backend config
                        newline += indent + '#pragma HLS interface control type(simple)\n'
                        for input_name in all_inputs:
                            newline += indent + f'#pragma HLS interface argument({input_name}_fifo) type(simple)\n'
                        for output_name in all_outputs:
                            newline += indent + f'#pragma HLS interface argument({output_name}_fifo) type(simple)\n'
                        newline += pipeline_pragma

                    if io_type == 'io_stream':
                        newline += indent + '#pragma HLS interface control type(axi_target)\n'
                        newline += indent + '#pragma HLS interface default type(axi_target)'
                        for bram_name in all_brams:
                            newline += indent + f'#pragma HLS interface argument({bram_name}) dma(true)\n'
                        newline += pipeline_pragma

                elif '// hls-fpga-machine-learning read input' in line:
                    newline = ''
                    for i in model_inputs:
                        if i.pragma:
                            newline += '    ' + self._make_array_pragma(i, is_argument=False) + '\n'
                        tmp_struct_var_name = f'{i.name}_struct'
                        newline += f'    {i.type.name} {i.name}[{i.size_cpp()}];\n'
                        newline += f'    {i.struct_name} {tmp_struct_var_name} = {i.name}_fifo.read();\n'
                        newline += f'    for (unsigned i = 0; i < {i.size_cpp()}; i++) {{\n'
                        newline += f'        {i.name}[i] = {tmp_struct_var_name}.data[i];\n'
                        newline += '    }\n'

                elif '// hls-fpga-machine-learning write output' in line:
                    newline = ''
                    for o in model_outputs:
                        tmp_struct_var_name = f'{o.name}_struct'
                        newline += f'    {o.struct_name} {tmp_struct_var_name};\n'
                        newline += f'    for (unsigned i = 0; i < {i.size_cpp()}; i++) {{\n'
                        newline += f'        {tmp_struct_var_name}.data[i] = {o.name}[i];\n'
                        newline += '    }\n'
                        newline += f'    {o.name}_fifo.write({tmp_struct_var_name});\n'

                elif '// hls-fpga-machine-learning insert layers' in line:
                    newline = line + '\n'
                    for layer in model.get_layers():
                        vars = layer.get_variables()
                        for var in vars:
                            if var not in model_inputs:
                                def_cpp = var.definition_cpp()
                                if def_cpp is not None:
                                    if var.pragma:
                                        newline += '    ' + self._make_array_pragma(var) + '\n'
                                    newline += '    ' + def_cpp + ';\n'
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
                dst.write(newline)

    def write_project_header(self, model):
        """Write the main architecture header file (myproject.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = Path(__file__).parent
        prj_name = model.config.get_project_name()
        prj_h_src = (filedir / '../templates/libero/firmware/myproject.h').resolve()
        prj_h_dst = Path(f'{model.config.get_output_dir()}/firmware/{prj_name}.h').resolve()

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        with open(prj_h_src) as src, open(prj_h_dst, 'w') as dst:
            for line in src.readlines():
                if 'MYPROJECT' in line:
                    newline = line.replace('MYPROJECT', format(prj_name.upper()))

                elif 'myproject' in line:
                    newline = line.replace('myproject', prj_name)

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

                else:
                    newline = line
                dst.write(newline)

    def write_defines(self, model):
        """Write the C++ type definitions file (defines.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = Path(__file__).parent
        defines_src = (filedir / '../templates/libero/firmware/defines.h').resolve()
        defines_dst = Path(f'{model.config.get_output_dir()}/firmware/defines.h').resolve()

        with open(defines_src) as src, open(defines_dst, 'w') as dst:
            for line in src.readlines():
                if '// hls-fpga-machine-learning insert layer-precision' in line:
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

                elif '// hls-fpga-machine-learning insert struct-definitions' in line:
                    newline = line

                    model_inputs = model.get_input_variables()
                    model_outputs = model.get_output_variables()

                    newline += '\n'.join([var.definition_cpp(as_struct=True) for var in model_inputs + model_outputs])

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

                else:
                    newline = line
                dst.write(newline)

    def write_parameters(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = Path(__file__).parent
        params_src = (filedir / '../templates/libero/firmware/parameters.h').resolve()
        params_dst = Path(f'{model.config.get_output_dir()}/firmware/parameters.h').resolve()

        with open(params_src) as src, open(params_dst, 'w') as dst:
            for line in src.readlines():
                if '// hls-fpga-machine-learning insert includes' in line:
                    newline = line
                    for include in sorted(
                        set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))
                    ):
                        newline += '#include "%s"\n' % include

                elif '// hls-fpga-machine-learning insert weights' in line:
                    newline = line
                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            if w.storage.lower() != 'bram':
                                newline += f'#include "weights/{w.name}.h"\n'

                elif "// hls-fpga-machine-learning insert layer-config" in line:
                    newline = line
                    for layer in model.get_layers():
                        config = layer.get_attr('config_cpp', None)
                        if config:
                            newline += '// ' + layer.name + '\n'
                            newline += config + '\n'

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

                else:
                    newline = line
                dst.write(newline)

    def write_weights(self, model):
        """Write the weights into header files

        Args:
            model (ModelGraph): the hls4ml model.
        """
        namespace = model.config.get_writer_config().get('Namespace', None)
        write_txt = model.config.get_writer_config().get('WriteWeightsTxt', True)
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(
                    weights, model.config.get_output_dir(), namespace=namespace, write_txt_file=write_txt
                )

    def __make_dat_file(self, original_path, project_path):
        """
        Convert other input/output data types into a dat file, which is
        a text file with the flattened matrix printed out. Note that ' ' is
        assumed to be the delimiter.
        """

        # Take in data from current supported data files
        if original_path[-3:] == "npy":
            data = np.load(original_path)
        else:
            raise Exception("Unsupported input/output data files.")

        # Flatten data, just keep first dimension
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
        """Write the testbench files (myproject_test.cpp and input/output .dat files)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = Path(__file__).parent
        prj_name = model.config.get_project_name()
        out_dir = model.config.get_output_dir()

        tb_data_dir = Path(f'{out_dir}/tb_data/').resolve()
        tb_data_dir.mkdir(parents=True, exist_ok=True)

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == 'dat':
                copyfile(input_data, f'{out_dir}/tb_data/tb_input_features.dat')
            else:
                self.__make_dat_file(input_data, f'{out_dir}/tb_data/tb_input_features.dat')

        if output_predictions:
            if output_predictions[-3:] == 'dat':
                copyfile(output_predictions, f'{out_dir}/tb_data/tb_output_predictions.dat')
            else:
                self.__make_dat_file(output_predictions, f'{out_dir}/tb_data/tb_output_predictions.dat')

        tb_src = (filedir / '../templates/libero/myproject_test.cpp').resolve()
        tb_dst = Path(f'{out_dir}/{prj_name}_test.cpp').resolve()

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        with open(tb_src) as src, open(tb_dst, 'w') as dst:
            for line in src.readlines():
                indent = ' ' * (len(line) - len(line.lstrip(' ')))

                # Insert numbers
                if 'myproject' in line:
                    newline = line.replace('myproject', model.config.get_project_name())

                elif '// hls-fpga-machine-learning insert bram' in line:
                    newline = line
                    for bram in model_brams:
                        newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

                elif '// hls-fpga-machine-learning insert data' in line:
                    newline = line
                    offset = 0
                    for inp in model_inputs:
                        newline += indent + inp.definition_cpp() + ';\n'
                        newline += indent + 'nnet::copy_data<float, {}, {}, {}>(in, {});\n'.format(
                            inp.type.name, offset, inp.size_cpp(), inp.name
                        )
                        offset += inp.size()

                elif '// hls-fpga-machine-learning insert zero' in line:
                    newline = line
                    for inp in model_inputs:
                        newline += indent + inp.definition_cpp() + ';\n'
                        newline += indent + f'nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                    for out in model_outputs:
                        newline += indent + out.definition_cpp() + ';\n'

                elif '// hls-fpga-machine-learning insert top-level-function' in line:
                    newline = line

                    input_vars = ','.join([i.name + '_fifo' for i in model_inputs])
                    output_vars = ','.join([o.name + '_fifo' for o in model_outputs])
                    bram_vars = ','.join([b.name for b in model_brams])

                    # Concatenate the input, output, and bram variables. Filter out empty/null values
                    all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                    top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'

                    newline += top_level

                elif '// hls-fpga-machine-learning pack-struct' in line:
                    newline = line
                    for inp in model_inputs:
                        tmp_struct_var_name = f'{inp.name}_struct'
                        newline += indent + f'{inp.struct_name} {tmp_struct_var_name};\n'
                        newline += indent + f'for (unsigned i = 0; i < {inp.size_cpp()}; i++) {{\n'
                        newline += indent + f'    {tmp_struct_var_name}.data[i] = {inp.name}[i];\n'
                        newline += indent + '}\n'
                        newline += indent + f'{inp.name}_fifo.write({tmp_struct_var_name});\n'

                elif '// hls-fpga-machine-learning unpack-struct' in line:
                    newline = line
                    for out in model_outputs:
                        tmp_struct_var_name = f'{out.name}_struct'
                        newline += indent + f'{out.struct_name} {tmp_struct_var_name} = {out.name}_fifo.read();\n'
                        newline += indent + out.definition_cpp() + ';\n'
                        newline += indent + f'for (unsigned i = 0; i < {out.size_cpp()}; i++) {{\n'
                        newline += indent + f'    {out.name}[i] = {tmp_struct_var_name}.data[i];\n'
                        newline += indent + '}\n'

                elif '// hls-fpga-machine-learning fifo-definitions' in line:
                    newline = line
                    for inp in model_inputs:
                        newline += indent + f'hls::FIFO<{inp.struct_name}> {inp.name}_fifo(DEFAULT_FIFO_DEPTH);\n'
                    for out in model_outputs:
                        newline += indent + f'hls::FIFO<{out.struct_name}> {out.name}_fifo(DEFAULT_FIFO_DEPTH);\n'

                elif '// hls-fpga-machine-learning zero-fifo-definitions' in line:
                    newline = line
                    for inp in model_inputs:
                        newline += indent + f'hls::FIFO<{inp.struct_name}> {inp.name}_fifo(NUM_TEST_SAMPLES);\n'
                    for out in model_outputs:
                        newline += indent + f'hls::FIFO<{out.struct_name}> {out.name}_fifo(NUM_TEST_SAMPLES);\n'

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
                        newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
                            out.type.name, out.size_cpp(), out.name
                        )  # TODO enable this

                elif (
                    '// hls-fpga-machine-learning insert output' in line
                    or '// hls-fpga-machine-learning insert quantized' in line
                ):
                    newline = line
                    for out in model_outputs:
                        newline += indent + 'nnet::print_result<{}, {}>({}, std::cout, true);\n'.format(
                            out.type.name, out.size_cpp(), out.name
                        )

                elif '// hls-fpga-machine-learning insert namespace' in line:
                    newline = ''

                    namespace = model.config.get_writer_config().get('Namespace', None)
                    if namespace is not None:
                        newline += indent + f'using namespace {namespace};\n'

                else:
                    newline = line
                dst.write(newline)

    def write_bridge(self, model):
        """Write the Python-C++ bridge (myproject_bridge.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = Path(__file__).parent
        prj_name = model.config.get_project_name()
        bridge_src = (filedir / '../templates/libero/myproject_bridge.cpp').resolve()
        bridge_dst = Path(f'{model.config.get_output_dir()}/{prj_name}_bridge.cpp').resolve()

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        with open(bridge_src) as src, open(bridge_dst, 'w') as dst:
            for line in src.readlines():
                if 'MYPROJECT' in line:
                    newline = line.replace('MYPROJECT', prj_name.upper())

                elif 'myproject' in line:
                    newline = line.replace('myproject', prj_name)

                elif '// hls-fpga-machine-learning insert bram' in line:
                    newline = line
                    for bram in model_brams:
                        newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

                elif '// hls-fpga-machine-learning insert header' in line:
                    dtype = line.split('#', 1)[1].strip()
                    inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                    outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])

                    newline = ''
                    newline += indent + inputs_str + ',\n'
                    newline += indent + outputs_str + '\n'

                elif '// hls-fpga-machine-learning pack-struct' in line:
                    newline = ''
                    for inp in model_inputs:
                        tmp_struct_var_name = f'{inp.name}_struct'
                        newline += indent + f'{inp.struct_name} {tmp_struct_var_name};\n'
                        newline += indent + f'for (unsigned i = 0; i < {inp.size_cpp()}; i++) {{\n'
                        newline += indent + f'    {tmp_struct_var_name}.data[i] = {inp.name}[i];\n'
                        newline += indent + '}\n'
                        newline += indent + f'{inp.name}_fifo.write({tmp_struct_var_name});\n'

                elif '// hls-fpga-machine-learning unpack-struct' in line:
                    newline = ''
                    for out in model_outputs:
                        tmp_struct_var_name = f'{out.name}_struct'
                        newline += indent + f'{out.struct_name} {tmp_struct_var_name} = {out.name}_fifo.read();\n'
                        newline += indent + f'for (unsigned i = 0; i < {out.size_cpp()}; i++) {{\n'
                        newline += indent + f'    {out.name}[i] = {tmp_struct_var_name}.data[i];\n'
                        newline += indent + '}\n'

                elif '// hls-fpga-machine-learning fifo-definitions' in line:
                    newline = ''
                    for inp in model_inputs:
                        newline += indent + f'hls::FIFO<{inp.struct_name}> {inp.name}_fifo(DEFAULT_FIFO_DEPTH);\n'
                    for out in model_outputs:
                        newline += indent + f'hls::FIFO<{out.struct_name}> {out.name}_fifo(DEFAULT_FIFO_DEPTH);\n'

                elif '// hls-fpga-machine-learning insert wrapper' in line:
                    dtype = line.split('#', 1)[1].strip()
                    newline = ''

                    input_vars = ','.join([i.name + '_fifo' for i in model_inputs])
                    bram_vars = ','.join([b.name for b in model_brams])
                    output_vars = ','.join([o.name + '_fifo' for o in model_outputs])

                    # Concatenate the input, output, and bram variables. Filter out empty/null values
                    all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                    top_level = indent + f'{prj_name}({all_vars});\n'
                    newline += top_level

                elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                    newline = ''
                    for layer in model.get_layers():
                        func = layer.get_attr('function_cpp', None)
                        if func and model.config.trace_output and layer.get_attr('trace', False):
                            vars = layer.get_variables()
                            for var in vars:
                                newline += (
                                    indent
                                    + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                    + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                                )

                elif '// hls-fpga-machine-learning insert namespace' in line:
                    newline = ''

                    namespace = model.config.get_writer_config().get('Namespace', None)
                    if namespace is not None:
                        newline += indent + f'using namespace {namespace};\n'

                else:
                    newline = line
                dst.write(newline)

    def write_build_script(self, model):
        """Write the TCL/Shell build scripts (config.tcl, Makefile, build_lib.sh)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = Path(__file__).parent
        prj_name = model.config.get_project_name()

        # project.tcl
        cfg_tcl_dst = Path(f'{model.config.get_output_dir()}/config.tcl')
        with open(cfg_tcl_dst, 'w') as f:
            f.write('source $env(SHLS_ROOT_DIR)/examples/legup.tcl\n')
            fpga_family = model.config.get_config_value('FPGAFamily')
            fpga_part = model.config.get_config_value('Part')
            board = model.config.get_config_value('Board')
            clock = model.config.get_config_value('ClockPeriod')
            f.write(f'set_project {fpga_family} {fpga_part} {board}\n')
            f.write(f'set_parameter CLOCK_PERIOD {clock}\n')

        # Makefile
        makefile_dst = Path(f'{model.config.get_output_dir()}/Makefile')
        with open(makefile_dst, 'w') as f:
            f.write(f'NAME = {prj_name}\n')
            f.write('LOCAL_CONFIG = -legup-config=config.tcl\n')
            f.write(f'SRCS = firmware/{prj_name}.cpp {prj_name}_test.cpp \n')
            # Not sure if this is required, it is present in both GUI- and CLI-generated projects
            f.write('LEVEL = $(SHLS_ROOT_DIR)/examples\n')
            # This must be the last line
            f.write('include $(LEVEL)/Makefile.common\n')

        # Makefile.compile
        makefile_dst = Path(f'{model.config.get_output_dir()}/Makefile.compile')
        with open(makefile_dst, 'w') as f:
            f.write(f'NAME = {prj_name}\n')
            f.write('LOCAL_CONFIG = -legup-config=config.tcl\n')
            f.write(f'SRCS = firmware/{prj_name}.cpp {prj_name}_bridge.cpp \n')
            f.write('LEVEL = $(SHLS_ROOT_DIR)/examples\n')
            f.write('USER_CXX_FLAG = -fPIC\n')
            f.write('USER_LINK_FLAG = -shared\n')
            f.write('include $(LEVEL)/Makefile.common\n')

        # build_lib.sh
        build_lib_src = (filedir / '../templates/libero/build_lib.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', prj_name)
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

                dst.write(line)
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = Path(__file__).parent
        out_dir = model.config.get_output_dir()

        srcpath = (filedir / '../templates/libero/nnet_utils/').resolve()
        dstpath = Path(f'{out_dir}/firmware/nnet_utils/').resolve()
        dstpath.mkdir(parents=True, exist_ok=True)

        headers = [Path(h).name for h in glob.glob(str(srcpath / '*.h'))]

        for h in headers:
            copyfile(srcpath / h, dstpath / h)

        # custom source
        custom_source = model.config.backend.get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = Path(f'{out_dir}/firmware/{dst}')
            copyfile(srcpath, dstpath)

    def write_generated_code(self, model):
        """Write the generated code (nnet_code_gen.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        codegen_path = Path(f'{model.config.get_output_dir()}/firmware/nnet_utils/nnet_code_gen.h')
        with open(codegen_path) as src:
            contents = src.readlines()
        with open(codegen_path, 'w') as dst:
            namespace = model.config.get_writer_config().get('Namespace', None)

            for line in contents:
                if '// hls4ml insert code' in line:
                    newline = line
                    for layer in model.get_layers():
                        for generated_code in layer.code.values():
                            newline += str(generated_code)
                else:
                    newline = line
                if namespace is not None:
                    if 'namespace nnet' in newline:
                        newline = newline.replace('namespace nnet', f'namespace {namespace}')
                dst.write(newline)

    def write_yml(self, model):
        """Write the config to the YAML file

        Args:
            model (ModelGraph): the hls4ml model.
        """

        def keras_model_representer(dumper, keras_model):
            model_path = model.config.get_output_dir() + '/keras_model.keras'
            keras_model.save(model_path)
            return dumper.represent_scalar('!keras_model', model_path)

        try:
            import keras

            KerasModel = keras.models.Model

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

        write_tar = model.config.get_writer_config().get('WriteTar', False)
        if write_tar:
            tar_path = Path(model.config.get_output_dir() + '.tar.gz')
            tar_path.unlink(missing_ok=True)
            with tarfile.open(tar_path, mode='w:gz') as archive:
                archive.add(model.config.get_output_dir(), recursive=True, arcname='')

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
        self.write_generated_code(model)
        self.write_yml(model)
        self.write_tar(model)
        print('Done')
