import copy
import glob
import math
import os
import shutil
import tarfile
from collections import OrderedDict
from shutil import copyfile

import numpy as np
import yaml

from hls4ml.backends import get_backend
from hls4ml.utils.fixed_point_utils import FixedPointEmulator, ceil_log2, uint_to_binary
from hls4ml.utils.string_utils import convert_to_pascal_case
from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class AlteraWriter(Writer):
    def __make_dat_file(self, original_path, project_path):
        """
        Convert other input/output data types into a dat file, which is
        a text file with the falttened matrix printed out. Note that ' ' is
        assumed to be the delimiter.
        """

        # Take in data from current supported data files
        if original_path[-3:] == 'npy':
            data = np.load(original_path)
        else:
            raise Exception('Unsupported input/output data files.')

        # Faltten data, just keep first dimension
        data = data.reshape(data.shape[0], -1)

        def print_data(f):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(data[i][j]) + ' ')
                f.write('\n')

        # Print out in dat file
        with open(project_path, 'w') as f:
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
        with open(f'{odir}/src/firmware/weights/{var.name}.h', 'w') as h_file:
            # meta data
            h_file.write(f'//Numpy array shape {var.shape}\n')
            h_file.write(f'//Min {np.min(var.min):.12f}\n')
            h_file.write(f'//Max {np.max(var.max):.12f}\n')
            h_file.write(f'//Number of zeros {var.nzeros}\n')
            h_file.write('\n')

            h_file.write(f'#ifndef {var.name.upper()}_H_\n')
            h_file.write(f'#define {var.name.upper()}_H_\n')
            h_file.write('\n')

            rf = int(layer.get_attr('reuse_factor', 1))

            h_file.write(var.definition_cpp(rf) + ' = {{')

            # fill c++ array.
            # not including internal brackets for multidimensional case
            sep = ''
            for x in var:
                h_file.write(sep + x)
                sep = ', '
            h_file.write('}};\n')
            h_file.write('\n#endif\n')

    def write_project_dir(self, model):
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f'{model.config.get_output_dir()}/src/firmware/weights'):
            os.makedirs(f'{model.config.get_output_dir()}/src/firmware/weights')

    def write_project_cpp(self, model):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        project_name = model.config.get_project_name()

        filedir = os.path.dirname(os.path.abspath(__file__))

        autoreg_model: bool = model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None
        # TODO - To be used later.
        # inp_pos_stream: bool = (
        #    model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault('InpPosStream', False)
        #    if autoreg_model
        #    else False
        # )
        maxInvoc = model.config.get_config_value('HLSConfig').setdefault('MaxInvoc', None)
        invoc_props = (
            (',ts_invoc_props>' if shutil.which('ahls') else f',{maxInvoc},{maxInvoc}>') if maxInvoc is not None else '>'
        )

        with (
            open(os.path.join(filedir, '../templates/altera/firmware/myproject.cpp')) as f,
            open(f'{model.config.get_output_dir()}/src/firmware/{project_name}.cpp', 'w') as fout,
        ):
            model_inputs = model.get_input_variables()
            model_outputs = model.get_output_variables()
            model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

            if len(model_brams) != 0:
                raise NotImplementedError('Weights on the interface is currently not supported')

            io_type = model.config.get_config_value('IOType')
            indent = '    '

            for line in f.readlines():
                # Add headers to weights and biases
                if 'myproject' in line:
                    newline = line.replace('myproject', project_name)
                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                # Altera pipes need to be declared and passed as template parameters
                elif '// hls-fpga-machine-learning insert inter-task pipes' in line:
                    newline = line
                    if io_type == 'io_stream':
                        if autoreg_model:
                            opt_names = [layer.name for layer in model_outputs]
                            backend_name = 'altera' if shutil.which('ahls') else 'intel'
                            pipe_template = (
                                'class {pipe_id};\nusing {pipe_name} = '
                                f'sycl::ext::{backend_name}::experimental::pipe<'
                                '{pipe_id}, {pipe_var_type}, {pipe_depth}>;\n'
                            )

                        for layer in model.get_layers():
                            if autoreg_model:
                                layer_out_var = layer.get_output_variable()
                                layer_pipe_name = layer_out_var.pipe_name
                                layer_pipe_id = layer_out_var.pipe_id
                                layer_pipe_depth = layer_out_var.pragma[1]
                                layer_pipe_var_type = layer_out_var.type.name
                                layer_unit_precision = layer_out_var.type.precision
                                layer_pipe_var_unit_type_decl = (
                                    f'using unit_{layer_pipe_var_type} = '
                                    f'ac_fixed<{layer_unit_precision.width}, {layer_unit_precision.integer}, '
                                    f'{"true" if layer_unit_precision.signed else "false"}>;\n'
                                )
                                layer_pipe_var_unit_type = f'nnet::array<unit_{layer_pipe_var_type}, 1>'
                                layer_pipe_var_unit_data_packet_type = f'nnet::DataPacket<{layer_pipe_var_unit_type}>'
                                layer_data_packet_type = f'nnet::DataPacket<{layer_pipe_var_type}>'

                                if layer.inputs == ['input']:
                                    newline += layer_pipe_var_unit_type_decl
                                    newline += pipe_template.format(
                                        pipe_name='SW_' + layer_pipe_name,
                                        pipe_id='SW_' + layer_pipe_id,
                                        pipe_depth=layer_pipe_depth,
                                        pipe_var_type=layer_pipe_var_unit_data_packet_type,
                                    )

                                    """
                                    # TODO - DETERMINE SEPERATE TYPE FOR POS?
                                    if inp_pos_stream:
                                        newline +=  pipe_template.format(
                                            pipe_name='SW_POS_' + layer_pipe_name,
                                            pipe_id='SW_POS_' + layer_pipe_id,
                                            pipe_depth=layer_pipe_depth,
                                            pipe_var_type=layer_pipe_var_unit_data_packet_type,
                                        )
                                    """

                                    newline += pipe_template.format(
                                        pipe_name='FB_' + layer_pipe_name,
                                        pipe_id='FB_' + layer_pipe_id,
                                        pipe_depth=layer_pipe_depth,
                                        pipe_var_type=layer_pipe_var_unit_type,
                                    )

                                    """
                                    newline +=  pipe_template.format(
                                        pipe_name= f'SwitchControl{idx}',
                                        pipe_id= f'SwitchControl{idx}ID',
                                        pipe_depth=layer_pipe_depth,
                                        pipe_var_type='nnet::PipeSignal',
                                    )
                                    """

                                elif layer_out_var.name in opt_names:
                                    # If the layer performs early argmax, update the return type
                                    if model.graph[layer.name].get_attr('argmax') == 'true':
                                        layer_data_packet_type = f'nnet::DataPacket<unit_{layer_pipe_var_type}>'

                                    newline += pipe_template.format(
                                        pipe_name='SW_' + layer_pipe_name,
                                        pipe_id='SW_' + layer_pipe_id,
                                        pipe_depth=layer_pipe_depth,
                                        pipe_var_type=layer_data_packet_type,
                                    )

                                vars = layer.get_variables()
                                for var in vars:
                                    if var not in model_inputs and var not in model_outputs:
                                        # Convert to DataPacket type for autoregressive model
                                        tmp = copy.deepcopy(var)
                                        tmp.type.name = f'nnet::DataPacket<{var.type.name}>'
                                        newline += tmp.declare_cpp()

                            else:
                                vars = layer.get_variables()
                                for var in vars:
                                    if var not in model_inputs and var not in model_outputs:
                                        newline += var.declare_cpp()

                            newline += '\n'

                elif (
                    '// hls-fpga-machine-learning insert invocation props' in line
                    and maxInvoc is not None
                    and shutil.which('ahls')
                ):
                    newline = line
                    newline += indent + 'using ts_invoc_props = decltype(sycl::ext::altera::experimental::properties{\n'
                    newline += indent + indent + f'sycl::ext::altera::experimental::invocation_capacity<{maxInvoc}>,\n'
                    newline += indent + indent + f'sycl::ext::altera::experimental::response_capacity<{maxInvoc}>' + '});\n'

                # Read in inputs
                elif '// hls-fpga-machine-learning read in' in line:
                    newline = line
                    if io_type == 'io_parallel':
                        for inp in model_inputs:
                            newline += indent + f'auto {inp.name} = {inp.pipe_name}::read();\n'
                    # for streaming we don't need to read it in

                # Insert task sequences
                elif '// hls-fpga-machine-learning declare task sequences' in line:
                    newline = line
                    if io_type == 'io_stream':  # only need this for io_stream
                        if autoreg_model:
                            assert len(model_inputs) == len(model_outputs), (
                                'Output is not feeding back the correct number of inputs.'
                            )

                            for idx, inp in enumerate(model_inputs):
                                name = inp.pipe_name
                                in_ts = (
                                    f'task_sequence<nnet::input_switch<{name}, {"FB_" + name}, {"SW_" + name}, ',
                                    f'switch_config>{invoc_props} inp_sw{idx};',
                                )
                                newline += '    ' + in_ts + '\n'

                        for layer in model.get_layers():
                            ts = layer.get_attr('task_sequence_cpp')
                            if ts:
                                newline += '    ' + ts + '\n'

                        if autoreg_model:
                            for idx, out in enumerate(model_outputs):
                                name = out.pipe_name
                                out_ts = (
                                    f'task_sequence<nnet::output_switch<{name}, '
                                    f'{"FB_" + model_inputs[idx].pipe_name}, '
                                    f'{"SW_" + name}, switch_config>{invoc_props} out_sw{idx};'
                                )
                                newline += '    ' + out_ts + '\n'

                # Neural net instantiation
                elif '// hls-fpga-machine-learning insert layers' in line:
                    newline = line + '\n'

                    # Add input swich next to host if model is tagged as autoregressive
                    if autoreg_model:
                        for idx in range(len(model_inputs)):
                            newline += '    ' + f'inp_sw{idx}.async();' + '\n'

                    for layer in model.get_layers():
                        if io_type != 'io_stream':
                            vars = layer.get_variables()
                            for var in vars:
                                if var not in model_inputs:
                                    def_cpp = var.definition_cpp()
                                    if def_cpp is not None:
                                        newline += '    ' + def_cpp + ';\n'
                        func = (
                            layer.get_attr('function_cpp')
                            if io_type == 'io_parallel'
                            else layer.get_attr('stream_function_cpp')
                        )
                        if func:
                            newline += '    ' + func + '\n'
                            if model.config.trace_output and layer.get_attr('trace', False):
                                newline += '#ifndef HLS_SYNTHESIS\n'
                                for var in vars:
                                    newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                        var.type.name, var.name, layer.name, var.size_cpp()
                                    )
                                newline += '#endif\n'

                    # Add output swich next to host if model is tagged as autoregressive
                    if autoreg_model:
                        for idx in range(len(model_outputs)):
                            newline += '    ' + f'out_sw{idx}.async();' + '\n'

                # Write the output
                elif '// hls-fpga-machine-learning return' in line:
                    newline = line
                    if io_type == 'io_parallel':
                        for out in model_outputs:
                            newline += indent + f'{out.pipe_name}::write({out.name});\n'
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
        autoreg_model: bool = model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None
        argmax = False
        if autoreg_model:
            argmax: bool = model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault('Argmax', False)
            # max_iterations = model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault('MaxIterations', 0)

        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/altera/firmware/myproject.h')) as f,
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
                        # Autoreg model is per-token stream so pipes are narrowed until embedding layer,
                        # so we scale the input
                        if autoreg_model:
                            inp_arr = copy.deepcopy(inp)
                            inp_arr.type.name = f'arr_{inp_arr.type.name}'
                            newline += inp_arr.declare_cpp(
                                pipe_min_size=inp.pragma[1] if inp.pragma[0] == 'stream' else 16
                            )  # max_iterations+16)
                        else:
                            newline += inp.declare_cpp(pipe_min_size=inp.pragma[1] if inp.pragma[0] == 'stream' else 16)

                # Insert weights
                elif '// hls-fpga-machine-learning insert weights' in line:
                    newline = line
                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            # if w not in model_brams:
                            newline += f'#include "weights/{w.name}.h"\n'

                # and declareations for the outputs
                elif '// hls-fpga-machine-learning insert outputs' in line:
                    newline = line
                    for out in model_outputs:
                        if autoreg_model:
                            out_dp = copy.deepcopy(out)
                            if argmax:
                                out_dp.type.name = f'nnet::DataPacket<unit_{out_dp.type.name}>'
                            else:
                                out_dp.type.name = f'nnet::DataPacket<{out_dp.type.name}>'
                            newline += out_dp.declare_cpp(
                                pipe_min_size=out_dp.pragma[1] if out_dp.pragma[0] == 'stream' else 16
                            )
                        else:
                            newline += out.declare_cpp(pipe_min_size=out.pragma[1] if out.pragma[0] == 'stream' else 16)

                # Simply copy line, if no inserts are required
                else:
                    newline = line

                fout.write(newline)

    def write_defines(self, model):
        """Write the C++ type definitions file (defines.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        model_outputs = {layer.name: layer.type for layer in model.get_output_variables()}
        autoreg_model: bool = model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None
        argmax = False
        if autoreg_model:
            argmax: bool = model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault('Argmax', False)

        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/altera/firmware/defines.h')) as f,
            open(f'{model.config.get_output_dir()}/src/firmware/defines.h', 'w') as fout,
        ):
            for line in f.readlines():
                if '// hls-fpga-machine-learning insert layer-precision' in line:
                    layer_io_names = []
                    newline = line
                    all_precision = OrderedDict()
                    for layer in model.get_layers():
                        layer_io_names += layer.inputs + layer.outputs
                        layer_precision = layer.get_layer_precision()

                        if argmax and (layer.get_output_variable().name in model_outputs.keys()):
                            layer_type = model_outputs[layer.get_output_variable().name]
                            lay_prec = layer_type.precision
                            # TODO - Maybe fix precision here to have smth better?
                            newline += (
                                f'typedef nnet::array<ac_fixed<{lay_prec.width},{lay_prec.width - 1},'
                                f'{"true" if lay_prec.signed else "false"}>, 1> unit_{layer_type.name};\n'
                            )

                        for type_name, type_var in layer_precision.items():
                            # Ensure that layer's types doesn't override existing types
                            # This can happen in case of InplaceVariable types
                            if type_name not in all_precision:
                                all_precision[type_name] = type_var

                            if autoreg_model:
                                inp_vars = {var.type.name: var.shape[-1] for var in model.get_input_variables()}
                                if type_name in inp_vars.keys():
                                    tmp_var = copy.deepcopy(type_var)
                                    tmp_var.name = f'arr_{tmp_var.name}'
                                    tmp_var.n_elem = inp_vars[type_name]
                                    all_precision['arr_' + type_name] = tmp_var

                    for used_type in all_precision.values():
                        newline += used_type.definition_cpp()

                else:
                    newline = line
                fout.write(newline)

    def write_parameters(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        io_type = model.config.get_config_value('IOType')
        autoreg_model: bool = model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None

        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/altera/firmware/parameters.h')) as f,
            open(f'{model.config.get_output_dir()}/src/firmware/parameters.h', 'w') as fout,
        ):
            for line in f.readlines():
                if '// hls-fpga-machine-learning insert includes' in line:
                    newline = line

                    # Include the pipe switches for autoregressive model
                    if autoreg_model and io_type == 'io_stream':
                        newline += '#include "nnet_utils/nnet_dma_helpers.h"\n'

                    for include in sorted(
                        set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))
                    ):
                        newline += '#include "%s"\n' % include

                    # Extra required include for host RW
                    if model.config.get_config_value('HLSConfig').setdefault('HostRW', 0):
                        newline += '#include "nnet_utils/nnet_data_movement.h"\n'

                elif '// hls-fpga-machine-learning insert layer-config' in line:
                    newline = line

                    if autoreg_model:
                        autoreg = model.config.get_config_value('HLSConfig')['Autoregressive']

                        assert type(autoreg) is dict, (
                            'Wrong type passed to autoregressive config, '
                            "it must be a dictionary with 'SwitchStateID' and 'StopStateID' arguments."
                        )

                        switch_id = model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault(
                            'SwitchStateID', None
                        )
                        stop_id = model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault(
                            'StopStateID', None
                        )
                        max_iterations = model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault(
                            'MaxIterations', None
                        )
                        argmax = (
                            model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault('Argmax', False)
                            if autoreg_model
                            else False
                        )
                        argmax_cpp = 'true' if (argmax is True) else 'false'

                        # We check if we have performed an 'early argmax' on the previous layer
                        # so we don't need to in the output switch
                        # This was done to optimise hardware and avoid carrying potentially
                        # long and large data inside the FIFO pipes to the next layer.

                        # TODO - Adjust this to seperate process for each switch config

                        for out_node in model.outputs:
                            if model.graph[out_node].get_attr('argmax') == 'true':
                                argmax_cpp = 'false'  # Revert argmax if it was performed on the previus layer

                        assert (switch_id is not None) and (stop_id is not None), (
                            'Switch/Stop conditions are undefined, '
                            "pass those as dictionary arguments: 'SwitchStateID' and 'StopStateID'"
                        )

                        MAX_ITER = 1024
                        if max_iterations is None:
                            print(f'Warning: Max Iterations is not set, default value of {MAX_ITER} used.')
                            max_iterations = MAX_ITER

                        indent = ' ' * (len(line) - len(line.lstrip(' ')))
                        newline += 'struct switch_config {\n'
                        newline += indent + f'static constexpr unsigned switch_signal = {switch_id};\n'
                        newline += indent + f'static constexpr unsigned stop_signal = {stop_id};\n'
                        newline += indent + f'static constexpr size_t max_iterations = {max_iterations};\n'
                        newline += indent + f'static constexpr bool argmax = {argmax_cpp};\n'
                        newline += '};\n\n'

                    for layer in model.get_layers():
                        config = layer.get_attr('config_cpp', None)
                        if config:
                            newline += config + '\n'

                elif '// hls-fpga-machine-learning insert softmax tables' in line:
                    newline = line
                    for layer in model.get_layers():
                        if (
                            layer.get_attr('activation') == 'softmax'
                            or layer.get_attr('recurrent_activation') == 'softmax'
                            or layer.get_attr('activation') == 'softmax_multidim'
                            or layer.get_attr('recurrent_activation') == 'softmax_multidim'
                        ) and 'implementation' in layer.attributes:
                            newline += f'#include "nnet_utils/activation_tables/{layer.name}_exp_table.h"\n'
                            newline += f'#include "nnet_utils/activation_tables/{layer.name}_inv_table.h"\n'

                else:
                    newline = line
                fout.write(newline)

    def write_weights(self, model):
        """Write the weights into header files

        Args:
            model (ModelGraph): the hls4ml model.
        """
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(weights, layer, model.config.get_output_dir())

    def write_test_bench(self, model):
        """Write the testbench

        Args:
            model (ModelGraph): the hls4ml model.
        """
        # TODO - This function only works with one model input
        # (NOT one data point - it works as expected with multiple data points)

        # copy the exception handler
        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath = os.path.join(filedir, '../templates/altera/exception_handler.hpp')
        dstpath = f'{model.config.get_output_dir()}/src/exception_handler.hpp'
        copyfile(srcpath, dstpath)

        project_name = model.config.get_project_name()
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        if len(model_brams) != 0:
            raise NotImplementedError('Weights on the interface is currently not supported')

        if len(model_inputs) != 1 or len(model_outputs) != 1:
            print('The testbench supports only single input arrays and single output arrays.')
            print('Please modify it before using it.')

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

        host_rw_model: bool = model.config.get_config_value('HLSConfig').setdefault('HostRW', 0)
        memory_type = 'malloc_shared' if host_rw_model else 'malloc_host'
        autoreg_model: bool = model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None
        max_invoc = model.config.get_config_value('HLSConfig').setdefault('MaxInvoc', 1)
        max_iterations = max_invoc
        if autoreg_model:
            max_iterations = model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault('MaxIterations', 0)

        with (
            open(os.path.join(filedir, '../templates/altera/myproject_test.cpp')) as f,
            open(f'{model.config.get_output_dir()}/src/{project_name}_test.cpp', 'w') as fout,
        ):
            for line in f.readlines():
                indent = ' ' * (len(line) - len(line.lstrip(' ')))

                if 'myproject' in line:
                    newline = line.replace('myproject', project_name)
                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                elif '// hls-fpga-machine-learning create host mems' in line and host_rw_model:
                    newline = line

                    for idx, inp in enumerate(model_inputs):
                        inp_type = inp.definition_cpp().split(' ')[0]
                        num = idx if idx >= 1 else ''
                        newline += indent + f'using {inp.name}_item_t = typename {inp_type}::value_type;\n'
                        newline += (
                            indent
                            + f'{inp.name}_item_t* {inp.name}_vals = '
                            + f'sycl::{memory_type}<{inp.name}_item_t>({inp.size_cpp()}, q);\n'
                        )
                        newline += indent + f'if ({inp.name}_vals == nullptr)' + '{\n'
                        newline += (
                            indent + indent + f'std::cerr << "ERROR: host allocation failed for {inp.name} (input{num})";\n'
                        )
                        newline += indent + indent + 'fout.close();\n'
                        newline += indent + indent + 'return 1;\n'
                        newline += indent + '}\n'
                        newline += (
                            indent
                            + f'std::fill({inp.name}_vals, {inp.name}_vals + {inp.size_cpp()}, {inp.name}_item_t{{}});\n\n'
                        )

                    for idx, out in enumerate(model_outputs):
                        if autoreg_model:
                            out_buffer_size = str(max_iterations)  # + out.pragma[1]) # pipe_width * num_reads
                        else:
                            out_buffer_size = str(out.pragma[1] * np.prod(out.shape))
                        out_type = out.definition_cpp().split(' ')[0]
                        num = idx if idx >= 1 else ''
                        newline += indent + f'using output{num}_item_t = typename {out_type}::value_type;\n'
                        newline += (
                            indent
                            + f'output{num}_item_t* output{num}_vals = '
                            + f'sycl::{memory_type}<output{num}_item_t>({out_buffer_size}, q);\n'
                        )
                        newline += indent + f'if (output{num}_vals == nullptr)' + '{\n'
                        newline += (
                            indent + indent + f'std::cerr << "ERROR: host allocation failed for {out.name} (output{num})";\n'
                        )
                        newline += indent + indent + 'fout.close();\n'
                        newline += indent + indent + 'return 1;\n'
                        newline += indent + '}\n'

                elif '// hls-fpga-machine-learning read inputs' in line and host_rw_model:
                    newline = line
                    for idx, inp in enumerate(model_inputs):
                        num = idx if idx >= 1 else ''
                        newline += (
                            indent + f'read_input<{inp.name}_item_t>("{inp.name}", '
                            f'"{inp.name}_vals.tb", {inp.size_cpp()}, {inp.name}_vals);\n'
                        )

                    # For debugging
                    newline += indent + 'std::cout << "Filled arrays are:" << std::endl;\n'
                    for inp in model_inputs:
                        size = math.prod([int(it) for it in inp.size_cpp().split('*')])
                        newline += indent + f'std::cout << "{inp.name} (input{num}): ";\n'
                        newline += indent + f'for(int j = 0; j < {size - 1}; j++) std::cout << {inp.name}_vals[j] << ",";\n'
                        newline += indent + f'std::cout << {inp.name}_vals[{size - 1}] << std::endl;\n'

                elif '// hls-fpga-machine-learning launch kernels' in line and host_rw_model:  # TODO: TRY WITH 2+ INPUTS
                    newline = line
                    inp_names = ','.join([f'{inp.name}_vals' for inp in model_inputs])
                    out_names = ','.join([f'output{idx if idx >= 1 else ""}_vals' for idx, out in enumerate(model_outputs)])
                    out_t = ','.join([f'output{idx if idx >= 1 else ""}_item_t' for idx, out in enumerate(model_outputs)])
                    out_pipe_names = ','.join([out.pipe_name for out in model_outputs])
                    out_sizes = ','.join(
                        [str(max_iterations if autoreg_model else np.prod(out.shape)) for out in model_outputs]
                    )

                    if len(model_inputs) > 99:
                        for idx, inp in enumerate(model_inputs):
                            num = idx if idx >= 1 else ''
                            newline += (
                                indent + f'using {inp.name}_pair = nnet::SrcPipePair<{inp.name}_item_t, {inp.pipe_name}>;\n'
                            )

                        pairs = ','.join([f'{inp.name}_pair' for idx, inp in enumerate(model_inputs)])
                        if len(model_inputs) > 1:
                            newline += (
                                indent
                                + f'q.single_task(nnet::DMA_convert_data<{pairs}>'
                                + '{'
                                + inp_names
                                + f', {inp.size_cpp()}'
                                + '});\n'
                            )
                    else:
                        # Assumes size == 1, will adjust this as an alternative "kernel-per-input" model
                        for inp in model_inputs:
                            newline += (
                                indent
                                + f'q.single_task(nnet::DMA_convert_data_single<{inp.name}_item_t, {inp.pipe_name}>'
                                + '{'
                                + f'{inp.name}_vals'
                                + ', 1'  # {inp.size_cpp()}'
                                + '});\n'
                            )

                    newline += indent + 'q.single_task(Myproject{});\n'
                    newline += (
                        indent
                        + 'constexpr unsigned packing = '
                        + f'{out_sizes}/std::tuple_size<typename nnet::ExtractPipeType<{out_pipe_names}>'
                        + f'::value_type{"::data_type" if autoreg_model else ""}>'
                        + '{'
                        + '};\n'
                    )  # TODO: EXTREMELY DODGY FOR OUT SIZE > 1
                    newline += (
                        indent
                        + 'q.single_task(nnet::DMA_convert_data_back<'
                        + out_pipe_names
                        + ', '
                        + out_t
                        + ', uint32_t, std::size_t>{'
                        + out_names
                        + ', ttft_flag, tx_counter, packing}).wait();\n'
                    )

                elif '// hls-fpga-machine-learning write out to file' in line and host_rw_model:
                    newline = line
                    for idx, out in enumerate(model_outputs):
                        num = idx if idx >= 1 else ''
                        newline += (
                            indent
                            + f'constexpr unsigned output{num}_pipeOutSize = '
                            + f'std::tuple_size<typename nnet::ExtractPipeType<{out.pipe_name}>'
                            + f'::value_type{"::data_type" if autoreg_model else ""}>'
                            + '{'
                            + '};\n'
                        )
                        newline += (
                            indent
                            + 'for (int i = 0; i < static_cast<int>(num_tokens); i++) '
                            + '{\n'  # indent + f'for (int i = 0; i < {out_total_item_size}; i++) ' + '{\n'
                        )  # TODO: ADJUST FOR NUMBER OF EXPECTED TOKENS BASED ON IF WE ARE DOING AUTOREG MODEL OR NOT
                        newline += indent + indent + f'for (int j = 0; j < output{num}_pipeOutSize; j++) ' + '{\n'
                        newline += (
                            indent + indent + indent + f'fout << output{num}_vals[i * output{num}_pipeOutSize + j] << " ";\n'
                        )
                        newline += indent + indent + '}\n'
                        newline += indent + indent + 'fout << std::endl;\n'
                        newline += indent + '}\n'

                # Free memory only if we have the host reads flag
                elif '// hls-fpga-machine-learning free host mem' in line and host_rw_model:
                    newline = line
                    for idx, inp in enumerate(model_inputs):
                        num = idx if idx >= 1 else ''
                        newline += indent + f'sycl::free({inp.name}_vals, q);\n'
                    for idx in range(len(model_outputs)):
                        num = idx if idx >= 1 else ''
                        newline += indent + f'sycl::free(output{num}_vals, q);\n'

                elif '// hls-fpga-machine-learning insert bram' in line:
                    newline = line
                    for bram in model_brams:
                        newline += f'#include "firmware/weights/{bram.name}.h"\n'

                elif '// hls-fpga-machine-learning insert zero' in line:
                    newline = line
                    for inp in model_inputs:
                        newline += indent + f'float {inp.name}_vals[{inp.size_cpp()}]; \n'
                        newline += indent + f'for (int j = 0 ; j < {inp.size_cpp()} ; j++) {{\n'
                        newline += indent + f'    {inp.name}_vals[j] = 0.0; \n'
                        newline += indent + '}\n'
                        newline += (
                            indent + f'nnet::convert_data<float, {inp.pipe_name}, {inp.size_cpp()}>(q, {inp.name}_vals);\n'
                        )

                elif '// hls-fpga-machine-learning insert data' in line:
                    newline = line
                    for inp in model_inputs:
                        newline += indent + f'float {inp.name}_vals[{inp.size_cpp()}]; \n'
                        newline += indent + f'for (int j = 0 ; j < {inp.size_cpp()} ; j++) {{\n'
                        newline += indent + f'    {inp.name}_vals[j] = in[j]; \n'
                        newline += indent + '}\n'
                        newline += (
                            indent + f'nnet::convert_data<float, {inp.pipe_name}, {inp.size_cpp()}>(q, {inp.name}_vals);\n'
                        )

                elif '// hls-fpga-machine-learning convert output' in line:
                    newline = line
                    for out in model_outputs:
                        newline += indent + f'float {out.name}_vals[{out.size_cpp()}];\n'
                        newline += (
                            indent
                            + f'nnet::convert_data_back<{out.pipe_name}, float, {out.size_cpp()}>(q, {out.name}_vals);\n\n'
                        )

                        newline += (
                            indent + f'fout << "OUTPUT: {out.name}, ITERATION: " << iteration << ", VALS: " << std::endl;\n'
                        )
                        newline += indent + f'for (auto outval : {out.name}_vals)' + '{\n'
                        newline += indent + indent + 'fout << outval << " ";\n'
                        newline += indent + '}\n'
                        newline += indent + 'fout << std::endl;\n\n'

                        newline += indent + f'std::cout << "OUTPUT: {out.name}, ITERATION: " << iteration << ", VALS: ";\n'
                        newline += indent + f'for (auto outval : {out.name}_vals)' + '{\n'
                        newline += indent + indent + 'std::cout << outval << " ";\n'
                        newline += indent + '}\n'
                        newline += indent + 'std::cout << std::endl;\n\n'

                elif '// hls-fpga-machine-learning insert quantized' in line:
                    newline = line
                    newline += indent + f'std::cout << "OUTPUT: {out.name}, ITERATION: " << iteration << ", VALS: ";\n'
                    newline += indent + f'for (auto outval : {out.name}_vals)' + '{\n'
                    newline += indent + indent + 'std::cout << outval << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n\n'

                else:
                    newline = line

                fout.write(newline)

    def write_bridge(self, model):
        """Write the Python-C++ bridge (myproject_bridge.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        project_name = model.config.get_project_name()
        stamp = model.config.get_config_value('Stamp')
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']
        # model brambs aren't actually supported yet

        # io_type = model.config.get_config_value('IOType')
        indent = '    '

        filedir = os.path.dirname(os.path.abspath(__file__))

        autoreg_model: bool = model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None
        if autoreg_model:
            max_iterations = model.config.get_config_value('HLSConfig')['Autoregressive'].setdefault('MaxIterations', 0)
            assert max_iterations >= 0, 'Invalid max iterations passed (ensure > 0).'
        host_rw_model: bool = model.config.get_config_value('HLSConfig').setdefault('HostRW', 0)

        with (
            open(os.path.join(filedir, '../templates/altera/myproject_bridge.cpp')) as f,
            open(f'{model.config.get_output_dir()}/src/{project_name}_bridge.cpp', 'w') as fout,
        ):
            for line in f.readlines():
                if 'MYPROJECT' in line:
                    newline = line.replace('MYPROJECT', format(project_name.upper()))

                elif 'myproject' in line:
                    newline = line.replace('myproject', format(project_name))

                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                elif '// hls-fpga-machine-learning insert bram' in line:
                    newline = line
                    for bram in model_brams:
                        newline += f'#include "firmware/weights/{bram.name}.h"\n'

                elif '// hls-fpga-machine-learning insert class def' in line:
                    dtype = line.split('#', 1)[1].strip()
                    newline = f'class {convert_to_pascal_case(project_name)}Class{dtype.capitalize()}_{stamp};\n'

                elif '// hls-fpga-machine-learning insert header' in line:
                    dtype = line.split('#', 1)[1].strip()

                    if not host_rw_model:
                        inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                        outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])
                    else:
                        inputs_str = ', '.join([f'{dtype} {i.name}_vals[{i.size_cpp()}]' for i in model_inputs])
                        if autoreg_model:
                            outputs_str = ', '.join(
                                [
                                    f'{dtype} output{idx if idx >= 1 else ""}_vals[{max_iterations}]'
                                    for idx, o in enumerate(model_outputs)
                                ]
                            )
                        else:
                            outputs_str = ', '.join(
                                [
                                    f'{dtype} output{idx if idx >= 1 else ""}_vals[{o.size_cpp()}]'
                                    for idx, o in enumerate(model_outputs)
                                ]
                            )

                    newline = ''
                    newline += indent + inputs_str + ',\n'
                    newline += indent + outputs_str + '\n'

                elif '// hls-fpga-machine-learning insert wrapper' in line:
                    if not host_rw_model:
                        dtype = line.split('#', 1)[1].strip()
                        newline = ''
                        for i in model_inputs:
                            newline += indent + f'nnet::convert_data<{dtype}, {i.pipe_name}, {i.size_cpp()}>(q, {i.name});\n'
                    else:
                        dtype = line.split('#', 1)[1].strip()
                        newline = ''

                        # TODO - MULTI INPUTS ARE INOP FOR THE TIME BEING THIS WILL BE FIXED
                        """if len(model_inputs) > 1:
                            for inp in model_inputs:
                                newline += indent + f'using {inp.name}_pair = nnet::SrcPipePair<{dtype}, {inp.pipe_name}>;\n'
                            pairs = ','.join([f'{inp.name}_pair' for idx, inp in enumerate(model_inputs)])
                            inp_names = ','.join([f'{inp.name}_vals' for inp in model_inputs])
                            newline += (
                                indent
                                + f'q.single_task(nnet::DMA_convert_data<{pairs}>'
                                + '{'
                                + inp_names
                                + f', {model_inputs[0].size_cpp()}'
                                + '});\n'
                            )
                        else:"""

                        inp_size = '1' if autoreg_model else model_inputs[0].size_cpp()
                        for inp in model_inputs:
                            newline += (
                                indent
                                + f'q.single_task(nnet::DMA_convert_data_single<{dtype}, {inp.pipe_name}>'
                                + '{'
                                + f'{inp.name}_vals'
                                + f', {inp_size}'
                                + '});\n'
                            )

                    newline += (
                        indent
                        + f'q.single_task<{convert_to_pascal_case(project_name)}Class{dtype.capitalize()}_{stamp}>'
                        + f'({convert_to_pascal_case(project_name)}{{}});\n'
                    )

                    if not host_rw_model:
                        for o in model_outputs:
                            newline += (
                                indent + f'nnet::convert_data_back<{o.pipe_name}, {dtype}, {o.size_cpp()}>(q, {o.name});\n'
                            )
                        newline += '\n'
                        newline += indent + 'q.wait();\n'
                    else:
                        for out in model_outputs:
                            out_names = ','.join(
                                [f'output{idx if idx >= 1 else ""}_vals' for idx, out in enumerate(model_outputs)]
                            )
                            out_pipe_names = ','.join([out.pipe_name for out in model_outputs])
                            out_sizes = ','.join([out.size_cpp() for out in model_outputs])
                            newline += (
                                indent
                                + f'constexpr unsigned packing = {out_sizes}'
                                + f'/std::tuple_size<typename nnet::ExtractPipeType<{out_pipe_names}>'
                                + f'::value_type{"::data_type" if autoreg_model else ""}>'
                                + '{'
                                + '};\n'
                            )
                            newline += (
                                indent
                                + 'q.single_task(nnet::DMA_convert_data_back_bridge_ver<'
                                + out_pipe_names
                                + ', '
                                + dtype
                                + '>{'
                                + out_names
                                + ', packing}).wait();\n'
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

    def write_build_script(self, model):
        """Write the build scripts (Makefile, build_lib.sh)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        autoreg_model: bool = model.config.get_config_value('HLSConfig').setdefault('Autoregressive', None) is not None
        host_rw_model: bool = model.config.get_config_value('HLSConfig').setdefault('HostRW', 0)

        # Makefile
        filedir = os.path.dirname(os.path.abspath(__file__))
        device = model.config.get_config_value('Part')
        period = model.config.get_config_value('ClockPeriod')
        hyper = model.config.get_config_value('HyperoptHandshake')
        first_seen = False
        with (
            open(os.path.join(filedir, '../templates/altera/CMakeLists.txt')) as f,
            open(f'{model.config.get_output_dir()}/CMakeLists.txt', 'w') as fout,
        ):
            for line in f.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

                if 'set(FPGA_DEVICE' in line:
                    line = f'    set(FPGA_DEVICE "{device}")\n'

                if 'set(USER_FPGA_FLAGS' in line and not first_seen:
                    first_seen = True
                    line += f'set(USER_FPGA_FLAGS -Xsclock={period}ns; ${{USER_FPGA_FLAGS}})\n'
                    if not autoreg_model:
                        if not hyper:
                            line += 'set(USER_FPGA_FLAGS -Xsoptimize=latency; ${USER_FPGA_FLAGS})\n'
                    elif autoreg_model:
                        line += f'set(USER_FPGA_FLAGS -Xsclock={period}ns; ${{USER_FPGA_FLAGS}})\n'
                        line += 'set(USER_FPGA_FLAGS -Xsfast-loop-orchestration=on; ${USER_FPGA_FLAGS})\n'

                if 'project(' in line:
                    line = line.replace('myproject', model.config.get_project_name())
                    if autoreg_model:
                        line += '\nadd_compile_definitions(AUTOREG)\n'
                    if host_rw_model:
                        line += 'add_compile_definitions(HOST_READS)\n'
                        line += (
                            'add_compile_definitions(USM_MEMORY)\n'  # TODO - Make this a conditional for usm/non-usm BSPs
                        )

                fout.write(line)

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/altera/firmware/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/src/firmware/nnet_utils/'

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        # custom source
        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = get_backend('Altera').get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = f'{model.config.get_output_dir()}/src/firmware/{dst}'
            copyfile(srcpath, dstpath)

    def __get_table_size(self, model, activation, size_attr_name='table_size'):
        for layer in model.get_layers():
            if (
                layer.get_attr('activation') == activation or layer.get_attr('recurrent_activation') == activation
            ) and layer.get_attr(size_attr_name) is not None:
                return int(layer.get_attr(size_attr_name))
        return 1024

    def __get_table_header(self, table_name, table_size, table_type='table_t'):
        table_header = f'static const typename CONFIG_T::{table_type} {table_name}[{table_size}] = {{'
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
            sep = ', '

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
                sep = ', '

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
                sep = ', '

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
            sep = ', '

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
                sep = ', '

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
            sep = ', '

        h_file.write('};\n')
        h_file.close()

    def __get_table_precision(self, model, activation, table_name='table_precision'):
        for layer in model.get_layers():
            if layer.get_attr('activation') == activation and layer.get_attr(table_name) is not None:
                precision = layer.get_attr(table_name)
                return precision.precision

        return None  # fp_bits, fp_integer, fp_signed

    def __write_exp_tables_stable(self, model, path):

        for layer in model.get_layers():
            # Last property is essential since it seperates layer with activation property from actual activation layers

            if (
                (
                    layer.get_attr('activation') == 'softmax'
                    or layer.get_attr('activation') == 'softmax_multidim'
                    or layer.get_attr('recurrent_activation') == 'softmax'
                    or layer.get_attr('recurrent_activation') == 'softmax_multidim'
                )
                and 'implementation' in layer.attributes
                and layer.get_attr('implementation') == 'stable'
            ):
                table_name = layer.name + '_exp_table'
                table_size = min(int(layer.get_attr('table_size')), int(layer.get_attr('exp_table_size')))

                with open(f'{path}/{table_name}.h', 'w') as h_file:
                    header_name = table_name
                    h_file.write(f'#ifndef {header_name.upper()}_H_\n')
                    h_file.write(f'#define {header_name.upper()}_H_\n\n')

                    h_file.write(
                        f'static constexpr nnet::array<{layer.get_attr("exp_table_t").name},{table_size}> {table_name} = {{'
                    )

                    ac_type = layer.get_attr('inp_norm_t')
                    fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                    fp_integer = ac_type.precision.integer

                    # Copy scaling from attributes
                    scale = (
                        layer.attributes['exp_scale']
                        if (('exp_scale' in layer.attributes) and (layer.attributes['exp_scale'] is not None))
                        else 1.0
                    )

                    N = ceil_log2(table_size)
                    if N > fp_bits:
                        raise Exception('Table size is bigger than what precision allows')
                    maxval = 2**fp_integer - 1

                    # Use the centre of the bin instead of rounding down:
                    # -1 is gives the half (1/2) (N - fp_int) is the available fractional bits
                    half_frac = 2.0 ** (fp_integer - N - 1) if N < fp_bits else 0.0

                    sep = ''
                    # Use the top bits if table_size < 2**bit_width
                    for i in range(table_size):
                        # Norm type is always > 1 so if input quantiser is set to be signed for any reason,
                        # force unsigned but keep the width
                        f = FixedPointEmulator(fp_bits, fp_integer, signed=False)
                        b = uint_to_binary(i, N)
                        f.set_msb_bits(b)
                        x = f.to_float()
                        if half_frac and x > 0:  # x > 0 to preserve (x_max - x) == 0 => exp(0) = 1
                            x += half_frac
                        real_val = math.exp(-(x * scale))
                        if real_val > maxval:
                            real_val = maxval
                        h_file.write(sep + str(real_val))
                        sep = ', '

                    h_file.write('};\n\n')
                    h_file.write('#endif')

    def __write_invert_tables_stable(self, model, path):
        for layer in model.get_layers():
            # Last property is essential since it seperates layer with activation property from actual activation layers

            if (
                (
                    layer.get_attr('activation') == 'softmax'
                    or layer.get_attr('activation') == 'softmax_multidim'
                    or layer.get_attr('recurrent_activation') == 'softmax'
                    or layer.get_attr('recurrent_activation') == 'softmax_multidim'
                )
                and 'implementation' in layer.attributes
                and layer.get_attr('implementation') == 'stable'
            ):
                table_name = layer.name + '_inv_table'
                table_size = min(int(layer.get_attr('table_size')), int(layer.get_attr('inv_table_size')))

                with open(f'{path}/{table_name}.h', 'w') as h_file:
                    header_name = table_name
                    h_file.write(f'#ifndef {header_name.upper()}_H_\n')
                    h_file.write(f'#define {header_name.upper()}_H_\n\n')

                    h_file.write(
                        f'static constexpr nnet::array<{layer.get_attr("inv_table_t").name},{table_size}> {table_name} = {{'
                    )

                    ac_type = layer.get_attr('inv_inp_t')
                    fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                    fp_integer = ac_type.precision.integer

                    N = ceil_log2(table_size)
                    if N > fp_bits:
                        raise Exception('Table size is bigger than what precision allows')
                    maxval = 2**fp_integer - 1

                    # Use the centre of the bin instead of rounding down:
                    # -1 is gives the half (1/2) (N - fp_int) is the available fractional bits
                    half_bin = 2.0 ** (fp_integer - N - 1) if N < fp_bits else 0.0

                    # Use the top bits if table_size < 2**bit_width
                    sep = ''
                    for i in range(table_size):
                        # Norm type is always > 1 so if input quantiser is set to be signed for any reason,
                        # force unsigned but keep the width
                        f = FixedPointEmulator(fp_bits, fp_integer, signed=False)
                        b = uint_to_binary(i, N)
                        f.set_msb_bits(b)
                        x = f.to_float()
                        if half_bin and x != 1.0:  # Again, preserve the special case where x = exp(0) => x = 0
                            x += half_bin
                        real_val = 1.0 / x if x > 0 else maxval
                        if real_val > maxval:
                            real_val = maxval
                        h_file.write(sep + str(real_val))
                        sep = ', '

                    h_file.write('};\n\n')
                    h_file.write('#endif')

    def __write_exp_tables_latency(self, model, path):
        for layer in model.get_layers():
            # Last property is essential since it seperates layer with activation property from actual activation layers
            if (
                (
                    layer.get_attr('activation') == 'softmax'
                    or layer.get_attr('activation') == 'softmax_multidim'
                    or layer.get_attr('recurrent_activation') == 'softmax'
                    or layer.get_attr('recurrent_activation') == 'softmax_multidim'
                )
                and 'implementation' in layer.attributes
                and layer.get_attr('implementation') == 'latency'
            ):
                table_name = layer.name + '_exp_table'
                table_size = int(layer.get_attr('exp_table_size'))

                with open(f'{path}/{table_name}.h', 'w') as h_file:
                    header_name = table_name
                    h_file.write(f'#ifndef {header_name.upper()}_H_\n')
                    h_file.write(f'#define {header_name.upper()}_H_\n\n')

                    h_file.write(
                        f'static constexpr nnet::array<{layer.get_attr("exp_table_t").name},{table_size}> {table_name} = {{'
                    )

                    ac_type = layer.get_input_variable().type
                    fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                    fp_integer = ac_type.precision.integer
                    fp_signed = ac_type.precision.signed

                    sep = ''
                    N = ceil_log2(table_size)
                    for i in range(table_size):
                        f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
                        f.set_msb_bits(uint_to_binary(i, N))
                        real_val = f.exp_float()
                        h_file.write(sep + str(real_val))
                        sep = ', '

                    h_file.write('};\n\n')
                    h_file.write('#endif')

    def __write_invert_tables_latency(self, model, path):
        for layer in model.get_layers():
            # Last property is essential since it seperates layer with activation property from actual activation layers
            if (
                (
                    layer.get_attr('activation') == 'softmax'
                    or layer.get_attr('activation') == 'softmax_multidim'
                    or layer.get_attr('recurrent_activation') == 'softmax'
                    or layer.get_attr('recurrent_activation') == 'softmax_multidim'
                )
                and 'implementation' in layer.attributes
                and layer.get_attr('implementation') == 'latency'
            ):
                table_name = layer.name + '_inv_table'
                table_size = int(layer.get_attr('inv_table_size'))

                with open(f'{path}/{table_name}.h', 'w') as h_file:
                    header_name = table_name
                    h_file.write(f'#ifndef {header_name.upper()}_H_\n')
                    h_file.write(f'#define {header_name.upper()}_H_\n\n')

                    h_file.write(
                        f'static constexpr nnet::array<{layer.get_attr("inv_table_t").name},{table_size}> {table_name} = {{'
                    )

                    ac_type = layer.get_attr('exp_table_t')
                    fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                    fp_integer = ac_type.precision.integer
                    fp_signed = ac_type.precision.signed

                    sep = ''
                    N = ceil_log2(table_size)
                    for i in range(table_size):
                        f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
                        f.set_msb_bits(uint_to_binary(i, N))
                        real_val = f.inv_float()
                        h_file.write(sep + str(real_val))
                        sep = ', '

                    h_file.write('};\n\n')
                    h_file.write('#endif')

    def __write_exp_table_legacy(self, model, path):

        for layer in model.get_layers():
            # Last property is essential since it seperates layer with activation property from actual activation layers
            if (
                (
                    layer.get_attr('activation') == 'softmax'
                    or layer.get_attr('activation') == 'softmax_multidim'
                    or layer.get_attr('recurrent_activation') == 'softmax'
                    or layer.get_attr('recurrent_activation') == 'softmax_multidim'
                )
                and 'implementation' in layer.attributes
                and layer.get_attr('implementation') == 'legacy'
            ):
                table_name = layer.name + '_exp_table'
                table_size = int(layer.get_attr('exp_table_size'))  # not sure if it works, have to test first

                with open(f'{path}/{table_name}.h', 'w') as h_file:
                    header_name = table_name
                    h_file.write(f'#ifndef {header_name.upper()}_H_\n')
                    h_file.write(f'#define {header_name.upper()}_H_\n\n')

                    h_file.write(
                        f'static constexpr nnet::array<{layer.get_attr("exp_table_t").name},{table_size}> {table_name} = {{'
                    )

                    sep = ''
                    for i in range(table_size):
                        in_val = 2 * 8.0 * (i - float(table_size) / 2.0) / float(table_size)
                        real_val = np.exp(in_val)
                        h_file.write(sep + str(real_val))
                        sep = ', '

                    h_file.write('};\n\n')
                    h_file.write('#endif')

    def __write_invert_table_legacy(self, model, path):

        for layer in model.get_layers():
            # Last property is essential since it seperates layer with activation property from actual activation layers
            if (
                (
                    layer.get_attr('activation') == 'softmax'
                    or layer.get_attr('activation') == 'softmax_multidim'
                    or layer.get_attr('recurrent_activation') == 'softmax'
                    or layer.get_attr('recurrent_activation') == 'softmax_multidim'
                )
                and 'implementation' in layer.attributes
                and layer.get_attr('implementation') == 'legacy'
            ):
                table_name = layer.name + '_inv_table'
                table_size = int(layer.get_attr('inv_table_size'))

                with open(f'{path}/{table_name}.h', 'w') as h_file:
                    header_name = table_name
                    h_file.write(f'#ifndef {header_name.upper()}_H_\n')
                    h_file.write(f'#define {header_name.upper()}_H_\n\n')

                    h_file.write(
                        f'static constexpr nnet::array<{layer.get_attr("inv_table_t").name},{table_size}> {table_name} = {{'
                    )

                    sep = ''
                    for i in range(table_size):
                        real_val = 0
                        in_val = 64.0 * i / float(table_size)
                        if in_val > 0.0:
                            real_val = 1.0 / in_val
                        h_file.write(sep + str(real_val))
                        sep = ', '

                    h_file.write('};\n\n')
                    h_file.write('#endif')

    def write_activation_tables(self, model):
        """Write the lookup tables for activation functions

        Args:
            model (ModelGraph): the hls4ml model.
        """
        # Output path
        dstpath = f'{model.config.get_output_dir()}/src/firmware/nnet_utils/activation_tables'
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
        self.__write_exp_tables_stable(model, dstpath)
        self.__write_invert_tables_stable(model, dstpath)
        self.__write_exp_tables_latency(model, dstpath)
        self.__write_invert_tables_latency(model, dstpath)
        self.__write_exp_table_legacy(model, dstpath)
        self.__write_invert_table_legacy(model, dstpath)

    def write_generated_code(self, model):
        """Write the generated code (nnet_code_gen.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        path = f'{model.config.get_output_dir()}/src/firmware/nnet_utils/nnet_code_gen.h'
        f = open(path)
        contents = f.readlines()
        f.close()
        f = open(path, 'w')
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
            f.write(newline)
        f.close()

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

        if model.config.get_writer_config().get('WriteTar', False):
            tar_path = model.config.get_output_dir() + '.tar.gz'
            if os.path.exists(tar_path):
                os.remove(tar_path)
            with tarfile.open(model.config.get_output_dir() + '.tar.gz', mode='w:gz') as archive:
                archive.add(model.config.get_output_dir(), recursive=True)

    def write_hls(self, model):
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
        self.write_generated_code(model)
        self.write_yml(model)
        self.write_tar(model)
