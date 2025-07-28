# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import List, Any, TYPE_CHECKING
from numpy.typing import NDArray
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer
    from subprocess import CompletedProcess

import os, sys
import re
import subprocess, shlex
import numpy as np
from warnings import warn
from fxpmath import Fxp

from hls4ml.backends import FPGABackend
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.flow import register_flow
from hls4ml.model.attributes import ChoiceAttribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.layers import (
    Dense,
    Layer,
    Activation,
    Softmax
)
from hls4ml.utils import attribute_descriptions as descriptions
from hls4ml.model.types import IntegerPrecisionType, NamedType

class XLSBackend(FPGABackend):
    def __init__(self) -> None:
        super().__init__('XLS')
        self._writer_flow = ''
        self._default_flow = ''

        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self) -> None:
        pass
        # all_layers = [
        #     Layer,
        #     Dense,
        #     Activation,
        #     Softmax,
        # ]

        # for layer in all_layers:
        #     attrs = self.attribute_map.get(layer, [])
        #     attrs.append(
        #         ConfigurableAttribute('skip', value_type=bool, default=True, description=descriptions.softmax_skip)
        #     )
        #     self.attribute_map[layer] = attrs

    def _register_flows(self) -> None:
        initializers: list = self._get_layer_initializers()
        init_flow: str = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        optimization_passes = [
            'infer_precision_types',
        ]
        optimization_flow: str = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        xls_attributes = [
            'xls:build_attr',
        ]
        xls_attributes_flow: str = register_flow('specific_attributes', xls_attributes, requires=[optimization_flow], backend=self.name)

        xls_optimization_passes = [
            'xls:merge_dense_relu',
        ]
        xls_optimization_passes_flow: str = register_flow('merge_dense_relu_layers', xls_optimization_passes, requires=[xls_attributes_flow], backend=self.name)

        writer_passes = ['make_stamp', 'xls:write_hls'] 
        self._writer_flow = register_flow('write', writer_passes, requires=['xls:ip'], backend=self.name) 

        all_passes: list = get_backend_passes(self.name)

        #TODO: what is this extras structure here
        extras = [
            # Ideally this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + writer_passes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            optimization_flow,
            xls_attributes_flow,
            xls_optimization_passes_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self) -> str:
        return self._default_flow

    def get_writer_flow(self) -> str:
        return self._writer_flow
    
    def create_initial_config(
        self,
        part='xcvu13p-flga2577-2-e',
        clock_period=5,
        clock_uncertainty='12.5%',
        io_type='io_parallel',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
        tb_output_stream='both',
        **_,
    ) -> dict[str, Any]:
        """Create initial configuration of the Vivado backend.

        Args:
            part (str, optional): The FPGA part to be used. Defaults to 'xcvu13p-flga2577-2-e'.
            clock_period (int, optional): The clock period. Defaults to 5.
            clock_uncertainty (str, optional): The clock uncertainty. Defaults to 12.5%.
            io_type (str, optional): Type of implementation used. One of
                'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
            namespace (str, optional): If defined, place all generated code within a namespace. Defaults to None.
            write_weights_txt (bool, optional): If True, writes weights to .txt files which speeds up compilation.
                Defaults to True.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.
            tb_output_stream (str, optional): Controls where to write the output. Options are 'stdout', 'file' and 'both'.
                Defaults to 'both'.

        Returns:
            dict: initial configuration.
        """
        config = {}

        config['Part'] = part if part is not None else 'xcvu13p-flga2577-2-e'
        config['ClockPeriod'] = clock_period if clock_period is not None else 5
        config['ClockUncertainty'] = clock_uncertainty if clock_uncertainty is not None else '12.5%'
        config['IOType'] = io_type if io_type is not None else 'io_parallel'
        config['HLSConfig'] = {}
        config['WriterConfig'] = {
            'Namespace': namespace,
            'WriteWeightsTxt': write_weights_txt,
            'WriteTar': write_tar,
            'TBOutputStream': tb_output_stream,
        }
        #TODO: update to a better way to access the bazel-vin project
        config['xls_bazel_bin_path'] = '$HOME/xls/bazel-bin'

        return config
    
    def _get_backend_exec_path(self, model: ModelGraph) -> str:
        if 'linux' in sys.platform:
            path: str = os.path.expandvars(model.config.get_config_value('xls_bazel_bin_path'))
            if os.path.isdir(path) == 0:
                raise Exception('XLS is expected to be installed in your $HOME dir. We are looking for `$HOME/xls/bazel-bin`')
        return path

    #TODO: this return value conflicts with the expected return value in ModelGraph of compile()
    def compile(self, model: ModelGraph) -> None:

        path = self._get_backend_exec_path(model)

        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')
        kernel_name = model.config.get_project_name()

        ## Generate IR
        with open(f'{kernel_name}.ir', 'w') as ir_file:
            gen_cmd = [ 
                f'{path}/xls/dslx/ir_convert/ir_converter_main',
                f'--top={kernel_name}',
                f'{kernel_name}.x'
            ]
            subprocess.run(gen_cmd, check=True, stdout=ir_file)
        ## Optimize IR
        with open(f'{kernel_name}.opt.ir', 'w') as opt_file:
            opt_cmd = [ 
                f'{path}/xls/tools/opt_main',
                f'{kernel_name}.ir'
            ]
            subprocess.run(opt_cmd, check=True, stdout=opt_file)

        os.chdir(curr_dir)

    def predict(self, model: ModelGraph, x: np.floating | NDArray[np.floating[Any]]) -> list[NDArray[np.floating]]:

        def _interpret_input(model: ModelGraph, 
                             path: str, 
                             x_list: NDArray[np.floating], 
                             n_samples: int, 
                             n_inputs: int, 
                             input_width: int, 
                             input_frac: int) -> CompletedProcess[str]:
            newline = ''
            for i in range(n_samples):
                if n_inputs == 1:
                    inp = [np.asarray(x_list[i])]
                else:
                    inp = [np.asarray(xj) for xj in x_list[i]]
                newline += '['
                fxp_x: list[NDArray[np.int_]] = Fxp(inp, signed=True, n_word=input_width, n_frac=input_frac).raw() 
                if n_inputs == 1:
                    newline += f'bits[{input_width}]:{fxp_x[0][0]}'
                else:
                    for i, inp in enumerate(fxp_x):
                        newline += f'bits[{input_width}]:{inp}'
                        if i < len(fxp_x) - 1:
                            newline += ','
                newline += ']\n'

            # run command
            interpret_cmd = [ 
                f'{path}/xls/tools/eval_ir_main',
                f'../firmware/{model.config.get_project_name()}.opt.ir',
                f'--input_file=-'
            ]
            result = subprocess.run(
                interpret_cmd,
                input=newline,        
                text=True,             
                check=True,
                stdout=subprocess.PIPE,
            )
            return result

        def _format_output(result: CompletedProcess[str]) -> list:
            hex_pat = re.compile(r"0x([0-9A-Fa-f]+)")
            output_type_pat = re.compile(r"bits\[(\d+)\]")

            # process output
            rows = []
            for line in result.stdout.splitlines():
                raw_outputs = hex_pat.findall(line)
                m = output_type_pat.search(line)
                output_width = int(m.group(1))
                if not raw_outputs:
                    continue
                int_outputs = [int(o, output_width) for o in raw_outputs]

                # signed interpretation w/ 2's complement
                sign_bit = 1 << (output_width - 1)
                full_mask = 1 << output_width
                sint_output = [(v - full_mask) if (v & sign_bit) else v for v in int_outputs]

                rows.append([sint_output])

            return rows

        def _go_to_original_type(rows: list, 
                                 n_samples: int, 
                                 n_outputs: int, 
                                 python_input_type: np.dtype[np.floating], 
                                 scale) -> list[NDArray[np.floating]]:
            output = np.array(rows, dtype=np.int32)
            output = output.astype(python_input_type) / scale
            output = [np.asarray([output[i_sample][i_output] for i_sample in range(n_samples)]) for i_output in range(n_outputs)]
            return output

        def _correct_dims(results_floats: list[NDArray[np.floating]], n_samples: int, n_outputs: int) -> list[NDArray[np.floating]]:
            if n_samples == 1 and n_outputs == 1:
                return result_floats[0][0]
            elif n_outputs == 1:
                return result_floats[0]
            elif n_samples == 1:
                return [output_i[0] for output_i in result_floats]
            else:
                return result_floats

        path: str = self._get_backend_exec_path(model)
        layers: list[Layer] = list(model.get_layers())

        # Extract dimensions
        n_samples: int = model._compute_n_samples(x)
        n_inputs: int = list(layers[0].get_output_variable().get_shape())[0][1] # Get input dimensions
        n_outputs: int = len(model.get_output_variables())

        # Extract type
        input_width: int = list(layers[0].get_layer_precision().items())[0][1].precision.width
        input_frac: int = input_width - list(layers[0].get_layer_precision().items())[0][1].precision.integer
        output_width: int = list(layers[len(layers)-1].get_layer_precision().items())[0][1].precision.width
        output_frac: int = output_width - list(layers[len(layers)-1].get_layer_precision().items())[0][1].precision.integer

        # extract python type (float/double)
        if isinstance(x, np.ndarray):
            python_input_type: np.dtype[np.floating] = x[0].dtype
        else:
            python_input_type: np.dtype[np.floating]  = x.dtype
        
        if n_samples == 1 and n_inputs == 1 and isinstance(x, np.floating):
            x_list: NDArray[np.floating] = np.array([x], dtype=x.dtype)
        elif isinstance(x, np.ndarray): 
            x_list: NDArray[np.floating] = x

        # Change dirs
        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/predictions')

        # Result processing pipeling
        result = _interpret_input(model, path, x_list, n_samples, n_inputs, input_width, input_frac)
        os.chdir(curr_dir)
        result_formatted = _format_output(result)
        result_floats: list[NDArray[np.floating]] = _go_to_original_type(result_formatted, 
            n_samples, 
            n_outputs, 
            python_input_type, 
            scale=2 ** output_frac
        )
        result_corrected_dims: list[NDArray[np.floating]] = _correct_dims(result_floats, n_samples, n_outputs)
        return result_corrected_dims


    #TODO: use the other flags
    def build(
        self,
        model,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
        codegen_flags='--delay_model=asap7 --fifo_module="xls_fifo_wrapper" --clock_period_ps=100 --reset=reset',
    ):
        if 'linux' in sys.platform:
            path = os.path.expandvars(model.config.get_config_value('xls_bazel_bin_path'))
            if os.path.isdir(path) == 0:
                raise Exception('XLS is expected to be installed in your $HOME dir. We are looking for `$HOME/xls/bazel-bin`')

        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')
        kernel_name = model.config.get_project_name()

        ## Generate RTL
        with open(f'{kernel_name}.sv', 'w') as synth_file:
            flags = shlex.split(codegen_flags)
            synth_cmd = [ 
                f'{path}/xls/tools/codegen_main',
                *flags,
                f'{kernel_name}.opt.ir',
            ]
            subprocess.run(synth_cmd, check=True, stdout=synth_file)

        os.chdir(curr_dir)

        #TODO: return parsed report
        # return parse_vivado_report(model.config.get_output_dir())
