# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import List, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph

import os, sys
import re
import subprocess, shlex
import numpy as np
from warnings import warn
from fxpmath import Fxp

from hls4ml.backends import FPGABackend
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import (
    Dense,
    Layer,
)
from hls4ml.model.types import IntegerPrecisionType, NamedType

class XLSBackend(FPGABackend):
    def __init__(self) -> None:
        super().__init__('XLS')
        self._writer_flow = ''
        self._default_flow = ''

        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        # TODO: implement this
        pass

    def _register_flows(self) -> None:
        initializers = self._get_layer_initializers()
        init_flow: str = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        optimization_passes = [
            'infer_precision_types',
        ]
        optimization_flow: str = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        vivado_types = [
            'xls:transform_types',
        ]
        vivado_types_flow: str = register_flow('specific_types', vivado_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'xls:write_hls'] 
        self._writer_flow = register_flow('write', writer_passes, requires=['xls:ip'], backend=self.name)   # TODO: what is this xls:ip

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + templates
            + writer_passes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            optimization_flow,
            vivado_types_flow,
            template_flow,
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
    
    #TODO: this return value conflicts with the expected return value in ModelGraph of compile()
    def compile(self, model: ModelGraph):

        if 'linux' in sys.platform:
            path = os.path.expandvars(model.config.get_config_value('xls_bazel_bin_path'))
            if os.path.isdir(path) == 0:
                raise Exception('XLS is expected to be installed in your $HOME dir. We are looking for `$HOME/xls/bazel-bin`')

        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')
        kernel_name = model.config.get_project_name()

        # ## Run interpreter
        # interpreter_cmd = [ 
        #     f'{path}/xls/dslx/interpreter_main',
        #     f'{kernel_name}.x'
        # ]
        # subprocess.run(interpreter_cmd, check=True)

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

    def predict(self, model: ModelGraph, x):

        if 'linux' in sys.platform:
            path = os.path.expandvars(model.config.get_config_value('xls_bazel_bin_path'))
            if os.path.isdir(path) == 0:
                raise Exception('XLS is expected to be installed in your $HOME dir. We are looking for `$HOME/xls/bazel-bin`')
        
        n_samples = model._compute_n_samples(x)
        n_inputs = len(model.get_input_variables())
        n_outputs = len(model.get_output_variables())

        # extract type info
        if n_inputs == 1:
            input_type = x.dtype
        else:
            input_type = x[0].dtype
        
        output = []
        if n_samples == 1 and n_inputs == 1:
            x = [x]

        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/predictions')
        # write input file
        scale = 2 ** 10
        with open(f'input.txt', 'w') as input_file:
            newline = ''
            for i in range(n_samples):
                newline += '['
                # predictions: list[ndarray[tuple[int], dtype[float64]]] = [np.zeros(yj.size()) for yj in model.get_output_variables()]
                fxp_x = Fxp(x[i], signed=True, n_word=16, n_frac=10).raw() 
                if n_inputs == 1:
                    #TODO: not always 16 bits
                    newline += f'bits[16]:{fxp_x[0]}'
                else:
                    for i, inp in enumerate(fxp_x[i]):
                        newline += f'bits[16]:{inp}'
                    if i < len(fxp_x[i]) - 1:
                        newline += ','
                newline += ']\n'
            input_file.write(newline)

        # predict to output
        interpret_cmd = [ 
            f'{path}/xls/tools/eval_ir_main',
            f'../firmware/{model.config.get_project_name()}.opt.ir',
            f'--input_file=input.txt'
        ]
        result = subprocess.run(interpret_cmd, check=True, stdout=subprocess.PIPE, text=True)
        

        # extract from output file
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
            int_outputs = [int(o, 16) for o in raw_outputs]

            # signed interpretation w/ 2's complement
            sign_bit = 1 << (output_width - 1)
            full_mask = 1 << output_width
            sint_output = [(v - full_mask) if (v & sign_bit) else v for v in int_outputs]

            rows.append([sint_output])

        # scale back from fixed point
        output = np.array(rows, dtype=np.int32)
        output = output.astype(input_type) / scale
        output = [np.asarray([output[i_sample][i_output] for i_sample in range(n_samples)]) for i_output in range(n_outputs)]

        if n_samples == 1 and n_outputs == 1:
            print('A')
            return output[0][0]
        elif n_outputs == 1:
            print(output[0].shape)
            print('B', output)
            return output[0]
        elif n_samples == 1:
            print('C')
            return [output_i[0] for output_i in output]
        else:
            print('D')
            return output


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
    ):
        # TODO: include vivado & understand exactly what this does
        # TODO: Use the real config
        config = {
            'output_dir': 'build',
            'workspace_path': '$HOME/workspace/xls4nn',
            'xls_bazel_bin_path': '$HOME/xls/bazel-bin',
            'kernel_name': 'proc_jet_tagging_dense',
            'codegen_flags': '--delay_model=asap7 --fifo_module="xls_fifo_wrapper" --clock_period_ps=5000 --pipeline_stages=2 --reset=reset'
        }

        if 'linux' in sys.platform:
            workspace = os.path.expandvars(config['workspace_path'])
            path = os.path.expandvars(config['xls_bazel_bin_path'])
            if os.path.isdir(path) == 0:
                raise Exception('XLS is expected to be installed in your $HOME dir. We are looking for `$HOME/xls/bazel-bin`')

        curr_dir = os.getcwd()
        os.chdir(config['output_dir'])
        kernel_name = config['kernel_name']

        ## Run interpreter
        interpreter_cmd = [ 
            f'{path}/xls/dslx/interpreter_main',
            f'{workspace}/kernels/end2end/{kernel_name}.x'
        ]
        subprocess.run(interpreter_cmd, check=True)

        ## Generate IR
        with open(f'{kernel_name}.ir', 'w') as ir_file:
            gen_cmd = [ 
                f'{path}/xls/dslx/ir_convert/ir_converter_main',
                f'--top={kernel_name}',
                f'{workspace}/kernels/end2end/{kernel_name}.x'
            ]
            subprocess.run(gen_cmd, check=True, stdout=ir_file)

        ## Optimize IR
        with open(f'{kernel_name}.opt.ir', 'w') as opt_file:
            opt_cmd = [ 
                f'{path}/xls/tools/opt_main',
                f'{kernel_name}.ir'
            ]
            subprocess.run(opt_cmd, check=True, stdout=opt_file)

        ## Generate RTL
        with open(f'{kernel_name}.sv', 'w') as opt_file:
            flags = shlex.split(config["codegen_flags"])
            rtl_cmd = [ 
                f'{path}/xls/tools/codegen_main',
                *flags,
                f'{kernel_name}.opt.ir',
            ]
            subprocess.run(rtl_cmd, check=True, stdout=opt_file)

        os.chdir(curr_dir)

        #TODO: return parsed report
        # return parse_vivado_report(model.config.get_output_dir())

    # TODO: What do the layer optimizers achieve?
    # @layer_optimizer(Layer)
    # def init_base_layer(self, layer):
    #     reuse_factor = layer.model.config.get_reuse_factor(layer)
    #     layer.set_attr('reuse_factor', reuse_factor)

    #     target_cycles = layer.model.config.get_target_cycles(layer)
    #     layer.set_attr('target_cycles', target_cycles)

    # @layer_optimizer(Dense)
    # def init_dense(self, layer):
    #     index_t = IntegerPrecisionType(width=1, signed=False)
    #     compression = layer.model.config.get_compression(layer)
    #     if layer.model.config.is_resource_strategy(layer):
    #         n_in, n_out = self.get_layer_mult_size(layer)
    #         self.set_target_reuse_factor(layer)
    #         self.set_closest_reuse_factor(layer, n_in, n_out)
    #         if compression:
    #             layer.set_attr('strategy', 'compressed')
    #             index_t = layer.get_weights('weight').type.index_precision
    #         else:
    #             layer.set_attr('strategy', 'resource')
    #     elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
    #         use_resource_instead = False
    #         if layer.get_attr('reuse_factor', 1) == 1:
    #             print(
    #                 f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}". '
    #                 'Using "resource" strategy instead.'
    #             )
    #             use_resource_instead = True
    #         n_in, n_out = self.get_layer_mult_size(layer)
    #         self.set_target_reuse_factor(layer)
    #         if use_resource_instead:
    #             self.set_closest_reuse_factor(layer, n_in, n_out)
    #             layer.set_attr('strategy', 'resource')
    #         else:
    #             self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
    #             layer.set_attr('strategy', 'resource_unrolled')
    #     else:
    #         layer.set_attr('strategy', 'latency')
    #     layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', index_t))