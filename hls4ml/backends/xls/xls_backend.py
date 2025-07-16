import os
import sys
import subprocess, shlex
from warnings import warn

import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import (
    Dense,
    Layer,
)
from hls4ml.model.types import IntegerPrecisionType, NamedType

class XLSBackend(FPGABackend):
    def __init__(self):
        super().__init__('XLS')
        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        # TODO: implement this
        pass

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        optimization_passes = [
            'infer_precision_types',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        vivado_types = [
            'xls:transform_types',
        ]
        vivado_types_flow = register_flow('specific_types', vivado_types, requires=[init_flow], backend=self.name)

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

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
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
    ):
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

        return config
    

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