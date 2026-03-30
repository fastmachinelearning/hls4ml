# Typing imports
from __future__ import annotations  # makes all annotations into strings

from pathlib import Path
from typing import Any, TYPE_CHECKING

import xls
from numpy.typing import NDArray, ArrayLike

from hls4ml.backends.xls.xls_types import float_to_significand
from hls4ml.model.types import FixedPrecisionType

if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph

import os, sys
import subprocess, shlex
import numpy as np
from warnings import warn

from hls4ml.backends import FPGABackend
from hls4ml.model.optimizer import get_backend_passes
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_xls_report


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
            # TODO: we fix table sizes in BuildTables, it should be merged into fix_softmax_table_size.
            # 'xls:fix_softmax_table_size',
            'xls:skip_softmax',
            'infer_precision_types',
        ]
        optimization_flow: str = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        xls_attributes = [
            'xls:build_attr',
        ]
        xls_attributes_flow: str = register_flow('specific_attributes', xls_attributes, requires=[optimization_flow],
                                                 backend=self.name)

        xls_build_graph_ir = [
            'xls:build_tables',
        ]
        xls_build_graph_ir_flow: str = register_flow('build_tables_ir', xls_build_graph_ir,
                                                     requires=[xls_attributes_flow], backend=self.name)

        writer_passes = ['make_stamp', 'xls:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['xls:ip'], backend=self.name)

        all_passes: list = get_backend_passes(self.name)

        # TODO: what is this extras structure here
        extras = [
            # Ideally, this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
               not in initializers
               + writer_passes
               + optimization_passes
               + xls_attributes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            optimization_flow,
            xls_attributes_flow,
            xls_build_graph_ir_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self) -> str:
        return self._default_flow

    def get_writer_flow(self) -> str:
        return self._writer_flow

    def create_initial_config(
            self,
            part='xcu250-figd2104-2L-e',
            clock_period=5,
            clock_uncertainty='12.5%',
            io_type='io_parallel',
            namespace=None,
            write_weights_txt=True,
            write_tar=False,
            tb_output_stream='both',
            **_,
    ) -> dict[str, Any]:
        """Create an initial configuration of the XLS backend.

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
        # TODO: update to a better way to access the bazel-bin project
        config['xls_bazel_bin_path'] = '$HOME/xls/bazel-bin'

        return config

    def _get_backend_exec_path(self, model: ModelGraph) -> str:
        if 'linux' in sys.platform:
            path: str = os.path.expandvars(model.config.get_config_value('xls_bazel_bin_path'))
            if os.path.isdir(path) == 0:
                raise Exception(
                    'XLS is expected to be installed in your $HOME dir. We are looking for `$HOME/xls/bazel-bin`')
        return path

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

    @staticmethod
    def _float_to_xls_ir(x: np.floating[Any] | NDArray[np.floating[Any]],
                         precision: FixedPrecisionType) -> xls.Value:
        if np.isscalar(x):
            significand = float_to_significand(x, precision)
            bits = xls.Value.make_sbits(bit_count=precision.width, val=significand)
            return bits
        else:
            return xls.Value.make_array([XLSBackend._float_to_xls_ir(item, precision) for item in x])

    @staticmethod
    def _xls_ir_to_float(x: xls.Value, precision: FixedPrecisionType,
                         dtype: np.typing.DTypeLike) -> ArrayLike:
        match x.get_kind():
            case xls.c_api.ValueKind.BITS:
                return x.get_bits().to_int64() / (2 ** precision.fractional)
            case xls.c_api.ValueKind.ARRAY:
                return np.asarray([
                    XLSBackend._xls_ir_to_float(x.get_element(i), precision, dtype)
                    for i in range(x.get_element_count())
                ], dtype=dtype)
            case _:
                raise ValueError(f'Unexpected output type: {x.get_kind()}')

    # TODO call it in compile() and save to model attribute
    @staticmethod
    def _get_top_function(model: ModelGraph):
        project_dir = model.config.get_output_dir()
        project_name = model.config.get_project_name()
        ir_path = Path(project_dir) / 'firmware' / f'{project_name}.opt.ir'
        ir_text = open(ir_path, 'r').read()
        pkg = xls.Package.parse_ir(ir_text)
        fn = pkg.get_function(f'__{project_name}__{project_name}')
        jit = fn.to_jit()

        input_vars = model.get_input_variables()
        output_vars = model.get_output_variables()

        def top_function(*args):
            assert len(args) == len(input_vars), f'Expected {len(input_vars)} inputs, got {len(args)}'
            ir_input = [
                XLSBackend._float_to_xls_ir(x, var.type.precision)
                for x, var in zip(args, input_vars)
            ]
            ir_output = jit.run(ir_input)
            if len(output_vars) == 1:
                return XLSBackend._xls_ir_to_float(ir_output, output_vars[0].type.precision,
                                                   dtype=np.asarray(args[0]).dtype)
            else:
                raise ValueError(f'Only one output variable is supported, got {len(output_vars)}')

        return top_function

    def predict(self, model: ModelGraph, x: np.floating | NDArray[np.floating[Any]]) -> list[NDArray[np.floating]]:
        top_function = self._get_top_function(model)
        n_samples = model._compute_n_samples(x)
        n_inputs = len(model.get_input_variables())
        n_outputs = len(model.get_output_variables())

        output = []
        if n_samples == 1 and n_inputs == 1:
            if np.isscalar(x):
                x = [x]
            if np.isscalar(x[0]):
                x = [x]

        for i in range(n_samples):
            if n_inputs == 1:
                inp = [np.asarray(x[i])]
            else:
                inp = [np.asarray(xj[i]) for xj in x]
            predictions = top_function(*inp)
            output.append(predictions)

        return np.asarray(output)

    def build(
            self,
            model: ModelGraph,
            reset: bool = True,
            pr: bool = False,
    ) -> dict:
        """ Builds the RTL (SystemVerilog) code and uses Vivado to return the resource utilization.

        Args:
            model (ModelGraph): the hls4ml model.
            reset (bool): the reset synthesis option
            clk_period (int):  clock period in nanoseconds (e.g., 5 ns => 1,000 / 5 = 200 MHz)
            pr (bool): place and route option
        """

        if 'linux' in sys.platform:
            path = os.path.expandvars(model.config.get_config_value('xls_bazel_bin_path'))
            if os.path.isdir(path) == 0:
                raise Exception(
                    'XLS is expected to be installed in your $HOME dir. We are looking for `$HOME/xls/bazel-bin`')

        def build_flags() -> str:
            flags = f'--delay_model=asap7 --fifo_module="xls_fifo_wrapper" --clock_period_ps={model.config.get_config_value("ClockPeriod") * 1000} '
            if reset:
                flags += '--reset=reset'
            return flags

        def build_vivado_flags() -> list[str]:
            f = [
                '-mode', 'batch',
                '-nolog',
                '-nojournal',
                '-source', './build_prj.tcl',
                '-tclargs',
                f'firmware/{model.config.get_project_name()}.sv',
                f'{model.config.get_config_value("Part")}',
                f'{model.config.get_config_value("ClockPeriod")}'
            ]
            if pr:
                f += '--pr'
            return f

        curr_dir: str = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')
        kernel_name = model.config.get_project_name()

        # Generate RTL
        codegen_flags: str = build_flags()
        with open(f'{kernel_name}.sv', 'w') as synth_file:
            flags = shlex.split(codegen_flags)
            synth_cmd = [
                f'{path}/xls/tools/codegen_main',
                *flags,
                f'{kernel_name}.opt.ir',
            ]
            subprocess.run(synth_cmd, check=True, stdout=synth_file)

        # Run Vivado for resource report
        os.chdir(curr_dir)
        os.chdir(f'{model.config.get_output_dir()}')

        vivado_command: list[str] = ['vivado'] + build_vivado_flags()
        subprocess.run(vivado_command, check=True)

        os.chdir(curr_dir)
        return parse_xls_report(model.config.get_output_dir())
