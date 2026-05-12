# Typing imports
from __future__ import annotations  # makes all annotations into strings

import functools
import importlib
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING, Dict, Iterable
from numpy.typing import NDArray, ArrayLike

from hls4ml.backends.xls.xls_types import float_to_significand
from hls4ml.model.types import FixedPrecisionType

if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph

import os
import subprocess
import numpy as np
from warnings import warn

from hls4ml.backends import FPGABackend
from hls4ml.model.optimizer import get_backend_passes
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_xls_report


@functools.lru_cache(maxsize=1)
def import_xls():
    try:
        return importlib.import_module('xls')
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "XLS backend requires optional dependency 'xls'. "
            "Please install hls4ml with XLS extras (or install package 'xls')."
        ) from e


class XLSBackend(FPGABackend):
    def __init__(self) -> None:
        super().__init__('XLS')
        self._writer_flow = ''
        self._default_flow = ''

        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self) -> None:
        pass

    def _register_flows(self) -> None:
        initializers: list = self._get_layer_initializers()
        init_flow: str = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        quantization_passes = [
            # 'xls:merge_batch_norm_quantized_tanh',
            # 'xls:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
            'xls:xnor_pooling',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            'xls:remove_final_reshape',
            'xls:inplace_parallel_reshape',
            'xls:skip_softmax',
            'infer_precision_types',
        ]
        optimization_flow: str = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        xls_attributes = [
            'xls:build_tables',
            'xls:build_attr',
        ]
        xls_attributes_flow: str = register_flow('xls', xls_attributes, requires=[optimization_flow],
                                                 backend=self.name)

        # TODO: stamp is currently unused, shall we add it to myproject.x, myproject.ir, myproject.opt.ir, ...?
        # In other backends, this is used to generate myproject-$STAMP.so.
        # In XLS, .opt.ir file plays the same role as .so
        # It is unclear whether we should copy or rename myproject.opt.ir to myproject-$STAMP.opt.ir.
        writer_passes = ['make_stamp', 'xls:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['xls:ip'], backend=self.name)

        # Passed that are irrelevant for XLS
        ignored_passes = [f'xls:{opt_pass}' for opt_pass in [
            # io_stream only:
            'reshape_stream',
            'inplace_stream_flatten',
            'repack_function_template',
            'clone_output',
            'clone_function_template',
            # HGQ passes, not implemented:
            'process_fixed_point_quantizer_layer',
            'fixedpointquantizer_function_template',
            'unarylut_function_template',
            # Embedding
            'embedding_config_template',
            'embedding_function_template',
            # we fix table sizes in xls:build_tables using a different method
            'fix_softmax_table_size',
            # BRAM not supported
            'register_bram_weights',
        ]]

        all_passes: list = get_backend_passes(self.name)

        extras = [
            # Ideally, this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
               not in initializers
               + quantization_passes
               + optimization_passes
               + xls_attributes
               + writer_passes
               + ignored_passes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            quantization_flow,
            optimization_flow,
            xls_attributes_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self) -> str:
        return self._default_flow

    def get_writer_flow(self) -> str:
        return self._writer_flow

    @staticmethod
    def _to_xls_clock_period_ps(clock_period) -> int:
        """Convert nanoseconds to picoseconds."""
        return int(float(clock_period) * 1000)

    @staticmethod
    def _to_xls_clock_margin_percent(clock_uncertainty: str) -> int:
        """Convert ClockUncertainty string to integer XLS option clock_margin_percent"""
        assert isinstance(clock_uncertainty, str) and clock_uncertainty.endswith('%'), \
            f'Clock uncertainty must be in percentage format, got {clock_uncertainty}'
        return math.ceil(float(clock_uncertainty.strip('%')))

    @staticmethod
    def _percent_to_float(percent: str) -> float:
        """Convert a string representing a percentage to a float."""
        assert isinstance(percent, str) and percent.endswith('%'), \
            f'Clock uncertainty must be in percentage format, got {percent}'
        return float(percent.strip('%')) / 100

    def create_initial_config(
            self,
            part='xcu250-figd2104-2L-e',
            clock_period=5,
            clock_uncertainty='12.5%',
            io_type='io_parallel',
            write_tar=False,
            xls_codegen_flags=None,
            **kwargs,
    ) -> dict[str, Any]:
        """Create an initial configuration of the XLS backend.

        Args:
            part (str, optional): The FPGA part to be used. Defaults to 'xcvu13p-flga2577-2-e'.
            clock_period (int, optional): The clock period. Defaults to 5.
            clock_uncertainty (str, optional): The clock uncertainty. Defaults to 12.5%.
            io_type (str, optional): Type of implementation used. Only 'io_parallel' is currently supported.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.
            xls_codegen_flags (dict, optional): Flags to pass to the XLS codegen. Defaults to None.

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
            'WriteTar': write_tar,
        }

        # Set default flags to mimic codegen_main executable behavior
        config['XLSCodegenFlags'] = xls_codegen_flags if xls_codegen_flags is not None else {
            'delay_model': 'asap7',
            'generator': 'pipeline',
            'use_system_verilog': True,
            'flop_inputs': True,
            'flop_outputs': True,
            'max_inline_depth': 5,
            'flop_single_value_channels': True,
            # convert nanoseconds to picoseconds
            'clock_period_ps': self._to_xls_clock_period_ps(config['ClockPeriod']),
            # NB: XLS needs integer percents
            'clock_margin_percent': self._to_xls_clock_margin_percent(config['ClockUncertainty']),
        }

        for arg in kwargs:
            warn(f'WARNING: Unknown argument {arg} for XLS backend will be ignored.')

        return config

    @staticmethod
    def _ir_top_function_name(model: ModelGraph):
        xls = import_xls()
        name = model.config.get_project_name()
        return xls.mangle_dslx_name(module_name=name, function_name=name)

    def compile(self, model: ModelGraph) -> None:
        xls = import_xls()
        io_type = model.config.get_config_value('IOType')
        if io_type != 'io_parallel':
            raise NotImplementedError(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')
        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')
        kernel_name = model.config.get_project_name()

        ir_text = xls.c_api.convert_dslx_path_to_ir(path=f'{kernel_name}.x')
        with open(f'{kernel_name}.ir', 'w') as ir_file:
            ir_file.write(ir_text)

        opt_ir_text = xls.optimize_ir(ir=ir_text, top=XLSBackend._ir_top_function_name(model))
        with open(f'{kernel_name}.opt.ir', 'w') as opt_ir_file:
            opt_ir_file.write(opt_ir_text)

        os.chdir(curr_dir)

    @staticmethod
    def _float_to_xls_ir(x: np.floating[Any] | NDArray[np.floating[Any]],
                         precision: FixedPrecisionType):
        xls = import_xls()
        if np.isscalar(x):
            significand = float_to_significand(x, precision)
            bits = xls.Value.make_sbits(bit_count=precision.width, val=significand)
            return bits
        else:
            return xls.Value.make_array([XLSBackend._float_to_xls_ir(item, precision) for item in x])

    @staticmethod
    def _bits_to_int(bits, signed: bool = True) -> int:
        # bits: xls.Bits
        n = bits.get_bit_count()
        if n <= 64:
            return bits.to_int64()
        value = int.from_bytes(bits.to_bytes(), byteorder='little', signed=False)
        value &= (1 << n) - 1
        if signed and (bits.get_bit(n - 1) == 1):
            value -= (1 << n)
        return value

    @staticmethod
    def _xls_ir_to_float(x, precision: FixedPrecisionType | Iterable[FixedPrecisionType],
                         dtype: np.typing.DTypeLike) -> ArrayLike | tuple[ArrayLike, ...]:
        xls = import_xls()
        # x: xls.Value
        match x.get_kind():
            case xls.c_api.ValueKind.BITS:
                assert isinstance(precision, FixedPrecisionType), \
                    f'Precision must be FixedPrecisionType, got {type(precision)}'
                return XLSBackend._bits_to_int(x.get_bits()) / (2 ** precision.fractional)
            case xls.c_api.ValueKind.ARRAY:
                return np.asarray([
                    XLSBackend._xls_ir_to_float(x.get_element(i), precision, dtype)
                    for i in range(x.get_element_count())
                ], dtype=dtype)
            case xls.c_api.ValueKind.TUPLE:
                precision = tuple(precision)
                assert len(precision) == x.get_element_count(), \
                    f'Precision mismatch for tuple: {len(precision)} != {x.get_element_count()}'
                return tuple(
                    XLSBackend._xls_ir_to_float(x.get_element(i), precision[i], dtype)
                    for i in range(x.get_element_count())
                )
            case _:
                raise ValueError(f'Unexpected output type: {x.get_kind()}')

    # TODO call it in compile() and save to model attribute
    @staticmethod
    def get_top_function(model: ModelGraph, x: np.floating | NDArray[np.floating[Any]]):
        xls = import_xls()
        project_dir = model.config.get_output_dir()
        project_name = model.config.get_project_name()
        ir_path = Path(project_dir) / 'firmware' / f'{project_name}.opt.ir'
        ir_text = ir_path.read_text()
        pkg = xls.Package.parse_ir(ir_text)
        fn = pkg.get_function(XLSBackend._ir_top_function_name(model))
        jit = fn.to_jit()

        input_vars = model.get_input_variables()
        output_vars = model.get_output_variables()

        def top_function(*args):
            assert len(args) == len(input_vars) + len(output_vars), \
                f'Expected {len(input_vars)} inputs and {len(output_vars)} outputs, got {len(args)}'
            inputs = args[:len(input_vars)]
            outputs = args[len(input_vars):]
            ir_input = [
                XLSBackend._float_to_xls_ir(np.asarray(x).reshape(var.shape), var.type.precision)
                for x, var in zip(inputs, input_vars)
            ]
            ir_output = jit.run(ir_input)

            out_precision = [output_var.type.precision for output_var in output_vars]
            if len(out_precision) == 1:
                out_precision = out_precision[0]
            dtype = np.asarray(inputs[0]).dtype
            output = XLSBackend._xls_ir_to_float(ir_output, out_precision, dtype)
            # This is the case when len(output_vars) == 1
            if not isinstance(output, tuple):
                output = (output,)
            for i in range(len(output_vars)):
                outputs[i][:] = np.reshape(output[i], -1)

        # TODO: this duplicates ModelGraph._get_top_function().
        # NB: ctype is not used in XLS, but it is required by ModelGraph._predict
        x0 = x[0] if isinstance(x, (list, tuple)) else x
        if np.asarray(x0).dtype in [np.single, np.float32]:
            ctype = np.float32
        elif np.asarray(x0).dtype in [np.double, np.float64]:
            ctype = np.float64
        else:
            raise Exception(
                'Invalid type ({}) of numpy array. Supported types are: single, float32, double, float64, float_.'.format(
                    np.asarray(x0).dtype
                )
            )

        return top_function, ctype

    def build(
            self,
            model: ModelGraph,
            reset: bool | None = None,
            pr: bool = False,
    ) -> dict:
        """ Builds the RTL (SystemVerilog) code and uses Vivado to return the resource utilization.

        Args:
            model (ModelGraph): the hls4ml model.
            reset (bool): the reset synthesis option
            pr (bool): place and route option
        """
        xls = import_xls()
        project_name = model.config.get_project_name()
        output_dir = model.config.get_output_dir()

        clock_period_ns = model.config.get_config_value('ClockPeriod')
        clock_period_ps = self._to_xls_clock_period_ps(clock_period_ns)

        clock_uncertainty_str = model.config.get_config_value('ClockUncertainty')
        clock_uncertainty_float = self._percent_to_float(clock_uncertainty_str)
        clock_margin_percent: int = self._to_xls_clock_margin_percent(clock_uncertainty_str)

        def build_codegen_flags() -> Dict[str, Any]:
            flags = dict(model.config.get_config_value('XLSCodegenFlags'))
            flags['clock_period_ps'] = clock_period_ps
            flags['clock_margin_percent'] = clock_margin_percent
            if reset is not None:
                flags['reset'] = 'reset' if reset else None
                flags['reset_data_path'] = reset
            return flags

        def build_vivado_flags() -> list[str]:
            flags = [
                '-mode', 'batch',
                '-nolog',
                '-nojournal',
                '-source', './build_prj.tcl',
                '-tclargs',
                project_name,
                model.config.get_config_value('Part'),
                clock_period_ps,
                clock_uncertainty_float,
            ]
            if pr:
                flags += ['--pr']
            return [str(flag) for flag in flags]

        curr_dir: str = os.getcwd()
        os.chdir(f'{output_dir}/firmware')

        # Generate RTL

        opt_ir_path = f'{project_name}.opt.ir'
        opt_ir_text = Path(opt_ir_path).read_text()
        codegen_flags = build_codegen_flags()

        pkg = xls.parse_ir_package(ir=opt_ir_text, filename=opt_ir_path)
        verilog_text = pkg.schedule_and_codegen(**codegen_flags).get_verilog_text()
        Path(f'{project_name}.sv').write_text(verilog_text)

        # Run Vivado for resource report
        os.chdir(curr_dir)
        os.chdir(f'{output_dir}')

        vivado_command: list[str] = ['vivado'] + build_vivado_flags()
        subprocess.run(vivado_command, check=True)

        os.chdir(curr_dir)
        return parse_xls_report(output_dir)
