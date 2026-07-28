import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from warnings import warn

import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.backends.bambu.bambu_types import BambuArrayVariableConverter, BambuHLSTypeConverter
from hls4ml.backends.fpga.fpga_types import APTypeConverter
from hls4ml.model.attributes import ChoiceAttribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv1D,
    DepthwiseConv2D,
    Einsum,
    EinsumDense,
    Embedding,
    GarNet,
    GarNetStack,
    Layer,
    LayerNormalization,
    Pooling1D,
    Pooling2D,
    SeparableConv1D,
    SeparableConv2D,
    SimpleRNN,
    TimeDistributed,
)
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.types import (
    FixedPrecisionType,
    IntegerPrecisionType,
    NamedType,
    PackedType,
    RoundingMode,
    SaturationMode,
)
from hls4ml.report import parse_bambu_report
from hls4ml.utils import attribute_descriptions as descriptions
from hls4ml.utils.einsum_utils import parse_einsum

partname_to_bambu = {
    # Intel/Altera
    '5CSEMA5F31C6': {'device_name': '5CSEMA5F31C6', 'family': 'Intel/Altera'},
    '5SGXEA7N2F45C1': {'device_name': '5SGXEA7N2F45C1', 'family': 'Intel/Altera'},
    'EP2C70F896C6': {'device_name': 'EP2C70F896C6', 'family': 'Intel/Altera'},
    'EP2C70F896C6-R': {'device_name': 'EP2C70F896C6-R', 'family': 'Intel/Altera'},
    'EP4SGX530KH40C2': {'device_name': 'EP4SGX530KH40C2', 'family': 'Intel/Altera'},
    # : "LFE335EA8FN484C",
    # : "LFE5U85F8BG756C",
    # : "LFE5UM85F8BG756C",
    # ASAP7 (ASIC)
    # : "asap7-BC",
    # : "asap7-TC",
    # : "asap7-WC",
    # Standard cells / tech libraries
    # : "nangate45",
    # NaNGate/Nextgrids
    # : "nx1h140tsp",
    # : "nx1h35S",
    'nx2h540tsc': {'device_name': 'nx2h540tsc', 'family': 'NanoXplore'},
    # Xilinx legacy
    # : "xc4vlx100-10ff1513",
    # : "xc5vlx110t-1ff1136",
    # : "xc5vlx330t-2ff1738",
    # : "xc5vlx50-3ff1153",
    # : "xc6vlx240t-1ff1156",
    # 7-series
    'xc7a100tcsg324-1': {
        'device_name': 'xc7a100t-1csg324',
        'family': 'Xilinx',
    },  # 7-series Artix; matches the entry in Bambu's `Available devices` listing
    # : "xc7vx330t-1ffg1157",
    # : "xc7vx485t-2ffg1761",
    # : "xc7vx690t-3ffg1930",
    # : "xc7z020-1clg484",
    # : "xc7z020-1clg484-YOSYS",
    # : "xc7z045-2ffg900",
    # UltraScale / UltraScale+
    # : "xcku060-3ffva1156",
    # : "xcu250-2Lfigd2104",
    # : "xcu280-2Lfsvh2892",
    # : "xcu50-2fsvh2104",
    'xczu7ev-ffvc1156-2-e': {'device_name': 'xczu7ev-2ffvc1156', 'family': 'Xilinx'},
    'xcu55c-fsvh2892-2L-e': {'device_name': 'xcu55c-2Lfsvh2892', 'family': 'Xilinx'},
}


class BambuBackend(FPGABackend):
    def __init__(self):
        super().__init__('Bambu')
        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        # Add RNN-specific attributes, recurrent_reuse_factor and static implementation
        rnn_layers = [SimpleRNN, LSTM, GRU]

        for layer in rnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('recurrent_reuse_factor', default=1, description=descriptions.reuse_factor))
            attrs.append(
                ConfigurableAttribute('static', value_type=bool, default=True, description=descriptions.recurrent_static)
            )
            attrs.append(ConfigurableAttribute('table_size', default=1024, description=descriptions.table_size))
            attrs.append(TypeAttribute('table', default=FixedPrecisionType(18, 8), description=descriptions.table_type))
            self.attribute_map[layer] = attrs

        bidir_rnn_layers = [Bidirectional]
        for layer in bidir_rnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('forward_reuse_factor', default=1, description=descriptions.reuse_factor))
            attrs.append(ConfigurableAttribute('backward_reuse_factor', default=1, description=descriptions.reuse_factor))
            attrs.append(
                ConfigurableAttribute('forward_recurrent_reuse_factor', default=1, description=descriptions.reuse_factor)
            )
            attrs.append(
                ConfigurableAttribute('backward_recurrent_reuse_factor', default=1, description=descriptions.reuse_factor)
            )
            attrs.append(
                ConfigurableAttribute('static', value_type=bool, default=True, description=descriptions.recurrent_static)
            )
            attrs.append(ConfigurableAttribute('table_size', default=1024, description=descriptions.table_size))
            attrs.append(TypeAttribute('table', default=FixedPrecisionType(18, 8), description=descriptions.table_type))
            self.attribute_map[layer] = attrs

        # Add ParallelizationFactor to Conv1D/2D
        pf_layers = [
            Conv1D,
            Conv2D,
        ]

        for layer in pf_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('parallelization_factor', default=1, description=descriptions.conv_pf))
            self.attribute_map[layer] = attrs

        # Add ConvImplementation to Convolution+Pooling layers
        cnn_layers = [Conv1D, Conv2D, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Pooling1D, Pooling2D]
        for layer in cnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(
                ChoiceAttribute(
                    'conv_implementation',
                    choices=['LineBuffer', 'Encoded'],
                    default='LineBuffer',
                    description=descriptions.conv_implementation,
                )
            )
            self.attribute_map[layer] = attrs

        # Add LayerNorm attributes
        ln_layers = [LayerNormalization]
        for layer in ln_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('table_range_power2', default=0, description=descriptions.table_range_power2))
            attrs.append(ConfigurableAttribute('table_size', default=4096, description=descriptions.table_size))
            attrs.append(
                TypeAttribute(
                    'table',
                    default=FixedPrecisionType(
                        8, 5, signed=False, rounding_mode=RoundingMode.RND_CONV, saturation_mode=SaturationMode.SAT
                    ),
                    description=descriptions.table_type,
                )
            )
            attrs.append(
                TypeAttribute(
                    'accum',
                    default=FixedPrecisionType(
                        14, 4, signed=True, rounding_mode=RoundingMode.RND_CONV, saturation_mode=SaturationMode.SAT
                    ),
                    description=descriptions.accum_type,
                )
            )
            self.attribute_map[layer] = attrs

        # Add TimeStepLoopParallelism to TimeDistributed
        attrs = self.attribute_map.get(TimeDistributed, [])
        attrs.append(
            ChoiceAttribute(
                'time_step_loop_parallelism',
                choices=['Off', 'Unroll', 'Pipeline'],
                default='Off',
                description=descriptions.time_distributed_loop,
            )
        )
        self.attribute_map[TimeDistributed] = attrs

    def _register_flows(self):
        # Register flows using bambu: passes (copied from vivado)
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        streaming_passes = [
            'bambu:inplace_stream_flatten',  # Inform downstream changed packsize in case of skipping flatten
            'bambu:reshape_stream',
            'bambu:clone_output',
            'bambu:insert_zero_padding_before_conv1d',
            'bambu:insert_zero_padding_before_conv2d',
            'bambu:broadcast_stream',
        ]
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        quantization_passes = [
            'bambu:merge_batch_norm_quantized_tanh',
            'bambu:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
            'bambu:xnor_pooling',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            'bambu:remove_final_reshape',
            'bambu:optimize_pointwise_conv',
            'bambu:inplace_parallel_reshape',
            'bambu:inplace_stream_flatten',
            'bambu:skip_softmax',
            'bambu:fix_softmax_table_size',
            'infer_precision_types',
            'bambu:distributed_arithmetic_codegen',
            'bambu:distributed_arithmetic_einsum_codegen',
            'bambu:fuse_quantizer_into_d_a_layers',
            'bambu:process_fixed_point_quantizer_layer',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        bambu_types = [
            'bambu:transform_types',
            'bambu:register_bram_weights',
            'bambu:generate_conv_streaming_instructions',
            'bambu:apply_resource_strategy',
            'bambu:generate_conv_im2col',
            'bambu:generate_unrolled_dense_resource',
            'bambu:set_pipeline_style',
            'bambu:d_a_latency_dense_template',
            'bambu:d_a_latency_conv_template',
        ]
        bambu_types_flow = register_flow('specific_types', bambu_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'bambu:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['bambu:ip'], backend=self.name)

        fifo_depth_opt_passes = [
            'bambu:fifo_depth_optimization'
        ] + writer_passes  # After optimization, a new project will be written

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=['bambu:ip'], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + streaming_passes
            + quantization_passes
            + optimization_passes
            + bambu_types
            + templates
            + writer_passes
            + fifo_depth_opt_passes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            streaming_flow,
            quantization_flow,
            optimization_flow,
            bambu_types_flow,
            template_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(
        self,
        part='xc7a100tcsg324-1',
        clock_period=5,
        clock_uncertainty='12.5%',
        io_type='io_parallel',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
        tb_output_stream='both',
        **_,
    ):
        """Create initial configuration of the Bambu backend.

        Args:
            part (str, optional): The FPGA part to be used. Defaults to 'xc7a100tcsg324-1'.
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

        partname = part if part is not None else 'xc7a100tcsg324-1'
        config['Part'] = partname
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
        if partname in partname_to_bambu.keys():
            config['FPGAFamily'] = partname_to_bambu[partname]['family']

        return config

    def build(
        self,
        model,
        *,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
        log_to_stdout=True,
        args=None,
        env=None,
        run_kwargs=None,
    ):
        """Run Bambu HLS on given model. Enable/disable parts of synthesis process based on boolean arguments.
        Pass extra Bambu-specific arguments via the ``args`` command.

        Args:
            model (ModelGraph): Model to be built with Bambu.
            reset (bool, optional): If true, deletes Bambu-generated artifacts if they are present.
            csim (bool, optional): Run C-Simulation of model on its testbench. Defaults to True.
            synth (bool, optional): Standard CPP to HDL translation with Bambu. If set to false, Bambu is not called.
            cosim (bool, optional): Run RTL-Cosimulation of model on its testbench. Defaults to false.
            validation (bool, optional): Checks for bitwise equality of csim and cosim results.
            export (bool, optional): NotImplemented (will create exported IP in project directory)
            vsynth (bool, optional):
                Optimize, Place, and Route synthesized design for any part that is supported by
                Bambu. User will need the part downloaded in their Vivado installation.
                Bambu requires cosim=True to run vsynth.
            fifo_opt (bool, optional): NotImplemented (will optimize FIFO length based on RTL cosim)
            log_to_stdout (bool, optional): Forward Bambu's ``stdout`` and ``stderr`` to system
            args (str | Sequence[str] | None): Arguments appended to default Bambu command for this model.
            env (Mapping[str, str] | None): Environment overrides applied to the subprocess.
            run_kwargs (dict | None): Additional keyword arguments forwarded to ``subprocess.run``.

        Returns:
            dict:
                'CSimResults' (np.array(float), optional):
                    C Simulation array of testbench predictions
                'CosimResults' (np.array(float), optional):
                    RTL Cosimulation array of testbench predictions
                'BambuMetrics' (dict, optional):
                    Metrics returned by Bambu's evaluation
                'ImplementationReport' (dict, optional):
                    Parsed metrics from post-route utilization report
                'TimingReport' (dict, optional):
                    Parsed metrics from post-route timing summary report
                'PowerReport' (dict, optional):
                    Parsed metrics from post-route power report

        Example:
            result = model.build(
                csim=True,
                cosim=True,
                validation=True,
                log_to_stdout=True
                args='-v4 --seed=5'
            )
        """

        project_name = model.config.get_project_name()
        project_dir = model.config.get_output_dir()
        part_family = model.config.get_config_value('FPGAFamily')
        # Bambu's InterfaceInfer pass (ChasePointerInterfaceRecurse) has a
        # 64-bit-pointer bug that fires only on the ac_channel FIFO ports
        # io_stream generates ("unexpected condition",
        # InterfaceInfer.cpp:1068); io_parallel's BRAM address/data ports
        # never hit it. Confirmed by bisecting Bambu's HLS flags: identical
        # stream C++ synthesizes to AXIS Verilog once -m64 is omitted.
        io_type = model.config.get_config_value('IOType')

        # Bambu-specific command/flags
        BASE_COMMAND = ['bambu'] + self._get_hls_sources(project_name) + [f'--top-fname={self._get_top_fname(project_name)}']
        # `-ftemplate-depth=2048` is forwarded by Bambu directly to its
        # clang-16 front-end. The default 1024-deep template instantiation
        # limit is hit by `std::make_index_sequence<N>` in libstdc++ 4.9.4
        # (the libstdc++ shipped inside Bambu's AppImage) when N == 1024,
        # which is exactly the upper bound `core_templates.py` clamps
        # softmax's `exp_table_size` to. The recursive index-tuple builder
        # at `bits/utility:215` in that libstdc++ requires N levels of
        # depth, so any softmax whose input `data_T` width is >= 10 (e.g.
        # `ap_fixed<16,6>` or a dense accumulator like `ac_fixed<18,10>`)
        # trips the limit and the front-end aborts with `recursive template
        # instantiation exceeded maximum depth of 1024`. Raising the
        # ceiling is the fix the compiler error itself suggests, and is
        # harmless for the smaller cases (no extra runtime/memory cost).
        # See firmware/nnet_utils/nnet_activation.h:228 in hls4ml's bambu
        # templates for the actual `make_index_sequence` call site.
        CC_TEMPLATE_DEPTH = '-ftemplate-depth=2048'
        if os.environ.get('USE_HLS4ML_AC_TYPES'):
            # Legacy escape hatch: compile against the ac_types checkout the
            # writer copies into firmware/ac_types, with a 64-bit host triple.
            # The -m64 this needs trips Bambu's InterfaceInfer 64-bit pointer
            # bug on io_stream's ac_channel FIFOs, so it is filtered below.
            REQ_ARGS = [
                '-lm',
                '-Ifirmware/ac_types/include',
                '--compiler=I386_CLANG16',
                CC_TEMPLATE_DEPTH,
                '--generate-interface=INFER',
                '-v4',
                '-m64',
            ]
        else:
            # Default: Bambu's own shipped ac/ap headers (usr/include/panda),
            # always in sync with the toolchain, 32-bit triple — no -m64
            # needed, which also sidesteps the InterfaceInfer io_stream bug.
            REQ_ARGS = ['-lm', '--compiler=I386_CLANG16', CC_TEMPLATE_DEPTH, '--generate-interface=INFER', '-v4']
        if io_type == 'io_stream':
            REQ_ARGS = [arg for arg in REQ_ARGS if arg != '-m64']
        CMD_ARGS = []

        result = {}

        # --- RESET ---
        bambu_output_patterns = [
            'HLS_output',
            'panda-temp',
            'vivado_reports',
            'bambu_results*.xml',
            'evaluate*.sh',
            'memory_allocation*.xml',
            f'{project_name}-*_tb.exe',
            f'{project_name}.v',
            'results.txt',
            'synthesize*.sh',
            'panda_libtech.v',
            '*.mem',
        ]
        matches = [p for pat in bambu_output_patterns for p in Path(project_dir).glob(pat)]
        is_dirty_directory = any(matches)
        if reset:
            if is_dirty_directory:
                for p in matches:
                    if p.is_file() or p.is_symlink():
                        p.unlink(missing_ok=True)
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    print(f'Removed: {p}')
        else:
            if is_dirty_directory:
                warn(
                    'WARNING: Bambu is being rerun on a directory instead of running on a fresh directory (not recommended).'
                )

        # --- CSIM ---
        if csim:
            self._build_testbench_exe(model)

            # Execute testbench
            ret = subprocess.run(
                [f'./{project_name}-{model.config.get_config_value("Stamp")}_tb.exe'],
                cwd=project_dir,
                capture_output=True,
                text=True,
            )

            if ret.returncode != 0:
                raise RuntimeError(f'C++ testbench execution failed:\nSTDOUT:\n{ret.stdout}\nSTDERR:\n{ret.stderr}')

        if synth:
            clock_period = model.config.get_config_value('ClockPeriod')
            part_name = model.config.get_config_value(
                'Part'
            )  # Bambu uses its own 'device name' which does NOT always coincide with part name
            device_name = partname_to_bambu.get(part_name, {}).get('device_name', None)
            if device_name is None:
                warn(
                    f'WARNING: Part name {part_name} has no registered mapping to a Bambu --device-name. '
                    f"Using '--device-name={part_name}'. "
                    "(See valid Bambu device names by running Bambu with High Verbosity flag '-v4')"
                )
                device_name = part_name
            CMD_ARGS += [f'--device-name={device_name}', f'--clock-period={clock_period}']

        # --- COSIM ---
        if cosim:
            if not synth:
                raise ValueError('To run RTL cosimulation, C/RTL synthesis must be run.')
            CMD_ARGS += [f'--generate-tb={self._get_cosim_testbench(project_name)}', '--simulate', '-DRTL_SIM']

            # Force Verilator for NanoXplore parts. Bambu's default
            # simulator selection picks a NanoXplore-native flow whose
            # XML device files fail to parse on the current toolchain
            # (`Error during XML parsing of device files`). Verilator is
            # toolchain-agnostic and works for cosim regardless of the
            # target FPGA family.
            part_name = model.config.get_config_value('Part')
            family = partname_to_bambu.get(part_name, {}).get('family', None)
            if family == 'NanoXplore':
                CMD_ARGS += ['--simulator=VERILATOR']

        # --- VALIDATION ---
        if validation:
            if not csim or not cosim:
                raise ValueError('To validate C simulation & RTL simulation equality, csim and cosim must both be run.')

        # --- EXPORT ---
        if export:
            raise NotImplementedError()  # TODO - Requires an ad-hoc .tcl script

        # --- VSYNTH ---
        if vsynth:
            if not synth:
                raise ValueError('To synthesize for specific part, C/RTL synthesis must be run.')
            if not cosim:
                raise ValueError('To synthesize for specific part in Bambu, RTL cosimulation must be run.')

            CMD_ARGS += ['--evaluation']

        # --- FIFO_OPT ---
        if fifo_opt:
            raise NotImplementedError()  # TODO - Requires an ad-hoc .tcl script

        # Build user's custom command with Bambu defaults
        command_tokens = BASE_COMMAND + REQ_ARGS + CMD_ARGS
        if args is not None:
            command_tokens += self._normalize_bambu_command(args)
        command_str = ' '.join(shlex.quote(str(token)) for token in command_tokens)

        # Write/rewrite formatted command to build_bambu.sh for later execution
        script_path = Path(project_dir) / 'build_bambu.sh'
        content = script_path.read_text()
        content = self._replace_block(
            content, '# HLS4ML insert_bambu_command BEGIN', '# HLS4ML insert_bambu_command END', command_str
        )
        copy_code = self._final_report_copying_code(part_family) if vsynth else ''
        content = self._replace_block(
            content, '# HLS4ML insert_final_report_copying BEGIN', '# HLS4ML insert_final_report_copying END', copy_code
        )
        script_path.write_text(content)

        if not synth:
            # "Dry run"
            result.update(parse_bambu_report(project_dir, part_family))
            return result
        else:
            self._ensure_bambu_available()

            # Alter os runtime environment
            run_env = os.environ.copy()
            if env is not None:
                if not hasattr(env, 'items'):
                    raise TypeError('env must be a mapping of environment variables.')
                for key, value in env.items():
                    if value is None:
                        run_env.pop(str(key), None)
                    else:
                        run_env[str(key)] = str(value)

            # Add optional runtime keyword arguments to subprocess
            if run_kwargs is None:
                run_kwargs = {}
            if not isinstance(run_kwargs, dict):
                raise TypeError('run_kwargs must be a mapping')
            if log_to_stdout and any(stream in run_kwargs for stream in ('stdout', 'stderr')):
                raise ValueError('Cannot set stdout/stderr in run_kwargs when log_to_stdout=True.')

            stdout_log = os.path.join(project_dir, 'build_stdout.log')
            stderr_log = os.path.join(project_dir, 'build_stderr.log')
            stdout_target = None
            stderr_target = None
            build_command = 'bash build_bambu.sh'
            output_dir = project_dir

            if log_to_stdout:
                stdout_target = None
                stderr_target = None
            else:
                if 'stdout' not in run_kwargs:
                    stdout_target = open(stdout_log, 'w')
                else:
                    stdout_target = run_kwargs.pop('stdout')
                if 'stderr' not in run_kwargs:
                    stderr_target = open(stderr_log, 'w')
                else:
                    stderr_target = run_kwargs.pop('stderr')

            # Run Bambu
            try:
                process = subprocess.Popen(
                    build_command,
                    shell=True,
                    cwd=output_dir,
                    stdout=stdout_target,
                    stderr=stderr_target,
                    env=run_env,
                    text=run_kwargs.get('text', True),
                    **run_kwargs,
                )
                process.communicate()
            finally:
                if stdout_target is not None and stdout_target is not subprocess.PIPE:
                    stdout_target.close()
                if stderr_target is not None and stderr_target is not subprocess.PIPE:
                    stderr_target.close()

            # A failed Bambu run must not fall through to report parsing: the reports
            # would either be missing (yielding an empty result that looks like success)
            # or left over from an earlier run when reset=False.
            if process.returncode != 0:
                logs = '' if log_to_stdout else f'\n  stdout log: {stdout_log}\n  stderr log: {stderr_log}'
                raise RuntimeError(
                    f'Bambu build failed with exit code {process.returncode}.\n'
                    f'  command: {build_command}\n'
                    f'  directory: {output_dir}{logs}'
                )

            # Add main results
            result.update(parse_bambu_report(project_dir, part_family))

            return result

    def _get_hls_sources(self, project_name):
        """Return the list of C++ source files to pass to Bambu."""
        return [os.path.join('firmware', f'{project_name}.cpp')]

    def _get_top_fname(self, project_name):
        """Return the top-level function name for Bambu synthesis."""
        return project_name

    def _get_cosim_testbench(self, project_name):
        """Return the testbench filename for Bambu RTL co-simulation."""
        return f'{project_name}_test.cpp'

    def _build_testbench_exe(self, model):
        ret = subprocess.run(
            ['bash', 'build_tb_exe.sh'],
            text=True,
            capture_output=True,
            cwd=model.config.get_output_dir(),
        )
        if ret.returncode != 0:
            project_name = model.config.get_project_name()
            raise RuntimeError(
                f'Failed to build testbench executable for "{project_name}":\nSTDOUT:\n{ret.stdout}\nSTDERR:\n{ret.stderr}'
            )

    def _final_report_copying_code(self, family):
        """Aggregate final reports in one directory based on Part Family/Software used"""
        if family == 'Xilinx':
            # Bambu's Vivado-flow output directory has moved across releases
            # (`HLS_output/Synthesis/vivado_flow` in older versions,
            # `HLS_output/xilinx/flow_backend` in current). Search the full
            # HLS_output tree so the script keeps working across versions.
            return (
                'src_root="HLS_output"\n'
                'dst_root="vivado_reports"\n'
                'mkdir -p "$dst_root"\n'
                r'find "$src_root" -type f \( -iname "*.rpt" -o -iname "*.xml" \) -exec cp -p {} "$dst_root"/ \;'
            )
        else:  # TODO: Add more parsing code for different families/softwares
            return ''

    def _replace_block(self, content, start, end, new_body):
        pattern = rf'{start}.*?{end}'
        replacement = f'{start}\n{new_body}\n{end}'
        return re.sub(pattern, replacement, content, flags=re.S)

    @staticmethod
    def _ensure_bambu_available():
        if shutil.which('bambu') is None:
            raise OSError('Bambu HLS installation not found. Make sure "bambu" is on PATH.')

    @staticmethod
    def _normalize_bambu_command(args):
        if args is not None:
            if isinstance(args, (list, tuple)):
                tokens = [str(token) for token in args]
            elif isinstance(args, str):
                tokens = shlex.split(args)
            else:
                raise TypeError('args must be a string or a sequence of strings.')
        else:
            tokens = []

        return tokens

    @layer_optimizer(Layer)
    def init_base_layer(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)

        target_cycles = layer.model.config.get_target_cycles(layer)
        layer.set_attr('target_cycles', target_cycles)

    @layer_optimizer(Dense)
    def init_dense(self, layer):
        index_t = IntegerPrecisionType(width=1, signed=False)
        compression = layer.model.config.get_compression(layer)
        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            if compression:
                layer.set_attr('strategy', 'compressed')
                index_t = layer.get_weights('weight').type.index_precision
            else:
                layer.set_attr('strategy', 'resource')
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                layer.set_attr('strategy', 'resource_unrolled')
        elif layer.model.config.get_strategy(layer).lower() in ('distributed_arithmetic', 'da'):
            rf = layer.get_attr('reuse_factor')
            if rf != 1:
                raise Exception(f'Layer {layer.name} has rf = {rf} != 1, but has strategy = "distributed_arithmetic".')
            layer.set_attr('strategy', 'distributed_arithmetic')
        else:
            layer.set_attr('strategy', 'latency')
        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', index_t))

    # TODO consolidate these functions into a single `init_conv`
    @layer_optimizer(Conv1D)
    def init_conv1d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv1D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}".'
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            elif layer.model.config.get_config_value('IOType') == 'io_parallel':
                print(
                    f'Unrolled resource strategy cannot be combined with io_parallel in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                layer.set_attr('strategy', 'resource_unrolled')
        elif layer.model.config.get_strategy(layer).lower() in ('distributed_arithmetic', 'da'):
            rf = layer.get_attr('reuse_factor')
            if rf != 1:
                raise Exception(f'Layer {layer.name} has rf = {rf} != 1, but has strategy = "distributed_arithmetic".')
            layer.set_attr('strategy', 'distributed_arithmetic')
        else:
            layer.set_attr('strategy', 'latency')

        out_width = layer.get_output_variable().shape[0]

        # Not overriding user parallelization factor, if already set and user has not specified a value
        user_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', None)
        layer_pf = layer.get_attr('parallelization_factor', None)
        chosen_pf = user_pf or layer_pf or 1
        valid_pf = self.get_valid_conv_partition_splits(1, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_width // closest_pf)
        layer.set_attr('parallelization_factor', closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(SeparableConv1D)
    def init_sepconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_width = layer.get_output_variable().shape[0]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(1, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_width // closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    @layer_optimizer(DepthwiseConv1D)
    def init_depconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_width = layer.get_output_variable().shape[0]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(1, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_width // closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Conv2D)
    def init_conv2d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_target_reuse_factor(layer)
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            elif layer.model.config.get_config_value('IOType') == 'io_parallel':
                print(
                    f'Unrolled resource strategy cannot be combined with io_parallel in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                layer.set_attr('strategy', 'resource_unrolled')
        elif layer.model.config.get_strategy(layer).lower() in ('distributed_arithmetic', 'da'):
            rf = layer.get_attr('reuse_factor')
            if rf != 1:
                raise Exception(f'Layer {layer.name} has rf = {rf} != 1, but has strategy = "distributed_arithmetic".')
            layer.set_attr('strategy', 'distributed_arithmetic')
        else:
            layer.set_attr('strategy', 'latency')

        out_height = layer.get_output_variable().shape[0]
        out_width = layer.get_output_variable().shape[1]

        # Not overriding user parallelization factor, if already set and user has not specified a value
        user_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', None)
        layer_pf = layer.get_attr('parallelization_factor', None)
        chosen_pf = user_pf or layer_pf or 1
        if user_pf is not None and layer_pf is not None:
            if user_pf != layer_pf:
                warn(
                    f'For layer {layer.name}, parallelization factor of {layer_pf} is defined in the proxy-model, but is overridden by the user to {user_pf}.'  # noqa: E501
                )

        valid_pf = self.get_valid_conv_partition_splits(out_height, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_height * out_width // closest_pf)
        layer.set_attr('parallelization_factor', closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(SeparableConv2D)
    def init_sepconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_height = layer.get_output_variable().shape[0]
        out_width = layer.get_output_variable().shape[1]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(out_height, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf

        layer.set_attr('n_partitions', out_height * out_width // closest_pf)
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    @layer_optimizer(DepthwiseConv2D)
    def init_depconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_height = layer.get_output_variable().shape[0]
        out_width = layer.get_output_variable().shape[1]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(out_height, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_height * out_width // closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Pooling1D)
    def init_pooling1d(self, layer):
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Pooling2D)
    def init_pooling2d(self, layer):
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Embedding)
    def init_embed(self, layer):
        if layer.attributes['n_in'] is None:
            raise Exception('Input length of Embedding layer must be specified.')

    @layer_optimizer(LSTM)
    def init_lstm(self, layer):
        # TODO Allow getting recurrent reuse factor from the config
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                self.set_closest_reuse_factor(
                    layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor', include_max_rf=False
                )
                layer.set_attr('strategy', 'resource_unrolled')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(GRU)
    def init_gru(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                self.set_closest_reuse_factor(
                    layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor', include_max_rf=False
                )
                layer.set_attr('strategy', 'resource_unrolled')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(TimeDistributed)
    def init_time_distributed(self, layer):
        loop_mode = layer.get_attr('time_step_loop_parallelism', 'off').lower()
        if loop_mode == 'unroll' and layer.model.config.get_config_value('IOType') == 'io_stream':
            warn(f'Cannot unroll time step loop in layer "{layer.name}" while using "io_stream".')
            loop_mode = 'off'
        layer.set_attr('time_step_loop_parallelism', loop_mode)

    @layer_optimizer(Bidirectional)
    def init_bidirectional(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)

        for i, d in enumerate(['forward', 'backward']):
            layer.set_attr(f'{d}_reuse_factor', reuse_factor)
            layer.set_attr(f'{d}_recurrent_reuse_factor', reuse_factor)

            if layer.model.config.is_resource_strategy(layer):
                n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)[i]
                self.set_closest_reuse_factor(layer, n_in, n_out, attribute=f'{d}_reuse_factor')
                self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute=f'{d}_recurrent_reuse_factor')
                layer.set_attr('strategy', 'resource')

            elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
                use_resource_instead = False
                if layer.get_attr('reuse_factor', 1) == 1:
                    print(
                        f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name} ({d})". '
                        'Using "resource" strategy instead.'
                    )
                use_resource_instead = True

                n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)[i]
                if use_resource_instead:
                    self.set_closest_reuse_factor(layer, n_in, n_out, attribute=f'{d}_reuse_factor')
                    self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute=f'{d}_recurrent_reuse_factor')
                    layer.set_attr('strategy', 'resource')
                else:
                    self.set_closest_reuse_factor(layer, n_in, n_out, attribute=f'{d}_reuse_factor', include_max_rf=False)
                    self.set_closest_reuse_factor(
                        layer, n_in_recr, n_out_recr, attribute=f'{d}_recurrent_reuse_factor', include_max_rf=False
                    )
                    layer.set_attr('strategy', 'resource_unrolled')
            else:
                layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(GarNet)
    def init_garnet(self, layer):
        reuse_factor = layer.attributes['reuse_factor']

        var_converter = BambuArrayVariableConverter(
            type_converter=BambuHLSTypeConverter(precision_converter=APTypeConverter())
        )

        # A bit controversial but we are going to set the partitioning of the input here
        in_layer = layer.model.graph[layer.inputs[0]]
        in_var = layer.get_input_variable(layer.inputs[0])
        partition_factor = in_var.shape[1] * (in_var.shape[0] // reuse_factor)
        in_pragma = ('partition', 'cyclic', partition_factor)
        new_in_var = var_converter.convert(in_var, pragma=in_pragma)
        in_layer.set_attr(layer.inputs[0], new_in_var)

        if layer.attributes['collapse']:
            out_pragma = 'partition'
        else:
            partition_factor = layer._output_features * (layer.attributes['n_vertices'] // reuse_factor)
            out_pragma = ('partition', 'cyclic', partition_factor)

        out_name, out_var = next(iter(layer.variables.items()))
        new_out_var = var_converter.convert(out_var, pragma=out_pragma)

        layer.set_attr(out_name, new_out_var)

    @layer_optimizer(GarNetStack)
    def init_garnet_stack(self, layer):
        self.init_garnet(layer)

    @layer_optimizer(EinsumDense)
    def init_einsum_dense(self, layer: EinsumDense) -> None:
        kernel: np.ndarray = layer.attributes['weight_data']
        bias: np.ndarray | None = layer.attributes['bias_data']
        equation = layer.attributes['equation']
        inp_shape = layer.attributes['inp_shape']
        out_shape = layer.attributes['out_shape']

        kernel_shape = kernel.shape
        recipe = parse_einsum(equation, inp_shape, kernel_shape)
        assert not any(recipe['direct_sum_axis']), (
            'Do not put direct sum indices (e.g., only appears in one of the operands) in the equation.'
            'Use explicit addition operator before instead.'
        )
        inp_tpose_idxs, ker_tpose_idxs = recipe['in_transpose_idxs']
        out_tpose_idxs = recipe['out_transpose_idxs']

        # Pre-transpose kernel (and bias) to save a transpose in cpp. Shouldn't matter for latency strategy though.
        # hls4ml dense acts like i,ij->j
        # parser assumes ij,j->i, so we need to transpose the kernel to match
        kernel = kernel.transpose(ker_tpose_idxs)
        kernel = kernel.reshape(recipe['I'], recipe['L1'], recipe['C']).transpose(0, 2, 1)

        def to_original_kernel(tkernel: np.ndarray) -> np.ndarray:
            _kernel = tkernel.transpose(0, 2, 1)
            _kernel = _kernel.reshape(tuple(kernel_shape[i] for i in ker_tpose_idxs))
            return _kernel.transpose(np.argsort(ker_tpose_idxs))

        # TODO: for weight in bram mode (resource), broadcasting bias here shall be avoided.
        if bias is not None:
            bias = np.broadcast_to(bias, out_shape).transpose(np.argsort(out_tpose_idxs))
        else:
            # The automatically created bias is just the last dimension of the output shape
            # Which is too small in general for einsum dense.
            # The transpose is just to match the shape in case of have real bias, no real effect.
            bias = np.zeros(out_shape).transpose(np.argsort(out_tpose_idxs))

        layer.attributes['weight_data'] = kernel
        layer.attributes['to_original_kernel'] = to_original_kernel
        layer.attributes['bias_data'] = bias
        layer.attributes['inp_tpose_idxs'] = inp_tpose_idxs
        layer.attributes['out_tpose_idxs'] = out_tpose_idxs
        layer.attributes['out_interpert_shape'] = recipe['out_interpert_shape']
        layer.attributes['n_free_data'] = recipe['L0']
        layer.attributes['n_free_kernel'] = recipe['L1']
        layer.attributes['n_inplace'] = recipe['I']
        layer.attributes['n_contract'] = recipe['C']
        pf = layer.attributes.get('parallelization_factor', recipe['L0'])
        layer.attributes['parallelization_factor'] = pf

        layer.add_weights(compression=layer.model.config.get_compression(layer))
        layer.add_bias()

        strategy: str | None = layer.model.config.get_strategy(layer)
        if not strategy:
            layer.set_attr('strategy', 'latency')
            return
        if strategy in ('latency', 'resource', 'distributed_arithmetic'):
            layer.set_attr('strategy', strategy)
            return
        warn(f'Invalid strategy "{strategy}" for EinsumDense layer "{layer.name}". Using "latency" strategy instead.')
        layer.set_attr('strategy', 'latency')

    @layer_optimizer(Einsum)
    def init_einsum(self, layer: Einsum) -> None:
        equation = layer.attributes['equation']
        inp0_shape = layer.attributes['inp0_shape']
        inp1_shape = layer.attributes['inp1_shape']

        recipe = parse_einsum(equation, inp0_shape, inp1_shape)
        assert not any(recipe['direct_sum_axis']), (
            'Do not put direct sum indices (e.g., only appears in one of the operands) in the equation.'
            'Use explicit addition operator before instead.'
        )
        inp0_tpose_idxs, inp1_tpose_idxs = recipe['in_transpose_idxs']
        out_tpose_idxs = recipe['out_transpose_idxs']

        layer.attributes.update(recipe)
        layer.attributes['n_free0'] = recipe['L0']
        layer.attributes['n_free1'] = recipe['L1']
        layer.attributes['n_inplace'] = recipe['I']
        layer.attributes['n_contract'] = recipe['C']
        layer.attributes['out_interpert_shape'] = recipe['out_interpert_shape']

        layer.attributes['inp0_tpose_idxs'] = inp0_tpose_idxs
        layer.attributes['inp1_tpose_idxs'] = inp1_tpose_idxs
        layer.attributes['out_tpose_idxs'] = out_tpose_idxs

        pf = layer.attributes.get('parallelization_factor', recipe['L0'])
        layer.attributes['parallelization_factor'] = pf

        strategy: str | None = layer.model.config.get_strategy(layer)
        if not strategy:
            layer.set_attr('strategy', 'latency')
            return
        if strategy.lower() == 'resource':
            layer.set_attr('strategy', 'resource')
            return
        if strategy.lower() in ('latency', 'distributed_arithmetic'):
            layer.set_attr('strategy', 'latency')
            return
        warn(f'Invalid strategy "{strategy}" for Einsum layer "{layer.name}". Using "latency" strategy instead.')
        layer.set_attr('strategy', 'latency')
