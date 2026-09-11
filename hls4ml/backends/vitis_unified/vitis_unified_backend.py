import os
import subprocess
import sys
import warnings
from shutil import rmtree

from hls4ml.backends import VitisBackend
from hls4ml.backends.vitis_unified.vitis_unified_validation import load_supported_boards, validate_config
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_vitis_unified_report


class VitisUnifiedBackend(VitisBackend):
    def __init__(self, name='VitisUnified'):
        super().__init__(name=name)

    def build(
        self,
        model,
        reset=False,
        csim=False,
        synth=False,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
        bitfile=False,
        log_to_stdout=True,
        vitis_fifo_sizing=False,
    ):
        for name, value in [('validation', validation), ('export', export), ('vsynth', vsynth)]:
            if value:
                warnings.warn(f'{name} is not supported by the VitisUnified backend and is ignored.', stacklevel=2)

        if fifo_opt and not cosim:
            warnings.warn('fifo_opt requires cosim to be enabled; cosim will be run automatically.', stacklevel=2)
            cosim = True

        if vitis_fifo_sizing and not cosim:
            warnings.warn('vitis_fifo_sizing requires cosim to be enabled; cosim will be run automatically.', stacklevel=2)
            cosim = True

        if cosim and not synth:
            warnings.warn('cosim requires synth to be enabled; synth will be run automatically.', stacklevel=2)
            synth = True

        output_dir = model.config.get_output_dir()
        writer = model.config.backend.writer

        if reset:
            for path in [writer.get_vitis_hls_exec_dir(model), os.path.join(writer.get_vitis_linker_dir(model), '_x')]:
                rmtree(path, ignore_errors=True)
            xclbin = os.path.join(writer.get_vitis_linker_dir(model), f'{model.config.get_project_name()}.xclbin')
            if os.path.isfile(xclbin):
                os.remove(xclbin)

        csim_cfg = os.path.join(output_dir, 'hls_kernel_config_csim.cfg')
        cosim_suffix = 'cosim_fifo_sizing' if vitis_fifo_sizing else 'cosim'
        cosim_cfg = os.path.join(output_dir, f'hls_kernel_config_{cosim_suffix}.cfg')
        # build command
        csynth_cmd = f'v++ -c --mode hls --config {csim_cfg} --work_dir vitis_unified_project'
        # util template (used in csim/cosim/package)
        util_command = 'vitis-run --mode hls --{op} --config {configPath} --work_dir vitis_unified_project'

        # command for each configuration
        vitis_hls_dir = writer.get_vitis_hls_dir(model)
        package_cmd = util_command.format(op='package', configPath=csim_cfg)
        cosim_cmd = util_command.format(op='cosim', configPath=cosim_cfg)
        csim_cmd = util_command.format(op='csim', configPath=csim_cfg)

        kerlink_cmd = './link_system.sh'
        kerlink_cwd = writer.get_vitis_linker_dir(model)

        commands = []
        if synth:
            commands.append(('csynth', csynth_cmd, vitis_hls_dir))
            commands.append(('package', package_cmd, vitis_hls_dir))

        if csim:
            commands.append(('csim', csim_cmd, vitis_hls_dir))

        if cosim or fifo_opt:
            commands.append(('cosim', cosim_cmd, vitis_hls_dir))

        if bitfile:
            commands.append(('kerlink', kerlink_cmd, kerlink_cwd))

        if commands and 'linux' in sys.platform:
            for tool in ['v++', 'vitis-run']:
                if os.system(f'command -v {tool} > /dev/null') != 0:
                    raise Exception(f'Vitis installation not found. Make sure "{tool}" is on PATH.')

        for task_name, command, cwd in commands:
            stdout_log = os.path.join(output_dir, f'{task_name}_stdout.log')
            stderr_log = os.path.join(output_dir, f'{task_name}_stderr.log')
            stdout_target = None if log_to_stdout else open(stdout_log, 'w')
            stderr_target = None if log_to_stdout else open(stderr_log, 'w')

            try:
                process = subprocess.Popen(
                    command, shell=True, cwd=cwd, stdout=stdout_target, stderr=stderr_target, text=True
                )
                process.communicate()

                if process.returncode != 0:
                    raise Exception(f'Build failed for {model.config.get_project_name()} during task "{task_name}".')
            finally:
                if not log_to_stdout:
                    stdout_target.close()
                    stderr_target.close()

        return parse_vitis_unified_report(output_dir)

    def create_initial_config(
        self,
        board='zcu102',
        part=None,
        clock_period=5,
        clock_uncertainty='12.5%',
        io_type='io_stream',
        driver='python',
        input_type='float',
        output_type='float',
        in_stream_buf_size=128,
        out_stream_buf_size=128,
        axi_mode='axi_master',
        **kwargs,
    ):
        supported_boards = load_supported_boards()
        validate_config(board, axi_mode, driver, input_type, output_type, supported_boards)
        part = part or supported_boards[board]['part']

        config = super().create_initial_config(
            part=part, clock_period=clock_period, clock_uncertainty=clock_uncertainty, io_type=io_type, **kwargs
        )

        config['VitisUnifiedConfig'] = {}
        config['VitisUnifiedConfig']['Board'] = board
        config['VitisUnifiedConfig']['axi_mode'] = axi_mode
        config['VitisUnifiedConfig']['in_stream_buf_size'] = in_stream_buf_size
        config['VitisUnifiedConfig']['out_stream_buf_size'] = out_stream_buf_size

        config['VitisUnifiedConfig']['Driver'] = driver
        config['VitisUnifiedConfig']['InputDtype'] = input_type  # float, double or ap_fixed<a,b>
        config['VitisUnifiedConfig']['OutputDtype'] = output_type  # float, double or ap_fixed<a,b>

        if io_type != 'io_stream':
            raise Exception('io_type must be io_stream')

        return config

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def _register_flows(self):
        validation_passes = ['vitisunified:validate_config', 'vitisunified:validate_bram_weights']
        self._default_flow = register_flow('ip', validation_passes, requires=['vitis:ip'], backend=self.name)

        writer_passes = ['make_stamp', 'vitisunified:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=[self._default_flow], backend=self.name)

        # register fifo depth optimization
        fifo_depth_opt_passes = ['vitisunified:fifo_depth_optimization'] + writer_passes

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=[self._default_flow], backend=self.name)
