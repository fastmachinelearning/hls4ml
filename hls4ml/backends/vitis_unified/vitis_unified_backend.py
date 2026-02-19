import os
import subprocess
import sys
from shutil import copy2

from hls4ml.backends import VitisBackend, VivadoBackend
from hls4ml.model.flow import register_flow


class VitisUnifiedBackend(VitisBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VitisUnified')
        self._register_layer_attributes()
        self._register_flows()

    def build(
        self,
        model,
        reset=False,
        csim=False,
        synth=False,
        cosim=False,
        vsynth=False,
        fifo_opt=False,
        bitfile=False,
        log_to_stdout=True,
    ):
        # it builds and return vivado reports
        if 'linux' in sys.platform:
            found = os.system('command -v v++ > /dev/null')
            if found != 0:
                raise Exception('Vitis installation not found. Make sure "vitis" is on PATH.')

            found = os.system('command -v vitis-run > /dev/null')
            if found != 0:
                raise Exception('Vitis installation not found. Make sure "vitis-run" is on PATH.')

        output_dir = model.config.get_output_dir()

        hls_config_file = os.path.join(output_dir, 'hls_kernel_config.cfg')
        # build command
        csynth_cmd = ('v++ -c --mode hls --config {configPath} --work_dir vitis_unified_project').format(
            configPath=hls_config_file
        )
        # util template (used in csim/cosim/package)
        util_command = 'vitis-run --mode hls --{op} --config {configPath} --work_dir vitis_unified_project'

        # command for each configuration
        vitis_hls_dir = model.config.backend.writer.get_vitis_hls_dir(model)
        package_cmd = util_command.format(op='package', configPath=hls_config_file)
        cosim_cmd = util_command.format(op='cosim', configPath=hls_config_file)
        csim_cmd = util_command.format(op='csim', configPath=hls_config_file)

        kerlink_cmd = './link_system.sh'
        kerlink_cwd = model.config.backend.writer.get_vitis_linker_dir(model)

        commands = []
        if synth:
            self.prepare_sim_config_file(model, True)
            commands.append(('csynth', csynth_cmd, vitis_hls_dir))
            commands.append(('package', package_cmd, vitis_hls_dir))

        if csim:
            self.prepare_sim_config_file(model, True)
            commands.append(('csim', csim_cmd, vitis_hls_dir))

        if cosim or fifo_opt:
            self.prepare_sim_config_file(model, False)
            commands.append(('cosim', cosim_cmd, vitis_hls_dir))

        if bitfile:
            commands.append(('kerlink', kerlink_cmd, kerlink_cwd))

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

    def prepare_sim_config_file(self, model, is_csim):
        suffix = 'csim' if is_csim else 'cosim'
        src = f'{model.config.get_output_dir()}/hls_kernel_config_{suffix}.cfg'
        des = f'{model.config.get_output_dir()}/hls_kernel_config.cfg'
        copy2(src, des)
        return des

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
        **_,
    ):
        supported_boards_path = os.path.join(
            os.path.dirname(__file__), 'supported_boards.json'
        )
        if os.path.exists(supported_boards_path):
            import json
            with open(supported_boards_path) as f:
                supported_boards = json.load(f)
            if board in supported_boards:
                part = part or supported_boards[board]['part']
        if part is None:
            part = 'xczu9eg-ffvb1156-2-e'

        config = super().create_initial_config(part, clock_period, clock_uncertainty, io_type)

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
        if input_type not in ['double', 'float']:
            raise Exception('input_type must be float or double')
        if output_type not in ['double', 'float']:
            raise Exception('output_type must be float or double')

        return config

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def _register_flows(self):
        vitis_ip = 'vitis:ip'
        writer_passes = ['make_stamp', 'vitisunified:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vitis:ip'], backend=self.name)
        self._default_flow = vitis_ip

        # register fifo depth optimization
        fifo_depth_opt_passes = ['vitisunified:fifo_depth_optimization'] + writer_passes

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=['vitis:ip'], backend=self.name)
