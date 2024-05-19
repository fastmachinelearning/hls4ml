import os
import sys
import subprocess

from hls4ml.backends import VitisBackend, VivadoBackend
from hls4ml.model.flow import get_flow, register_flow


class VitisAcceleratorBackend(VitisBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VitisAccelerator')
        self._register_layer_attributes()
        self._register_flows()

    def create_initial_config(
        self,
        board='alveo-u55c',
        part=None,
        clock_period=5,
        io_type='io_parallel',
        num_kernel=1,
        num_thread=1,
        batchsize=8192
    ):
        '''
        Create initial accelerator config with default parameters

        Args:
            board: one of the keys defined in supported_boards.json
            clock_period: clock period passed to hls project
            io_type: io_parallel or io_stream
            num_kernel: how many compute units to create on the fpga
            num_thread: how many threads the host cpu uses to drive the fpga
        Returns:
            populated config
        '''
        board = board if board is not None else 'alveo-u55c'
        config = super().create_initial_config(part, clock_period, io_type)
        config['AcceleratorConfig'] = {}
        config['AcceleratorConfig']['Board'] = board
        config['AcceleratorConfig']['Num_Kernel'] = num_kernel
        config['AcceleratorConfig']['Num_Thread'] = num_thread
        config['AcceleratorConfig']['Batchsize'] = batchsize
        return config

    def build(self, model, target="all"):
        if 'linux' in sys.platform:
            if 'XILINX_VITIS' not in os.environ:
                raise Exception("XILINX_VITIS environmental variable missing. Please install XRT and Vitis, and run the setup scripts before building")
            if 'XILINX_XRT' not in os.environ:
                raise Exception("XILINX_XRT environmental variable missing. Please install XRT and Vitis, and run the setup scripts before building")
            if 'XILINX_VIVADO' not in os.environ:
                raise Exception("XILINX_VIVADO environmental variable missing. Please install XRT and Vitis, and run the setup scripts before building")

            if target not in ["all", "host", "hls", "xclbin"]:
                raise Exception("Invalid build target")

            curr_dir = os.getcwd()
            os.chdir(model.config.get_output_dir())
            command = "make " + target
            # Pre-loading libudev
            ldconfig_output = subprocess.check_output(["ldconfig", "-p"]).decode("utf-8")
            for line in ldconfig_output.split("\n"):
                if "libudev.so" in line and "x86" in line:
                    command = "LD_PRELOAD=" + line.split("=>")[1].strip() + " " + command
                    break
            os.system(command)
            os.chdir(curr_dir)
        else:
            raise Exception("Currently untested on non-Linux OS")

    def predict(self, model, x):
        raise Exception("TODO: Needs to be implemented")

    def _register_flows(self):
        validation_passes = [
            'vitisaccelerator:validate_conv_implementation',
            'vitisaccelerator:validate_strategy',
        ]
        validation_flow = register_flow('validation', validation_passes, requires=['vivado:init_layers'], backend=self.name)

        # Any potential templates registered specifically for Vitis backend
        template_flow = register_flow(
            'apply_templates', self._get_layer_templates, requires=['vivado:init_layers'], backend=self.name
        )

        writer_passes = ['make_stamp', 'vitisaccelerator:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vitis:ip'], backend=self.name)

        ip_flow_requirements = get_flow('vivado:ip').requires.copy()
        ip_flow_requirements.insert(ip_flow_requirements.index('vivado:init_layers'), validation_flow)
        ip_flow_requirements.insert(ip_flow_requirements.index('vivado:apply_templates'), template_flow)

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)