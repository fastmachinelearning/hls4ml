import os
import sys
import subprocess
import numpy as np

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
        batchsize=8192,
        vivado_directives=[]
    ):
        '''
        Create initial accelerator config with default parameters

        Args:
            board: one of the keys defined in supported_boards.json
            clock_period: clock period passed to hls project
            io_type: io_parallel or io_stream
            num_kernel: how many compute units to create on the fpga
            num_thread: how many threads the host cpu uses to drive the fpga
            batchsize: how many samples to process within a single buffer on the fpga
            vivado_directives: Directives passed down to Vivado that controls the hardware synthesis and implementation steps
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
        config['AcceleratorConfig']['Vivado_Directives'] = vivado_directives
        return config

    def build(self, model, reset=False, synth=True, vsynth=True, **kwargs):
        if 'linux' in sys.platform:
            if 'XILINX_VITIS' not in os.environ:
                raise Exception("XILINX_VITIS environmental variable missing. Please install XRT and Vitis, and run the setup scripts before building")
            if 'XILINX_XRT' not in os.environ:
                raise Exception("XILINX_XRT environmental variable missing. Please install XRT and Vitis, and run the setup scripts before building")
            if 'XILINX_VIVADO' not in os.environ:
                raise Exception("XILINX_VIVADO environmental variable missing. Please install XRT and Vitis, and run the setup scripts before building")

            curr_dir = os.getcwd()
            os.chdir(model.config.get_output_dir())

            if reset:
                if vsynth:
                    os.system("make cleanxclbin")
                if synth:
                    os.system("make cleanhls")
                os.system("rm -rf host")
            
            if vsynth:
                if synth:
                    target = "all"
                else:
                    target = "xclbin"
            elif synth:
                target = "hls"
            else:
                target = "host"
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

    def numpy_to_dat(self, model, x):
        if len(model.get_input_variables()) != 1:
            raise Exception("Currently unsupported for multi-input/output projects")
        
        # Verify numpy array of correct shape
        expected_shape = model.get_input_variables()[0].size()
        actual_shape = np.prod(x.shape[1:])
        if expected_shape != actual_shape:
            raise Exception(f'Input shape mismatch, got {x.shape}, expected (_, {expected_shape})')
        
        # Write to tb_data/tb_input_features.dat
        samples = x.reshape(x.shape[0], -1)
        input_dat = f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat'
        np.savetxt(input_dat, samples, fmt='%.4e')

    def dat_to_numpy(self, model):
        expected_shape = model.get_output_variables()[0].size()
        output_file = f'{model.config.get_output_dir()}/tb_data/hw_results.dat'
        y = np.loadtxt(output_file, dtype=float).reshape(-1, expected_shape)
        return y

    def hardware_predict(self, model, x):
        self.numpy_to_dat(model, x)

        currdir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        os.system("./host build/kernel_wrapper.xclbin")
        os.chdir(currdir)

        return self.dat_to_numpy(model)

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