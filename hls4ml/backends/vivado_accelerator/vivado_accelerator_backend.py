import os

from hls4ml.backends import VivadoBackend
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_vivado_report


class VivadoAcceleratorBackend(VivadoBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VivadoAccelerator')
        self._register_layer_attributes()
        self._register_flows()

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
        bitfile=False,
    ):
        # run the VivadoBackend build
        super().build(
            model,
            reset=reset,
            csim=csim,
            synth=synth,
            cosim=cosim,
            validation=validation,
            export=export,
            vsynth=vsynth,
            fifo_opt=fifo_opt,
        )
        # Get Config to view Board and Platform
        from hls4ml.backends import VivadoAcceleratorConfig

        vivado_accelerator_config = VivadoAcceleratorConfig(
            model.config, model.get_input_variables(), model.get_output_variables()
        )
        # now make a bitfile
        if bitfile:
            if vivado_accelerator_config.get_board().startswith('alveo'):
                self.make_xclbin(model, vivado_accelerator_config.get_platform())
            else:
                curr_dir = os.getcwd()
                os.chdir(model.config.get_output_dir())
                try:
                    os.system('vivado -mode batch -source design.tcl')
                except Exception:
                    print("Something went wrong, check the Vivado logs")
                os.chdir(curr_dir)

        return parse_vivado_report(model.config.get_output_dir())

    def make_xclbin(self, model, platform='xilinx_u250_xdma_201830_2'):
        """Create the xclbin for the given model and target platform.

        Args:
            model (ModelGraph): Compiled and build model.
            platform (str, optional): Development/Deployment target platform, must be installed first.
                The host machine only requires the deployment target platform. Refer to the Getting Started section of
                the Alveo guide. Defaults to 'xilinx_u250_xdma_201830_2'.
        """
        curr_dir = os.getcwd()
        abs_path_dir = os.path.abspath(model.config.get_output_dir())
        os.chdir(abs_path_dir)
        os.makedirs('xo_files', exist_ok=True)
        try:
            os.system('vivado -mode batch -source design.tcl')
        except Exception:
            print("Something went wrong, check the Vivado logs")
        project_name = model.config.get_project_name()
        ip_repo_path = abs_path_dir + '/' + project_name + '_prj' + '/solution1/impl/ip'
        os.makedirs('xclbin_files', exist_ok=True)
        os.chdir(abs_path_dir + '/xclbin_files')
        # TODO Add other platforms
        vitis_cmd = (
            "v++ -t hw --platform "
            + platform
            + " --link ../xo_files/"
            + project_name
            + "_kernel.xo -o'"
            + project_name
            + "_kernel.xclbin' --user_ip_repo_paths "
            + ip_repo_path
        )
        try:
            os.system(vitis_cmd)
        except Exception:
            print("Something went wrong, check the Vitis/Vivado logs")
        os.chdir(curr_dir)

    def create_initial_config(
        self,
        board='pynq-z2',
        part=None,
        clock_period=5,
        io_type='io_parallel',
        interface='axi_stream',
        driver='python',
        input_type='float',
        output_type='float',
        platform='xilinx_u250_xdma_201830_2',
    ):
        '''
        Create initial accelerator config with default parameters

        Args:
            board: one of the keys defined in supported_boards.json
            clock_period: clock period passed to hls project
            io_type: io_parallel or io_stream
            interface: `axi_stream`: generate hardware designs and drivers which exploit axi stream channels.
                       `axi_master`: generate hardware designs and drivers which exploit axi master channels.
                       `axi_lite` : generate hardware designs and drivers which exploit axi lite channels. (Don't use it
                       to exchange large amount of data)
            driver: `python`: generates the python driver to use the accelerator in the PYNQ stack.
                    `c`: generates the c driver to use the accelerator bare-metal.
            input_type: the wrapper input precision. Can be `float` or an `ap_type`. Note: VivadoAcceleratorBackend
                             will round the number of bits used to the next power-of-2 value.
            output_type: the wrapper output precision. Can be `float` or an `ap_type`. Note:
                              VivadoAcceleratorBackend will round the number of bits used to the next power-of-2 value.
            platform: development target platform

        Returns:
            populated config
        '''
        board = board if board is not None else 'pynq-z2'
        config = super().create_initial_config(part, clock_period, io_type)
        config['AcceleratorConfig'] = {}
        config['AcceleratorConfig']['Board'] = board
        config['AcceleratorConfig']['Interface'] = interface  # axi_stream, axi_master, axi_lite
        config['AcceleratorConfig']['Driver'] = driver
        config['AcceleratorConfig']['Precision'] = {}
        config['AcceleratorConfig']['Precision']['Input'] = {}
        config['AcceleratorConfig']['Precision']['Output'] = {}
        config['AcceleratorConfig']['Precision']['Input'] = input_type  # float, double or ap_fixed<a,b>
        config['AcceleratorConfig']['Precision']['Output'] = output_type  # float, double or ap_fixed<a,b>
        if board.startswith('alveo'):
            config['AcceleratorConfig']['Platform'] = platform

        return config

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def _register_flows(self):
        vivado_ip = 'vivado:ip'
        writer_passes = ['make_stamp', 'vivadoaccelerator:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=[vivado_ip], backend=self.name)
        self._default_flow = vivado_ip

        fifo_depth_opt_passes = ['vivadoaccelerator:fifo_depth_optimization'] + writer_passes

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=[vivado_ip], backend=self.name)
