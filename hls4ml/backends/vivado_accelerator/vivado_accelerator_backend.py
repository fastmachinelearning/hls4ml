import os
import sys

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
        if 'linux' in sys.platform:
            found = os.system('command -v vivado_hls > /dev/null')
            if found != 0:
                raise Exception('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')

        build_opts = (
            'array set opt {\n'
            f'    reset      {int(reset)}\n'
            f'    csim       {int(csim)}\n'
            f'    synth      {int(synth)}\n'
            f'    cosim      {int(cosim)}\n'
            f'    validation {int(validation)}\n'
            f'    export     {int(export)}\n'
            f'    vsynth     {int(vsynth)}\n'
            f'    fifo_opt   {int(fifo_opt)}\n'
            f'    bitfile    {int(bitfile)}\n'
            '}\n'
        )
        output_dir = model.config.get_output_dir()
        tcl_path = os.path.join(output_dir, 'build_opt.tcl')
        with open(tcl_path, 'w') as f:
            f.write(build_opts)

        curr_dir = os.getcwd()
        os.chdir(output_dir)
        vivado_cmd = 'vivado_hls -f build_prj.tcl'
        os.system(vivado_cmd)
        os.chdir(curr_dir)

        return parse_vivado_report(output_dir)

    def create_initial_config(
        self,
        board='pynq-z2',
        part=None,
        clock_period=5,
        clock_uncertainty='12.5%',
        io_type='io_parallel',
        interface='axi_stream',
        driver='python',
        input_type='float',
        output_type='float',
        platform='xilinx_u250_xdma_201830_2',
    ):
        """
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
        """
        board = board if board is not None else 'pynq-z2'
        config = super().create_initial_config(part, clock_period, clock_uncertainty, io_type)
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
