from hls4ml.backends import VitisBackend, VivadoBackend
from hls4ml.model.flow import get_flow, register_flow


class VitisAcceleratorBackend(VitisBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VitisAccelerator')
        self._register_layer_attributes()
        self._register_flows()

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
        config['AcceleratorConfig']['Platform'] = platform

        return config
