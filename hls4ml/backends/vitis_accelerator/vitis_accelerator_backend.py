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