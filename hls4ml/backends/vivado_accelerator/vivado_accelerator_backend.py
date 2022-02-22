import os

from hls4ml.backends import VivadoBackend
from hls4ml.model.flow import get_backend_flows, get_flow, register_flow

class VivadoAcceleratorBackend(VivadoBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VivadoAccelerator')
        self._register_flows()

    def make_bitfile(model):
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        try:
            os.system('vivado -mode batch -source design.tcl')
        except:
            print("Something went wrong, check the Vivado logs")
        # These should work but Vivado seems to return before the files are written...
        # copyfile('{}_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper.bit'.format(model.config.get_project_name()), './{}.bit'.format(model.config.get_project_name()))
        # copyfile('{}_vivado_accelerator/project_1.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh'.format(model.config.get_project_name()), './{}.hwh'.format(model.config.get_project_name()))
        os.chdir(curr_dir)

    def create_initial_config(self, board='pynq-z2', part=None, clock_period=5, io_type='io_parallel', interface='axi_stream',
                              driver='python', input_type='float', output_type='float'):
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

        Returns:
            populated config
        '''
        board = board if board is not None else 'pynq-z2'
        config = super(VivadoAcceleratorBackend, self).create_initial_config(part, clock_period, io_type)
        config['AcceleratorConfig'] = {}
        config['AcceleratorConfig']['Board'] = board
        config['AcceleratorConfig']['Interface'] = interface  # axi_stream, axi_master, axi_lite
        config['AcceleratorConfig']['Driver'] = driver
        config['AcceleratorConfig']['Precision'] = {}
        config['AcceleratorConfig']['Precision']['Input'] = {}
        config['AcceleratorConfig']['Precision']['Output'] = {}
        config['AcceleratorConfig']['Precision']['Input'] = input_type  # float, double or ap_fixed<a,b>
        config['AcceleratorConfig']['Precision']['Output'] = output_type  # float, double or ap_fixed<a,b>
        return config

    def _register_flows(self):
        #TODO expand this to include new accelerator flow
        parent_flows = get_backend_flows(backend='vivado')
        for flow_name in parent_flows:
            flow = get_flow(flow_name)
            acc_flow = register_flow(flow_name.replace('vivado:', ''), flow.optimizers, requires=flow.requires, backend=self.name)
            if ':write' in flow_name:
                self._writer_flow = acc_flow
        self._default_flow = 'vivadoaccelerator:ip'
