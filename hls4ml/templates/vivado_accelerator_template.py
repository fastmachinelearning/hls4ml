from hls4ml.templates.templates import get_supported_devices_dict
from hls4ml.templates.vivado_template import VivadoBackend
import os
from shutil import copyfile

class VivadoAcceleratorBackend(VivadoBackend):
    def __init__(self):
        super(VivadoAcceleratorBackend, self).__init__(name='VivadoAccelerator')

    def make_bitfile(model):
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        try:
            os.system('vivado -mode batch -source design.tcl')
        except:
            print("Something went wrong, check the Vivado logs")
        # These should work but Vivado seems to return before the files are written...
        #copyfile('{}_pynq/project_1.runs/impl_1/design_1_wrapper.bit'.format(model.config.get_project_name()), './{}.bit'.format(model.config.get_project_name()))
        #copyfile('{}_pynq/project_1.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh'.format(model.config.get_project_name()), './{}.hwh'.format(model.config.get_project_name()))
        os.chdir(curr_dir)
    
    def create_initial_config(self, device='pynq-z2', clock_period=5, io_type='io_parallel', interface='axi_stream',
                              driver='python', input_type='float', output_type='float'):
        '''
        Create initial accelerator config with default parameters
        Args:
            device: one of the keys defined in supported_devices.json
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
        device = device if device is not None else 'pynq-z2'
        devices = get_supported_devices_dict()
        if device in devices.keys():
            device_dict = devices[device]
            part = device_dict['part']
            driver_ext = '.py' if driver is 'python' else '.h'
            driver = device_dict[driver + '_drivers'][interface]
        else:
            raise Exception('The device is still not supported')
        config = super(VivadoAcceleratorBackend, self).create_initial_config(part, clock_period, io_type)
        config['AcceleratorConfig'] = {}
        config['AcceleratorConfig']['Interface'] = interface  # axi_stream, axi_master, axi_lite
        config['AcceleratorConfig']['Driver'] = device + '/' + driver + '_drivers/' + interface + '_driver' + driver_ext
        config['AcceleratorConfig']['Precision'] = {}
        config['AcceleratorConfig']['Precision']['Input'] = {}
        config['AcceleratorConfig']['Precision']['Output'] = {}
        config['AcceleratorConfig']['Precision']['Input']['Type'] = input_type  # float, double or ap_fixed<a,b>
        config['AcceleratorConfig']['Precision']['Input']['Bitwidth'] = {}
        config['AcceleratorConfig']['Precision']['Output']['Type'] = output_type  # float, double or ap_fixed<a,b>
        config['AcceleratorConfig']['Precision']['Output']['Bitwidth'] = {}
        config['HLSConfig'] = {}

        return config