import numpy as np

from hls4ml.model.hls_layers import FixedPrecisionType, IntegerPrecisionType
from hls4ml.templates import get_supported_devices_dict


class VivadoAcceleratorConfig(object):
    def __init__(self, config, model_inputs, model_outputs):
        self.config = config.config
        self.device = self.config.get('Device', 'pynq-z2')
        devices = get_supported_devices_dict()
        if self.device in devices.keys():
            device_info = devices[self.device]
            self.part = device_info['part']
        else:
            raise Exception('The device is still not supported')

        accel_config = self.config.get('AcceleratorConfig', None)
        if accel_config is not None:
            prec = accel_config.get('Precision')
            if prec is None:
                raise Exception('Precision must be provided in the AcceleratorConfig')
            else:
                if prec.get('Input') is None or prec.get('Output') is None:
                    raise Exception('Input and Output fields must be provided in the AcceleratorConfig->Precision')
        else:
            accel_config = {'Precision': {}}
            config['AcceleratorConfig'] = accel_config

        self.interface = self.config['AcceleratorConfig'].get('Interface',
                                                              'axi_stream')  # axi_stream, axi_master, axi_lite
        self.driver = self.config['AcceleratorConfig'].get('Driver', 'python')  # python or c
        self.input_type = self.config['AcceleratorConfig']['Precision'].get('Input',
                                                                            'float')  # float, double or ap_fixed<a,b>
        self.output_type = self.config['AcceleratorConfig']['Precision'].get('Output',
                                                                             'float')  # float, double or ap_fixed<a,b>

        assert len(
            model_inputs) == 1, "Only models with one input tensor are currently supported by VivadoAcceleratorBackend"
        assert len(
            model_outputs) == 1, "Only models with one output tensor are currently supported by VivadoAcceleratorBackend"
        self.inp = model_inputs[0]
        self.out = model_outputs[0]
        inp_axi_t = self.input_type
        out_axi_t = self.output_type

        if inp_axi_t not in ['float', 'double']:
            self.input_type = self._next_factor8_type(config.backend.convert_precision_string(inp_axi_t))
        if out_axi_t not in ['float', 'double']:
            self.output_type = self._next_factor8_type(config.backend.convert_precision_string(out_axi_t))

        if self.input_type is 'float':
            self.input_bitwidth = 32
        elif self.input_type is 'double':
            self.input_bitwidth = 64
        else:
            self.input_bitwidth = inp_axi_t.width

        if out_axi_t is 'float':
            self.output_bitwidth = 32
        elif out_axi_t is 'double':
            self.output_bitwidth = 64
        else:
            self.output_bitwidth = out_axi_t.width

    def _next_factor8_type(self, p):
        ''' Return a new type with the width rounded to the next factor of 8 up to p's width
            Args:
                p : IntegerPrecisionType or FixedPrecisionType
            Returns:
                An IntegerPrecisionType or FixedPrecisionType with the width rounder up to the next factor of 8
                of p's width. Other parameters (fractional bits, extra modes) stay the same.
        '''
        W = p.width
        newW = int(np.ceil(W / 8) * 8)
        if isinstance(p, FixedPrecisionType):
            return FixedPrecisionType(newW, p.integer, p.signed, p.rounding_mode, p.saturation_mode,
                                      p.saturation_bits)
        elif isinstance(p, IntegerPrecisionType):
            return IntegerPrecisionType(newW, p.signed)

    def get_io_bitwidth(self):
        return self.input_bitwidth, self.output_bitwidth

    def get_corrected_types(self):
        return self.input_type, self.output_type, self.inp, self.out

    def get_interface(self):
        return self.interface

    def get_device_info(self, device):
        devices = get_supported_devices_dict()
        if device in devices.keys():
            return devices[self.device]
        else:
            raise Exception('The device is still not supported')

    def get_part(self):
        return self.part

    def get_driver(self):
        return self.driver

    def get_device(self):
        return self.device

    def get_driver_path(self):
        return self.device + '/' + self.driver + '_drivers/' + self.get_driver_file()

    def get_driver_file(self):
        driver_ext = '.py' if self.driver is 'python' else '.h'
        return self.interface + '_driver' + driver_ext

    def get_input_type(self):
        return self.input_type

    def get_output_type(self):
        return self.output_type
