import json
import os

import numpy as np

from hls4ml.model.layers import FixedPrecisionType, IntegerPrecisionType

class VitisUnifiedConfig:
    def __init__(self, config, model_inputs, model_outputs):
        self.config = config.config
        self.board = self.config.get('UnifiedConfig', {}).get('Board', 'pynq-z2')

        self.gmem_in_bufferSz  = self.config["UnifiedConfig"]["bufInSize"]
        self.gmem_out_bufferSz = self.config["UnifiedConfig"]["bufOutSize"]
        self.XPFMPath          = self.config["UnifiedConfig"]["XPFMPath"]

        self.driver            = self.config['UnifiedConfig']['Driver']
        self.input_type        = self.config['UnifiedConfig']['InputDtype' ]
        self.output_type       = self.config['UnifiedConfig']['OutputDtype']

        assert(
            self.input_type == self.output_type
        ), "Input and Output data types must be the same type different"
        assert (
            len(model_inputs) >= 1
        ), "Only models with at least one input tensor are currently supported by VitisUnified"
        assert (
            len(model_outputs) >= 1
        ), "Only models with one output tensor are currently supported by VitisUnified"
        self.inps = model_inputs.copy()
        self.outs = model_outputs.copy()
        inp_axi_t = self.input_type
        out_axi_t = self.output_type

        if self.input_type == 'float':
            self.input_bitwidth = 32
        else:# self.input_type == 'double':
            self.input_bitwidth = 64

        if out_axi_t == 'float':
            self.output_bitwidth = 32
        else:# out_axi_t == 'double':
            self.output_bitwidth = 64

    def get_io_bitwidth(self):
        return self.input_bitwidth, self.output_bitwidth

    def get_corrected_types(self):
        return self.input_type, self.output_type, self.inps, self.outs

    def get_driver(self):
        return self.driver

    def get_board(self):
        return self.board

    def get_input_type(self):
        return self.input_type

    def get_output_type(self):
        return self.output_type

    def get_gmem_in_bufferSz(self):
        return self.gmem_in_bufferSz

    def get_gmem_out_bufferSz(self):
        return self.gmem_out_bufferSz

    def get_XPFMPath(self):
        return self.XPFMPath