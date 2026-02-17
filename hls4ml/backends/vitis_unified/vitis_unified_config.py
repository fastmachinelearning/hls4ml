class VitisUnifiedConfig:
    def __init__(self, config, model_inputs, model_outputs):
        self.config = config.config
        self.board = self.config.get('VitisUnifiedConfig', {}).get('Board', 'pynq-z2')

        self.axi_mode = self.config['VitisUnifiedConfig']['axi_mode']

        if self.axi_mode not in ['axis', 'axim']:
            raise Exception('AXIMode must be either axis or axim')

        # before first and after last layer we have the configurable buffer
        # [platform]<-->[in_stream_buf_size]<-->[hls]<-->[out_stream_buf_size]<-->[platform]
        self.in_stream_buf_size = self.config['VitisUnifiedConfig']['in_stream_buf_size']
        self.out_stream_buf_size = self.config['VitisUnifiedConfig']['out_stream_buf_size']

        # the path to the generated platform
        self.XPFMPath = self.config['VitisUnifiedConfig']['XPFMPath']

        self.driver = self.config['VitisUnifiedConfig']['Driver']

        # c++ type for input and output of the hls kernel it must be str (float/double)
        self.input_type = self.config['VitisUnifiedConfig']['InputDtype']
        self.output_type = self.config['VitisUnifiedConfig']['OutputDtype']

        assert self.input_type == self.output_type, 'Input and Output data types must be the same type different'
        assert len(model_inputs) >= 1, 'Only models with at least one input tensor are currently supported by VitisUnified'
        assert len(model_outputs) >= 1, 'Only models with one output tensor are currently supported by VitisUnified'
        self.inps = model_inputs.copy()
        self.outs = model_outputs.copy()

    def get_corrected_types(self):
        return self.input_type, self.output_type, self.inps, self.outs

    def get_driver(self):
        return self.driver

    def get_board(self):
        return self.board

    def get_axi_mode(self):
        return self.axi_mode

    def get_input_type(self):
        return self.input_type

    def get_output_type(self):
        return self.output_type

    def get_in_stream_buf_size(self):
        return self.in_stream_buf_size

    def get_out_stream_buf_size(self):
        return self.out_stream_buf_size

    def get_XPFMPath(self):
        return self.XPFMPath
