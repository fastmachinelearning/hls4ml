class VitisUnifiedConfig:

    def __init__(self, config, model_inputs, model_outputs):
        self.config = config.config
        self.board = self.config.get('UnifiedConfig', {}).get('Board', 'pynq-z2')

        self.axi_mode = self.config["UnifiedConfig"]["axi_mode"]

        if self.axi_mode not in ["axis", "axim"]:
            raise Exception("AXIMode must be either axis or axim")

        # before first and afterlast layer we have the configuratble buffer
        # [platform]<-->[in_steram_bufferSz]<-->[hls]<-->[out_stream_bufferSz]<-->[platform]
        self.in_steram_bufferSz = self.config["UnifiedConfig"]["in_stream_buf_Size"]
        self.out_stream_bufferSz = self.config["UnifiedConfig"]["out_stream_buf_Size"]

        # the path to the generated platform
        self.XPFMPath = self.config["UnifiedConfig"]["XPFMPath"]

        self.driver = self.config['UnifiedConfig']['Driver']

        # c++ type for input and output of the hls kernel it must be str (float/double)
        self.input_type = self.config['UnifiedConfig']['InputDtype']
        self.output_type = self.config['UnifiedConfig']['OutputDtype']

        assert self.input_type == self.output_type, "Input and Output data types must be the same type different"
        assert len(model_inputs) >= 1, "Only models with at least one input tensor are currently supported by VitisUnified"
        assert len(model_outputs) >= 1, "Only models with one output tensor are currently supported by VitisUnified"
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

    def get_in_stream_bufferSz(self):
        return self.in_steram_bufferSz

    def get_out_stream_bufferSz(self):
        return self.out_stream_bufferSz

    def get_XPFMPath(self):
        return self.XPFMPath
