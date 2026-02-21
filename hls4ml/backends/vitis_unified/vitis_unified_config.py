import json
import os


class VitisUnifiedConfig:
    def __init__(self, config, model_inputs, model_outputs):
        self.config = config.config
        self.board = self.config.get('VitisUnifiedConfig', {}).get('Board', 'zcu102')
        self.supported_boards = self._load_supported_boards()

        if self.board not in self.supported_boards:
            raise Exception(
                f'Board "{self.board}" does not appear in supported_boards.json. '
                f'Available boards: {list(self.supported_boards.keys())}'
            )

        self.axi_mode = self.config['VitisUnifiedConfig']['axi_mode']

        if self.axi_mode not in ['axi_stream', 'axi_master']:
            raise Exception('AXIMode must be either axi_stream or axi_master')

        # before first and after last layer we have the configurable buffer
        # [platform]<-->[in_stream_buf_size]<-->[hls]<-->[out_stream_buf_size]<-->[platform]
        self.in_stream_buf_size = self.config['VitisUnifiedConfig']['in_stream_buf_size']
        self.out_stream_buf_size = self.config['VitisUnifiedConfig']['out_stream_buf_size']

        # Platform is resolved from supported_boards.json based on board + axi_mode
        board_info = self.supported_boards.get(self.board, {})
        mode_config = board_info.get(self.axi_mode, {})
        if not isinstance(mode_config, dict):
            mode_config = {}

        tcl_rel = mode_config.get('platform_generator_tcl') or (
            board_info.get('platform_generator_tcl', {}).get(self.axi_mode)
            if isinstance(board_info.get('platform_generator_tcl'), dict)
            else board_info.get('platform_generator_tcl')
        )

        if tcl_rel:
            out_rel = mode_config.get('platform_output') or board_info.get('platform_output', 'output/platform.xsa')
            output_dir = config.get_output_dir()
            workspace_root = os.path.join(output_dir, 'vitis_workspace')
            tcl_path = tcl_rel if os.path.isabs(tcl_rel) else os.path.join(workspace_root, tcl_rel)
            self._platform_generator_tcl = os.path.abspath(os.path.expanduser(tcl_path))
            if not os.path.isabs(out_rel):
                out_path = os.path.join(workspace_root, out_rel)
            else:
                out_path = out_rel
            self._platform_output_path = os.path.abspath(out_path)
            self._platform_path = self._platform_output_path
        elif mode_config.get('platform_file') or (
            board_info.get('platform_file') and board_info['platform_file'].get(self.axi_mode)
        ):
            self._platform_path = self._get_xpfm_path_from_board()
            self._platform_generator_tcl = None
            self._platform_output_path = None
        else:
            raise Exception(f'Board "{self.board}" has no platform for axi_mode "{self.axi_mode}" in supported_boards.json.')

        self.driver = self.config['VitisUnifiedConfig']['Driver']

        # c++ type for input and output of the hls kernel it must be str (float/double)
        self.input_type = self.config['VitisUnifiedConfig']['InputDtype']
        self.output_type = self.config['VitisUnifiedConfig']['OutputDtype']

        assert self.input_type == self.output_type, 'Input and Output data types must be the same type different'
        assert len(model_inputs) >= 1, 'Only models with at least one input tensor are currently supported by VitisUnified'
        assert len(model_outputs) >= 1, 'Only models with one output tensor are currently supported by VitisUnified'
        self.inps = model_inputs.copy()
        self.outs = model_outputs.copy()

    def _load_supported_boards(self):
        path = os.path.join(os.path.dirname(__file__), 'supported_boards.json')
        with open(path) as f:
            return json.load(f)

    def _get_xpfm_path_from_board(self):
        board_info = self.supported_boards[self.board]
        mode_config = board_info.get(self.axi_mode, {})
        if isinstance(mode_config, dict) and mode_config.get('platform_file'):
            platform_rel = mode_config['platform_file']
        else:
            platform_file = board_info.get('platform_file')
            if not platform_file:
                raise Exception(f'No platform_file definition for board "{self.board}" in supported_boards.json')
            platform_rel = platform_file.get(self.axi_mode) if isinstance(platform_file, dict) else platform_file
        if not platform_rel:
            raise Exception(
                f'No platform file for axi_mode "{self.axi_mode}" in supported_boards.json for board "{self.board}"'
            )
        # Resolve relative to XILINX_VITIS if path is relative
        if not os.path.isabs(platform_rel):
            xilinx_vitis = os.environ.get('XILINX_VITIS', '/opt/Xilinx/Vitis/2023.2')
            return os.path.join(xilinx_vitis, platform_rel)
        return platform_rel

    def get_board_info(self, board=None):
        board = board or self.board
        if board not in self.supported_boards:
            raise Exception(f'Board "{board}" is not supported')
        return self.supported_boards[board]

    def get_part(self):
        return self.get_board_info()['part']

    def get_driver_file(self):
        """Return driver filename for current board and axi_mode (from supported_boards)."""
        board_info = self.get_board_info()
        drivers = board_info.get('python_drivers' if self.driver == 'python' else 'c_drivers', {})
        return drivers.get(self.axi_mode)

    def get_driver_template_path(self):
        """Return absolute path to driver template for current board and axi_mode.

        Derives path from python_drivers in supported_boards: {board}/python_drivers/{driver_file}.hls4ml
        """
        board_info = self.get_board_info()
        driver_file = board_info.get('python_drivers', {}).get(self.axi_mode)
        if not driver_file:
            raise Exception(
                f'No python_driver for axi_mode "{self.axi_mode}" in supported_boards.json for board "{self.board}"'
            )
        template_rel = f'{self.board}/python_drivers/{driver_file}.hls4ml'
        return os.path.join(os.path.dirname(__file__), '../../templates/vitis_unified', template_rel)

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
        return self._platform_path

    def get_platform_path(self):
        """Path to platform (.xpfm or .xsa). Alias for get_XPFMPath for backward compatibility."""
        return self._platform_path

    def get_platform_generator_tcl(self):
        """Path to TCL script that generates platform, or None if using pre-built platform."""
        return getattr(self, '_platform_generator_tcl', None)

    def get_platform_output_path(self):
        """Path where platform generator writes output, or None if not using generator."""
        return getattr(self, '_platform_output_path', None)
