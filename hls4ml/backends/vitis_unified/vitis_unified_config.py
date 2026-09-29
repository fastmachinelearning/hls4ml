import os

from hls4ml.backends.vitis_unified.vitis_unified_validation import (
    load_supported_boards,
    mode_config,
    platform_file,
    platform_generator_tcl,
)


class VitisUnifiedConfig:
    def __init__(self, config, model_inputs, model_outputs):
        self.config = config.config
        unified_config = self.config['VitisUnifiedConfig']
        self.board = unified_config.get('Board', 'zcu102')
        self.axi_mode = unified_config['axi_mode']
        self.driver = unified_config['Driver']
        self.input_type = unified_config['InputDtype']
        self.output_type = unified_config['OutputDtype']
        self.supported_boards = load_supported_boards()

        # axi master buffer size
        # before first and after last layer we have the configurable buffer
        # [platform]<-->[in_stream_buf_size]<-->[hls]<-->[out_stream_buf_size]<-->[platform]
        self.in_stream_buf_size = unified_config['in_stream_buf_size']
        self.out_stream_buf_size = unified_config['out_stream_buf_size']

        # Platform is the user's own file, or resolved from supported_boards.json based on board + axi_mode
        board_info = self.supported_boards.get(self.board, {})
        platform = unified_config.get('Platform')
        tcl_rel = None if platform else platform_generator_tcl(board_info, self.axi_mode)
        if platform:
            self._platform_path = platform
            self._platform_generator_tcl = None
            self._platform_output_path = None
        elif tcl_rel:
            out_rel = mode_config(board_info, self.axi_mode).get('platform_output') or board_info.get(
                'platform_output', 'output/platform.xsa'
            )
            output_dir = config.get_output_dir()
            workspace_root = os.path.join(output_dir, 'vitis_workspace')
            tcl_path = tcl_rel if os.path.isabs(tcl_rel) else os.path.join(workspace_root, tcl_rel)
            out_path = out_rel if os.path.isabs(out_rel) else os.path.join(workspace_root, out_rel)
            self._platform_path = os.path.abspath(out_path)
            self._platform_generator_tcl = os.path.abspath(os.path.expanduser(tcl_path))
            self._platform_output_path = self._platform_path
        else:
            self._platform_path = self._get_xpfm_path_from_board()
            self._platform_generator_tcl = None
            self._platform_output_path = None

        self.inps = model_inputs.copy()
        self.outs = model_outputs.copy()

    def _get_xpfm_path_from_board(self):
        platform_rel = platform_file(self.supported_boards[self.board], self.axi_mode)
        if not platform_rel:
            raise Exception(
                f'No platform file for axi_mode "{self.axi_mode}" in supported_boards.json for board "{self.board}"'
            )
        # Resolve relative to XILINX_VITIS if path is relative
        if not os.path.isabs(platform_rel):
            return os.path.join('${XILINX_VITIS}', platform_rel)
        return platform_rel

    def get_board_info(self, board=None):
        board = board or self.board
        if board not in self.supported_boards:
            raise Exception(f'Board "{board}" is not supported')
        return self.supported_boards[board]

    def get_part(self):
        return self.get_board_info()['part']

    # main driver generation
    def get_driver_file(self):
        return f'{self.axi_mode}_driver.py'

    def get_driver_template_path(self):
        template_dir = os.path.join(os.path.dirname(__file__), '../../templates/vitis_unified/drivers')
        return os.path.join(template_dir, f'{self.get_driver_file()}.hls4ml')

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

    def get_platform_path(self):
        """Path to platform (.xpfm or .xsa)."""
        return self._platform_path

    def get_platform_generator_tcl(self):
        """Path to TCL script that generates platform, or None if using pre-built platform."""
        return self._platform_generator_tcl

    def get_platform_output_path(self):
        """Path where platform generator writes output, or None if not using generator."""
        return self._platform_output_path
