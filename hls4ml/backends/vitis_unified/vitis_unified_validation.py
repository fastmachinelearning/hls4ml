import json
import os


def load_supported_boards():
    path = os.path.join(os.path.dirname(__file__), 'supported_boards.json')
    with open(path) as f:
        return json.load(f)


def mode_config(board_info, axi_mode):
    config = board_info.get(axi_mode, {})
    return config if isinstance(config, dict) else {}


def platform_generator_tcl(board_info, axi_mode):
    tcl = mode_config(board_info, axi_mode).get('platform_generator_tcl')
    if tcl:
        return tcl
    tcl = board_info.get('platform_generator_tcl')
    return tcl.get(axi_mode) if isinstance(tcl, dict) else tcl


def platform_file(board_info, axi_mode):
    path = mode_config(board_info, axi_mode).get('platform_file')
    if path:
        return path
    path = board_info.get('platform_file')
    return path.get(axi_mode) if isinstance(path, dict) else path


def validate_config(board, axi_mode, driver, input_type, output_type, supported_boards=None, platform=None, part=None):
    supported_boards = supported_boards or load_supported_boards()
    if axi_mode not in ['axi_stream', 'axi_master']:
        raise Exception('axi_mode must be either axi_stream or axi_master')
    if platform is None:
        if board not in supported_boards:
            raise Exception(
                f'Board "{board}" does not appear in supported_boards.json. Available boards: {list(supported_boards)}. '
                'Pass platform and part to use a board that is not listed.'
            )
        board_info = supported_boards[board]
        if not platform_generator_tcl(board_info, axi_mode) and not platform_file(board_info, axi_mode):
            raise Exception(f'Board "{board}" has no platform for axi_mode "{axi_mode}" in supported_boards.json.')
    else:
        if not str(platform).endswith(('.xpfm', '.xsa')):
            raise Exception('platform must be a .xpfm or .xsa file')
        if board not in supported_boards and part is None:
            raise Exception(f'Board "{board}" is not in supported_boards.json, so part must be given together with platform')
    if driver != 'python':
        raise Exception('driver must be python; the current version only generates the PYNQ driver')
    for name, value in [('input_type', input_type), ('output_type', output_type)]:
        if value not in ['float', 'double']:
            raise Exception(f'{name} must be float or double')
    if input_type != output_type:
        raise Exception('input_type and output_type must be the same')
    if platform is None and axi_mode == 'axi_stream' and input_type == 'double':
        if platform_generator_tcl(supported_boards[board], axi_mode):
            raise Exception(
                f'The AXI DMA of the shipped {board} axi_stream platform is 32 bits wide, so input_type double is not '
                'supported with it. Use float, or pass your own platform with a 64-bit DMA to use double.'
            )
