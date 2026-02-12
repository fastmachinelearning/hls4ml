import os

import pytest

# Characters that are problematic in file/directory names
_PROBLEMATIC_CHARS = ':;=%\'"<>|?*\\'


def _sanitize_test_id(s: str) -> str:
    """
    Sanitize a test identifier for use in paths.
    - remove .py
    - remove test/pytest/ prefix
    - : → _ (:: → _)
    - / → _
    - [ and ] → _
    - remove problematic chars: % ' " < > | ? * \\ etc.
    """
    s = s.replace('.py', '')
    s = s.replace('hls4ml/test/pytest/', '')
    s = s.replace('::', '_')
    s = s.replace('/', '_')
    s = s.replace('[', '_')
    s = s.replace(']', '_')
    for c in _PROBLEMATIC_CHARS:
        s = s.replace(c, '')
    return s.strip('_')


@pytest.fixture
def test_case_id(request):
    """
    Return a unique identifier for the current parametrized test case.
    Format: test_file_test_name_param_id (from test_file.py::test_name[param_id]).
    Used for generating output directory names.
    """
    return _sanitize_test_id(request.node.nodeid)


def str_to_bool(val):
    return str(val).lower() in ('1', 'true')


@pytest.fixture(scope='module')
def synthesis_config():
    """
    Fixture that provides synthesis configuration for tests.

    It gathers:
    - Whether synthesis should be run (from the RUN_SYNTHESIS env var)
    - Tool versions for each supported backend (from env vars)
    - Build arguments specific to each backend toolchain

    """
    return {
        'run_synthesis': str_to_bool(os.getenv('RUN_SYNTHESIS', 'false')),
        'tools_version': {
            'Vivado': os.getenv('VIVADO_VERSION', '2020.1'),
            'Vitis': os.getenv('VITIS_VERSION', '2024.1'),
            'Quartus': os.getenv('QUARTUS_VERSION', 'latest'),
            'oneAPI': os.getenv('ONEAPI_VERSION', '2025.0.1'),
        },
        'build_args': {
            'Vivado': {'csim': False, 'synth': True, 'export': False},
            'Vitis': {'csim': False, 'synth': True, 'export': False},
            'Quartus': {'synth': True, 'fpgasynth': False},
            'oneAPI': {'build_type': 'report', 'run': False},
        },
    }
