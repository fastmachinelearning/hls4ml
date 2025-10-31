import os

import pytest


def str_to_bool(val):
    return str(val).lower() in ("1", "true")


@pytest.fixture(scope="module")
def synthesis_config():
    """
    Fixture that provides synthesis configuration for tests.

    It gathers:
    - Whether synthesis should be run (from the RUN_SYNTHESIS env var)
    - Tool versions for each supported backend (from env vars)
    - Build arguments specific to each backend toolchain

    """
    return {
        "run_synthesis": str_to_bool(os.getenv("RUN_SYNTHESIS", "false")),
        "tools_version": {
            "Vivado": os.getenv("VIVADO_VERSION", "2020.1"),
            "Vitis": os.getenv("VITIS_VERSION", "2024.1"),
            "Quartus": os.getenv("QUARTUS_VERSION", "latest"),
            "oneAPI": os.getenv("ONEAPI_VERSION", "2025.0.1"),
        },
        "build_args": {
            "Vivado": {"csim": False, "synth": True, "export": False},
            "Vitis": {"csim": False, "synth": True, "export": False},
            "Quartus": {"synth": True, "fpgasynth": False},
            "oneAPI": {"build_type": "report", "run": False},
        },
    }
