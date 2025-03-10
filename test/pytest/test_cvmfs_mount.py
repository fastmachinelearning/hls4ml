import os
import subprocess

import pytest


def test_vivado_hls_availability():

    vivado_bin_dir = '/cvmfs/projects.cern.ch/hls4ml/vivado/2020.1_v1/vivado-2020.1_v1/opt/Xilinx/Vivado/2020.1/bin'

    try:
        contents = os.listdir(vivado_bin_dir)
        print("Contents of Vivado HLS bin directory:", contents)
    except Exception as e:
        print("Failed to list directory contents:", e)
        pytest.fail(f"Unable to access the directory {vivado_bin_dir}")

    os.environ['PATH'] += os.pathsep + vivado_bin_dir
    os.environ['XILINX_VIVADO'] = '/cvmfs/projects.cern.ch/hls4ml/vivado/2020.1_v1/vivado-2020.1_v1/opt/Xilinx/Vivado/2020.1'

    try:
        result = subprocess.run(['vivado', '-version'], capture_output=True, check=True, text=True)
        print("Vivado HLS Version Information:")
        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Failed to execute vivado for version check:", e)
        print(e.stderr)
        pytest.fail("Vivado HLS version check failed.")
