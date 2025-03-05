import os
import subprocess

import pytest


def test_vivado_hls_availability():

    vivado_bin_dir = '/cvmfs/projects.cern.ch/hls4ml/vivado/2020.1_v1/vivado-2020.1_v1/bin'

    try:
        contents = os.listdir(vivado_bin_dir)
        print("Contents of Vivado HLS bin directory:", contents)
    except Exception as e:
        print("Failed to list directory contents:", e)
        pytest.fail(f"Unable to access the directory {vivado_bin_dir}")

    os.environ['PATH'] += os.pathsep + vivado_bin_dir

    try:
        result = subprocess.run(['vivado', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("Vivado HLS Version Information:")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Failed to execute vivado_hls for version check:", e)
        pytest.fail("Vivado HLS version check failed.")
