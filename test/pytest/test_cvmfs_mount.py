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


def test_vitis_availability():

    base_path = '/cvmfs/projects.cern.ch/hls4ml/vivado/2020.1_v1/vivado-2020.1_v1/opt/Xilinx/Vitis/2020.1'
    vitis_path = "/opt/Xilinx/Vitis/2020.1"
    original_paths = (
        "/opt/Xilinx/Vitis/2020.1/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/microblaze/lin/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/arm/lin/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/microblaze/linux_toolchain/lin64_le/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-none/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:"
        + "/opt/Xilinx/Vitis/2020.1/tps/lnx64/cmake-3.3.2/bin:"
        + "/opt/Xilinx/Vitis/2020.1/cardano/bin"
    )

    update_environment(base_path, original_paths, vitis_path)

    try:
        result = subprocess.run(['vitis', '-version'], capture_output=True, check=True, text=True)
        print("Vivado HLS Version Information:")
        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Failed to execute vivado for version check:", e)
        print(e.stderr)
        pytest.fail("Vivado HLS version check failed.")


def update_environment(base_path, original_paths, vivado_path):
    # Append the new paths to the PATH environment variable
    for path in original_paths.split(':'):
        full_path = os.path.join(base_path, path.lstrip('/'))  # Remove leading '/' to correctly join paths
        os.environ['PATH'] += os.pathsep + full_path

    # Set the XILINX_VIVADO environment variable
    os.environ['XILINX_VIVADO'] = os.path.join(base_path, vivado_path.lstrip('/'))
