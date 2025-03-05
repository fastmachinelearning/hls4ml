import os


def test_vivado_hls_availability():

    vivado_dir = '/cvmfs/projects.cern.ch/hls4ml/vivado/2020.1_v1/vivado-2020.1_v1'

    contents = os.listdir(vivado_dir)
    print("Contents of Vivado HLS bin directory:", contents)
