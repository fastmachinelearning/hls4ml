#! /bin/bash

clear
printf '\033[3J'

# This script runs the Vivado flows to generate the HLS.

VENV=$HOME/venv

MGC_HOME=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home
export MGC_HOME

export PATH=/wv/hlstools/python/python37/bin:$PATH:$XILINX_VIVADO/bin:$MGC_HOME/bin
export LD_LIBRARY_PATH=/wv/hlstools/python/python37/lib:$XILINX_VIVADO/lib/lnx64.o:$MGC_HOME/lib
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# run just the C++ execution
echo ""
echo "====================================================="
echo "====================================================="
echo "C++ EXECUTION"
rm -f a.out; $MGC_HOME/bin/g++ -g -std=c++11 -I. -DWEIGHTS_DIR=\"my-Vivado-test/firmware/weights\" -Imy-Vivado-test/firmware -Imy-Vivado-test/firmware/ap_types -I$MGC_HOME/shared/include my-Vivado-test/firmware/myproject.cpp my-Vivado-test/myproject_test.cpp; a.out

