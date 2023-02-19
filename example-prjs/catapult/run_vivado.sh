#! /bin/bash

# This script runs the Catapult flows to generate the HLS.

VENV=$HOME/venv

export PATH=/wv/hlstools/python/python37/bin:$PATH:$XILINX_VIVADO/bin:$MGC_HOME/bin
export LD_LIBRARY_PATH=/wv/hlstools/python/python37/lib:$XILINX_VIVADO/lib/lnx64.o:$MGC_HOME/lib
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# needed for pytest
export OSTYPE=linux-gnu

echo "Activating Virtual Environment..."
#    bash
source $VENV/bin/activate

rm -rf ./my-Vivado-test*

python3 sample_config.py

echo ""
echo "====================================================="
echo "====================================================="
echo "C++ EXECUTION"
pushd my-Vivado-test; rm -f a.out; $MGC_HOME/bin/g++ -std=c++11 -I. -DWEIGHTS_DIR=\"firmware/weights\" -Ifirmware -Ifirmware/ap_types -I$MGC_HOME/shared/include firmware/myproject.cpp myproject_test.cpp; a.out; popd

