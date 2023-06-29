#! /bin/bash

# This script runs the Vivado flows to generate the HLS.

VENV=$HOME/venv

MGC_HOME=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home
export MGC_HOME

export PATH=/wv/hlstools/python/python37/bin:$PATH:$XILINX_VIVADO/bin:$MGC_HOME/bin:/wv/hlstools/vivado/ixl/Vivado_HLS/2017.1/bin
export LD_LIBRARY_PATH=/wv/hlstools/python/python37/lib:$XILINX_VIVADO/lib/lnx64.o:$MGC_HOME/lib
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# needed for pytest
export OSTYPE=linux-gnu

echo "Activating Virtual Environment..."
#    bash
source $VENV/bin/activate

rm -rf ./my-Vivado-test*

mkdir -p tb_data

# to run catapult+vivado_rtl
sed -e 's/Vivado/Catapult/g' vivado.py >catapult.py
# to only run catapult
# sed -e 's/Vivado/Catapult/g' vivado.py | sed -e 's/vsynth=True/vsynth=False/g' >catapult.py

# actually run HLS4ML + Vivado HLS
python vivado.py

# run just the C++ execution
echo ""
echo "====================================================="
echo "====================================================="
echo "C++ EXECUTION"
pushd my-Vivado-test; rm -f a.out; $MGC_HOME/bin/g++ -g -std=c++11 -I. -DWEIGHTS_DIR=\"firmware/weights\" -Ifirmware -Ifirmware/ap_types -I$MGC_HOME/shared/include firmware/leaky_relu.cpp leaky_relu_test.cpp; a.out; popd

