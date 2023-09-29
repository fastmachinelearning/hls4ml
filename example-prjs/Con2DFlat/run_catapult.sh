#! /bin/bash

clear
printf '\033[3J'
# This script runs the Catapult flows to generate the HLS.

VENV=../../../../venv

MGC_HOME=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home
export MGC_HOME

export PATH=/wv/hlstools/python/python38/bin:$PATH:$XILINX_VIVADO/bin:$MGC_HOME/bin
export LD_LIBRARY_PATH=/wv/hlstools/python/python38/lib:$XILINX_VIVADO/lib/lnx64.o:$MGC_HOME/lib
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# needed for pytest
export OSTYPE=linux-gnu

echo "Activating Virtual Environment..."
#    bash
source $VENV/bin/activate

# run just the C++ execution
echo ""
echo "====================================================="
echo "====================================================="
echo "Running Catapult"
# pushd my-Catapult-test; $MGC_HOME/bin/catapult -f build_prj.tcl "csim=1 cosim=1 synth=1 vsynth=1 validation=1"; popd
# hls4ml build -p  my-Catapult-test -a
python3 catapult.py
