#! /bin/bash


# clear
# printf '\033[3J'

# # This script runs the Catapult flows to generate the HLS.

VENV=$HOME/venv

MGC_HOME=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home
export MGC_HOME

export PATH=/wv/hlstools/python/python38/bin:$PATH:$XILINX_VIVADO/bin:$MGC_HOME/bin
export LD_LIBRARY_PATH=/wv/hlstools/python/python38/lib:$XILINX_VIVADO/lib/lnx64.o:$MGC_HOME/lib
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


# run just the C++ execution
echo ""
echo "====================================================="
echo "====================================================="
echo "C++ EXECUTION"
rm -f a.out; $MGC_HOME/bin/g++ -g -std=c++17 -I. -DWEIGHTS_DIR=\"my-Catapult-test/firmware/weights\" -Imy-Catapult-test/firmware -I$MGC_HOME/shared/include my-Catapult-test/firmware/myproject.cpp my-Catapult-test/myproject_test.cpp;
perf record -g ./a.out
# python3 agrmax.py

