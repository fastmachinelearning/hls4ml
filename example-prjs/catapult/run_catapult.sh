#! /bin/bash

# This script runs the Catapult flows to generate the HLS.

VENV=$HOME/venv

MGC_HOME=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home
export MGC_HOME

export PATH=/wv/hlstools/python/python37/bin:$PATH:$XILINX_VIVADO/bin:$MGC_HOME/bin
export LD_LIBRARY_PATH=/wv/hlstools/python/python37/lib:$XILINX_VIVADO/lib/lnx64.o:$MGC_HOME/lib
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# needed for pytest
export OSTYPE=linux-gnu

echo "Activating Virtual Environment..."
#    bash
source $VENV/bin/activate

rm -rf ./my-Catapult-test*

# to run catapult+vivado_rtl
sed -e 's/Vivado/Catapult/g' sample_config.py >catapult.py
# to only run catapult
# sed -e 's/Vivado/Catapult/g' sample_config.py | sed -e 's/vsynth=True/vsynth=False/g' >catapult.py

# actually run HLS4ML + Catapult (+ optional vivado RTL)
python3 catapult.py

# run just the C++ execution
echo ""
echo "====================================================="
echo "====================================================="
echo "C++ EXECUTION"
pushd my-Catapult-test; rm -f a.out; $MGC_HOME/bin/g++ -std=c++17 -I. -DWEIGHTS_DIR=\"firmware/weights\" -Ifirmware -I$MGC_HOME/shared/include firmware/myproject.cpp myproject_test.cpp; a.out; popd

