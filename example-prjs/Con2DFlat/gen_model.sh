#! /bin/bash
rm -rf tb_data my-* conv2d* a.out perf.data
mkdir tb_data
# This script runs the Vivado flows to generate the TensorFlow Keras Model.
VENV=../../../../venv

MGC_HOME=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home
export MGC_HOME

export PATH=/wv/hlstools/python/python38/bin:$PATH:$XILINX_VIVADO/bin:$MGC_HOME/bin
export LD_LIBRARY_PATH=/wv/hlstools/python/python38/lib:$XILINX_VIVADO/lib/lnx64.o:$MGC_HOME/lib
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# needed for pytest
export OSTYPE=linux-gnu

echo "Activating Virtual Environment..."
#bash
source $VENV/bin/activate

# actually run HLS4ML + Vivado HLS
python3 model.py
