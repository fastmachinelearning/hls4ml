#! /bin/bash

# Modify this script to have the correct path to Python 3.7 ...
PPATH=/wv/hlstools/python/python37
# ... and the path to your Catapult install ...
MGC_HOME=/wv/hlsb/CATAPULT/2022.2_1/UPDATE_1/aol/Mgc_home
# ... and where you want your python virtual env created
VENV=$HOME/venv

# The rest of this should not need changes

export PATH=$PPATH/bin:$PATH:$XILINX_VIVADO/bin:$MGC_HOME/bin
export LD_LIBRARY_PATH=$PPATH/lib:$XILINX_VIVADO/lib/lnx64.o:$MGC_HOME/lib

echo "Creating Python3 Virtual Environment..."
$PPATH/bin/python3 -m venv $VENV

cd $VENV

echo "Activating Virtual Environment..."
#    bash
source $VENV/bin/activate

echo "Installing packages..."
pip install --upgrade pip
pip install --upgrade tensorflow
pip install --upgrade torch
pip install --upgrade numpy
pip install --upgrade matplotlib
pip install --upgrade scikit-learn
pip install --upgrade pandas
pip install --upgrade pytest
pip install --upgrade pytest-cov
pip install --upgrade pydot
pip install --upgrade graphviz
pip install --upgrade pyDigitalWaveTools

# Install latest hls4ml
# pip install --upgrade hls4ml
# pip install --upgrade hls4ml[profiling]

# Install latest hls4ml from github 'master'
# git clone --recursive git@github.com:fastmachinelearning/hls4ml.git
# pip install -e $VENV/hls4ml

# Install latest hls4ml from github '0.6.0'
# git clone --recursive git@github.com:fastmachinelearning/hls4ml.git -b v0.6.0
# pip install -e $VENV/hls4ml

# Install pointing to Siemens hls4ml under github
git clone --recursive git@github.com:dgburnette/hls4ml.git
pip install -e $VENV/hls4ml

# Install pointing to Siemens hls4ml under Perforce
# pip install -e $HOME/sb/sif/subprojs/sif_toolkits/src/HLS4ML/hls4ml

pip install git+https://github.com/google/qkeras.git@master
#pip install --upgrade opencv-python
#pip install --upgrade calmjs
#pip install --upgrade tabulate
#pip install --upgrade plotting
#pip install --upgrade pprint

echo "Done."
echo "Execute: 'source $VENV/bin/activate' in a new shell to get environment configured"

