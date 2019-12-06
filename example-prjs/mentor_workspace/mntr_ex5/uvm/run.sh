#!/bin/bash
MODEL_TECH=$CAD_PATH/msim/modeltech/bin \
QUESTA_HOME=$CAD_PATH/msim/modeltech \
QHOME=$QHOME \
UVMF_HOME=$MGC_HOME/pkgs/uvmf/UVMF \
UVMF_VIP_LIBRARY_HOME=$PWD/asic/verification_ip \
CCS_UVMF_VIP_HOME=$MGC_HOME/pkgs/ccs_uvmf/vip \
UVMF_PROJECT_DIR=$PWD/asic/project_benches/mnist_mlp_bench \
PROJECT_HOME=$PWD/../syn-catapult-hls \
make -C asic/project_benches/mnist_mlp_bench/sim cli
