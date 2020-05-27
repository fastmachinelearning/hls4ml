<p float="left">
   <img src="https://hls-fpga-machine-learning.github.io/hls4ml/img/logo.jpg" alt="hls4ml" width="400"/>
</p>

[![DOI](https://zenodo.org/badge/108329371.svg)](https://zenodo.org/badge/latestdoi/108329371)
[![PyPI version](https://badge.fury.io/py/hls4ml.svg)](https://badge.fury.io/py/hls4ml)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/hls4ml.svg)](https://pypi.org/project/hls4ml/)

A package for machine learning inference in FPGAs. We create firmware implementations of machine learning algorithms using high level synthesis language (HLS). We translate traditional open-source machine learning package models into HLS that can be configured for your use-case!

contact: hls4ml.help@gmail.com

For more information visit the webpage: [https://hls-fpga-machine-learning.github.io/hls4ml/](https://hls-fpga-machine-learning.github.io/hls4ml/)


# Installation
```
pip install hls4ml
```

# Getting Started
### Creating an HLS project
```
wget https://raw.githubusercontent.com/hls-fpga-machine-learning/hls4ml/master/example-models/keras/KERAS_3layer.json -P keras
wget https://raw.githubusercontent.com/hls-fpga-machine-learning/hls4ml/master/example-models/keras/KERAS_3layer_weights.h5 -P keras
wget https://raw.githubusercontent.com/hls-fpga-machine-learning/hls4ml/master/example-models/keras-config.yml
hls4ml convert -c keras-config.yml
```

### Building a project with Xilinx Vitis (after downloading and installing from [here](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html))
```
hls4ml build -cs -p my-hls-test
```
