<p float="left">
   <img src="https://fastmachinelearning.github.io/hls4ml/img/logo.jpg" alt="hls4ml" width="400"/>
</p>

[![DOI](https://zenodo.org/badge/108329371.svg)](https://zenodo.org/badge/latestdoi/108329371)
[![PyPI version](https://badge.fury.io/py/hls4ml.svg)](https://badge.fury.io/py/hls4ml)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/hls4ml.svg)](https://pypi.org/project/hls4ml/)

A package for machine learning inference in FPGAs. We create firmware implementations of machine learning algorithms using high level synthesis language (HLS). We translate traditional open-source machine learning package models into HLS that can be configured for your use-case!

**Contact:** hls4ml.help@gmail.com

# Documentation & Tutorial

For more information visit the webpage: [https://fastmachinelearning.org/hls4ml/](https://fastmachinelearning.org/hls4ml/)

Detailed tutorials on how to use `hls4ml`'s various functionalities can be found [here](https://github.com/hls-fpga-machine-learning/hls4ml-tutorial).

# Installation
```
pip install hls4ml
```

To install the extra dependencies for profiling: 

```
pip install hls4ml[profiling]
```

# Getting Started
### Creating an HLS project
```Python
import hls4ml

#Fetch a keras model from our example repository
#This will download our example model to your working directory and return an example configuration file
config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')

print(config) #You can print the configuration to see some default parameters

#Convert it to a hls project
hls_model = hls4ml.converters.keras_to_hls(config)

# Print full list of example models if you want to explore more
hls4ml.utils.fetch_example_list()
```

### Building a project with Xilinx Vivado HLS (after downloading and installing from [here](https://www.xilinx.com/products/design-tools/vivado/integration/esl-design.html))
Note: Vitis HLS is not yet supported. Vivado HLS versions between 2018.2 and 2020.1 are recommended.

```Python
#Use Vivado HLS to synthesize the model
#This might take several minutes
hls_model.build()

#Print out the report if you want
hls4ml.report.read_vivado_report('my-hls-test')
```
