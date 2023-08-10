<p float="left">
   <img src="https://github.com/fastmachinelearning/fastmachinelearning.github.io/raw/master/images/hls4ml_logo.svg" alt="hls4ml" width="400"/>
</p>

[![DOI](https://zenodo.org/badge/108329371.svg)](https://zenodo.org/badge/latestdoi/108329371)
[![License](https://img.shields.io/badge/License-Apache_2.0-red.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://github.com/fastmachinelearning/hls4ml/actions/workflows/build-sphinx.yml/badge.svg)](https://fastmachinelearning.org/hls4ml)
[![PyPI version](https://badge.fury.io/py/hls4ml.svg)](https://badge.fury.io/py/hls4ml)
[![Downloads](https://static.pepy.tech/personalized-badge/hls4ml?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/hls4ml)
<a href="https://anaconda.org/conda-forge/hls4ml/"><img alt="conda-forge" src="https://img.shields.io/conda/dn/conda-forge/hls4ml.svg?label=conda-forge"></a>

A package for machine learning inference in FPGAs. We create firmware implementations of machine learning algorithms using high level synthesis language (HLS). We translate traditional open-source machine learning package models into HLS that can be configured for your use-case!

If you have any questions, comments, or ideas regarding hls4ml or just want to show us how you use hls4ml, don't hesitate to reach us through the [discussions](https://github.com/fastmachinelearning/hls4ml/discussions) tab.

# Documentation & Tutorial

For more information visit the webpage: [https://fastmachinelearning.org/hls4ml/](https://fastmachinelearning.org/hls4ml/)

Detailed tutorials on how to use `hls4ml`'s various functionalities can be found [here](https://github.com/hls-fpga-machine-learning/hls4ml-tutorial).

# Installation
```bash
pip install hls4ml
```

To install the extra dependencies for profiling:

```bash
pip install hls4ml[profiling]
```

# Getting Started
### Creating an HLS project
```Python
import hls4ml

# Fetch a keras model from our example repository
# This will download our example model to your working directory and return an example configuration file
config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')

# You can print the configuration to see some default parameters
print(config)

# Convert it to a hls project
hls_model = hls4ml.converters.keras_to_hls(config)

# Print full list of example models if you want to explore more
hls4ml.utils.fetch_example_list()
```

### Building a project with Xilinx Vivado HLS (after downloading and installing from [here](https://www.xilinx.com/products/design-tools/vivado/integration/esl-design.html))
Note: Vitis HLS is not yet supported. Vivado HLS versions between 2018.2 and 2020.1 are recommended.

```Python
# Use Vivado HLS to synthesize the model
# This might take several minutes
hls_model.build()

# Print out the report if you want
hls4ml.report.read_vivado_report('my-hls-test')
```

# Citation
If you use this software in a publication, please cite the software
```bibtex
@software{fastml_hls4ml,
  author       = {{FastML Team}},
  title        = {fastmachinelearning/hls4ml},
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.7.1},
  doi          = {10.5281/zenodo.1201549},
  url          = {https://github.com/fastmachinelearning/hls4ml}
}
```
and first publication:
```bibtex
@article{Duarte:2018ite,
    author = "Duarte, Javier and others",
    title = "{Fast inference of deep neural networks in FPGAs for particle physics}",
    eprint = "1804.06913",
    archivePrefix = "arXiv",
    primaryClass = "physics.ins-det",
    reportNumber = "FERMILAB-PUB-18-089-E",
    doi = "10.1088/1748-0221/13/07/P07027",
    journal = "JINST",
    volume = "13",
    number = "07",
    pages = "P07027",
    year = "2018"
}
```
Additionally, if you use specific features developed in later papers, please cite those as well. For example, CNNs:
```bibtex
@article{Aarrestad:2021zos,
    author = "Aarrestad, Thea and others",
    title = "{Fast convolutional neural networks on FPGAs with hls4ml}",
    eprint = "2101.05108",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    reportNumber = "FERMILAB-PUB-21-130-SCD",
    doi = "10.1088/2632-2153/ac0ea1",
    journal = "Mach. Learn. Sci. Tech.",
    volume = "2",
    number = "4",
    pages = "045015",
    year = "2021"
}
@article{Ghielmetti:2022ndm,
    author = "Ghielmetti, Nicol\`{o} and others",
    title = "{Real-time semantic segmentation on FPGAs for autonomous vehicles with hls4ml}",
    eprint = "2205.07690",
    archivePrefix = "arXiv",
    primaryClass = "cs.CV",
    reportNumber = "FERMILAB-PUB-22-435-PPD",
    doi = "10.1088/2632-2153/ac9cb5",
    journal ="Mach. Learn. Sci. Tech.",
    year = "2022"
}
```
binary/ternary networks:
```bibtex
@article{Loncar:2020hqp,
    author = "Ngadiuba, Jennifer and others",
    title = "{Compressing deep neural networks on FPGAs to binary and ternary precision with HLS4ML}",
    eprint = "2003.06308",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    reportNumber = "FERMILAB-PUB-20-167-PPD-SCD",
    doi = "10.1088/2632-2153/aba042",
    journal = "Mach. Learn. Sci. Tech.",
    volume = "2",
    pages = "015001",
    year = "2021"
}
```
