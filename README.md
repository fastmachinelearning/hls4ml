<p align="center">
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

For more information visit the webpage: [https://fastmachinelearning.org/hls4ml/](https://fastmachinelearning.org/hls4ml/).

For introductory material on FPGAs, HLS and ML inferences using hls4ml, check out the [video](https://www.youtube.com/watch?v=2y3GNY4tf7A&ab_channel=SystemsGroupatETHZ%C3%BCrich).

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

### Building a project.
We will build the project using Xilinx Vivado HLS, which can be downloaded and installed from [here](https://www.xilinx.com/products/design-tools/vivado/integration/esl-design.html). Alongside Vivado HLS, hls4ml also supports Vitis HLS, Intel HLS, Catapult HLS and has some experimental support dor Intel oneAPI. The target back-end can be changed using the argument backend when building the model.

```Python
# Use Vivado HLS to synthesize the model
# This might take several minutes
hls_model.build()

# Print out the report if you want
hls4ml.report.read_vivado_report('my-hls-test')
```

# FAQ

List of frequently asked questions and common HLS synthesis can be found [here](https://fastmachinelearning.org/hls4ml/faq.html)

# Citation
If you use this software in a publication, please cite the software
```bibtex
@software{fastml_hls4ml,
  author       = {{FastML Team}},
  title        = {fastmachinelearning/hls4ml},
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
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

# Acknowledgments
If you benefited from participating in our community, we ask that you please acknowledge the Fast Machine Learning collaboration, and particular individuals who helped you, in any publications.
Please use the following text for this acknowledgment:
  > We acknowledge the Fast Machine Learning collective as an open community of multi-domain experts and collaborators. This community and \<names of individuals\>, in particular, were important for the development of this project.

# Funding
We gratefully acknowledge previous and current support from the U.S. National Science Foundation (NSF) Harnessing the Data Revolution (HDR) Institute for <a href="https://a3d3.ai">Accelerating AI Algorithms for Data Driven Discovery (A3D3)</a> under Cooperative Agreement No. <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2117997">PHY-2117997</a>, U.S. Department of Energy (DOE) Office of Science, Office of Advanced Scientific Computing Research under the Real‐time Data Reduction Codesign at the Extreme Edge for Science (XDR) Project (<a href="https://science.osti.gov/-/media/grants/pdf/foas/2021/SC_FOA_0002501.pdf">DE-FOA-0002501</a>), DOE Office of Science, Office of High Energy Physics Early Career Research Program (<a href="https://pamspublic.science.energy.gov/WebPAMSExternal/Interface/Common/ViewPublicAbstract.aspx?rv=df0ae4ab-a46e-481a-9acc-3856b6b041e5&rtc=24&PRoleId=10">DE-SC0021187</a>, DE-0000247070), and the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (Grant No. <a href="https://doi.org/10.3030/772369">772369</a>).

<p align="center">
<img src="https://github.com/fastmachinelearning/hls4ml/assets/29201053/bd1217d4-9930-47b7-8917-ad3fc430c75d" alt="A3D3" width="130"/>
<img src="https://github.com/fastmachinelearning/hls4ml/assets/4932543/16e77374-9829-40a8-800e-8d12018a7cb3" alt="NSF" width="130"/>
<img src="https://github.com/fastmachinelearning/hls4ml/assets/4932543/de6ca6ea-4d1c-4c56-9d93-f759914bbbf9" alt="DOE" width="130"/>
<img src="https://github.com/fastmachinelearning/hls4ml/assets/4932543/7a369971-a381-4bb8-932a-7162b173cbac" alt="ERC" width="130"/>
</p>
