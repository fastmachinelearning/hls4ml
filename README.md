# hls4snn

`hls4snn` is a fork of `hls4ml` focused on spiking neural network (SNN) functionality in the PyTorch frontend, while keeping the upstream `hls4ml` workflow and documentation structure.

## What's Different In This Fork

- Added initial SNN conversion support for PyTorch models using `snntorch` modules.
- Added SNN neuron/readout templates and related conversion backend plumbing.
- Added SNN-focused tests and documentation.

## SNN Quick Start

Install SNN development dependencies from this repository:

```bash
pip install -r requirements-snn-dev.txt
```

SNN feature details, supported modules, and behavior notes are documented in:

- [docs/advanced/snn.rst](docs/advanced/snn.rst)

A demo of the SNN functionality is in:

- [hls4ml-snn-example.ipynb](hls4ml-snn-example.ipynb)
- [snn-hls4ml-demo.ipynb](snn-hls4ml-demo.ipynb)

These take you from building and training a simple SNN in snntorch, to running it through hls4ml and getting Vitis reports. The longer demo also includes optional NeuroBench metrics.
