===================
Status and Features
===================

Status
======

The latest version (built from ``main``) is |version|.
The stable version (released on PyPI) is |release|.
See the :ref:`Release Notes` section for a changelog.


Features
========

A list of supported ML frameworks, HLS backends, and neural network architectures, including a summary table is below.  Dependencies are given in the :doc:`Setup <setup>` page.

ML framework support:

* (Q)Keras
* PyTorch
* (Q)ONNX

Neural network architectures:

* Fully connected NN (multilayer perceptron, MLP)
* Convolutional NN
* Recurrent NN (LSTM)
* Graph NN (GarNet)

HLS backends:

* Vivado HLS
* Intel HLS
* Vitis HLS
* Catapult HLS
* oneAPI (experimental)

A summary of the on-going status of the ``hls4ml`` tool is in the table below.

.. list-table::
   :header-rows: 1

   * - ML framework/HLS backend
     - (Q)Keras
     - PyTorch
     - (Q)ONNX
     - Vivado HLS
     - Intel HLS
     - Vitis HLS
     - Catapult HLS
     - oneAPI
   * - MLP
     - ``supported``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``experimental``
   * - CNN
     - ``supported``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``experimental``
   * - RNN (LSTM)
     - ``supported``
     - ``supported``
     - ``N/A``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``supported``
     - ``experimental``
   * - GNN (GarNet)
     - ``supported``
     - ``in development``
     - ``N/A``
     - ``N/A``
     - ``N/A``
     - ``N/A``
     - ``N/A``
     - ``N/A``

Other feature notes:

* ``hls4ml`` is tested on the following platforms. Newer versions might work just fine, but try at your own risk.
   * Vivado HLS versions 2018.2 to 2020.1
   * Intel HLS versions 20.1 to 21.4, versions \> 21.4 have not been tested.
   * Vitis HLS versions 2022.2 to 2024.1. Versions \<= 2022.1 are known not to work.
   * Catapult HLS versions 2024.1_1 to 2024.2
   * oneAPI versions 2024.1 to 2025.0

* ``hls4ml`` supports Linux and requires python \>=3.10. hlsml does not require a specific Linux distribution version and we recommended to follow the requirements of the HLS tool you are using.
* Windows and macOS are not supported. Setting up ``hls4ml`` on these platforms, for example using the Windows Subsystem for Linux (WSL) should be possible, but we do not provide support for such use cases.
* BDT support has moved to the `Conifer <https://github.com/thesps/conifer>`__ package

Example Models
==============

We also provide and document several example ``hls4ml`` models in `this GitHub repository <https://github.com/fastmachinelearning/example-models>`_, which is included as a submodule.
You can check it out by doing ``git submodule update --init --recursive`` from the top level directory of ``hls4ml``.
