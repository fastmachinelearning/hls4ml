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
* PyTorch (limited)
* (Q)ONNX (in development)

Neural network architectures:

* Fully connected NN (multilayer perceptron, MLP)
* Convolutional NN
* Recurrent NN (LSTM)
* Graph NN (GarNet)

HLS backends:

* Vivado HLS
* Intel HLS
* Vitis HLS (experimental)

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
   * - MLP
     - ``supported``
     - ``limited``
     - ``in development``
     - ``supported``
     - ``supported``
     - ``experimental``
   * - CNN
     - ``supported``
     - ``limited``
     - ``in development``
     - ``supported``
     - ``supported``
     - ``experimental``
   * - RNN (LSTM)
     - ``supported``
     - ``N/A``
     - ``in development``
     - ``supported``
     - ``supported``
     - ``N/A``
   * - GNN (GarNet)
     - ``supported``
     - ``N/A``
     - ``N/A``
     - ``N/A``
     - ``N/A``
     - ``N/A``


Other feature notes:

* ``hls4ml`` is tested on Linux, and supports
   * Vivado HLS versions 2018.2 to 2020.1
   * Intel HLS versions 20.1 to 21.4
   * Vitis HLS versions 2022.2 to 2024.1
* Windows and macOS are not supported
* BDT support has moved to the `Conifer <https://github.com/thesps/conifer>`__ package

Example Models
==============

We also provide and document several example ``hls4ml`` models in `this GitHub repository <https://github.com/fastmachinelearning/example-models>`_, which is included as a submodule.
You can check it out by doing ``git submodule update --init --recursive`` from the top level directory of ``hls4ml``.
