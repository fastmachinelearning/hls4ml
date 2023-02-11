===================
Status and Features
===================

Status
========

The latest stable release is :doc:`v0.7.0 <release_notes>`.


Features
========

A list of supported ML codes and architectures, including a summary table is below.  Dependencies are given in the :doc:`Setup <setup>` page.

ML code support:

* Keras/Tensorflow/QKeras
* PyTorch (limited)
* (Q)ONNX (in development)

Neural network architectures:

* Fully connected NNs (multilayer perceptron, MLP)
* Convolutional NNs (1D and 2D)
* Recurrent NN (LSTM)
* Graph NN (GarNet)

HLS backends:

* Vivado HLS
* Vitis HLS (experimental)
* Intel HLS

A summary of the on-going status of the ``hls4ml`` tool is in the table below.

.. list-table::
   :header-rows: 1

   * - Architectures/Toolkits
     - Keras/TensorFlow/QKeras
     - PyTorch
     - (Q)ONNX
   * - MLP
     - ``supported``
     - ``supported``
     - ``supported``
   * - CNN
     - ``supported``
     - ``in development``
     - ``in development``
   * - RNN/LSTM
     - ``in development``
     - ``in development``
     - ``in development``


Other feature notes:

* ``hls4ml`` is tested on Linux, and supports Vivado HLS versions 2018.2 to 2020.1 and Intel HLS versions XXX. Vitis HLS is experimentally supported in v0.7.0. Windows and macOS are not supported.

* BDT support has moved to the `Conifer <https://github.com/thesps/conifer>`__ package

Example Models
==============

We also provide and documented several example models that have been implemented in ``hls4ml`` in `this Github repository <https://github.com/fastmachinelearning/example-models>`_.
