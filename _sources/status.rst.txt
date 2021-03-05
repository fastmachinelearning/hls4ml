===================
Status and Features
===================

Status
========

The latest stable release is :doc:`v0.5.0 <release_notes>`. This release brings the new `IOType: io_stream` and support for larger CNN models, see: <https://arxiv.org/abs/2101.05108>.


Features
========

A list of suppported ML codes and architectures, including a summary table is below.  Dependences are given in the :doc:`Setup <setup>` page.

ML code support: 

* Keras/Tensorflow/QKeras, PyTorch, Onnx

Neural network architectures:

* Fully Connected NNs (multi-layer perceptron)
* Convolutional NNs (1D/2D)
* Recurrent NN/LSTM, in prototyping

A summary of the on-going status of the ``hls4ml`` tool is in the table below.

.. list-table::
   :header-rows: 1

   * - Architectures/Toolkits
     - Keras/TensorFlow/QKeras
     - PyTorch
     - ONNX
   * - MLP
     - ``supported``
     - ``supported``
     - ``supported``
   * - Conv1D/Conv2D
     - ``supported``
     - ``in development``
     - ``in development`` 
   * - RNN/LSTM
     - ``in development``
     - ``in development``
     - ``in development``


Other feature notes:

* ``hls4ml`` is tested on Linux, and supports Vivado HLS versions 2018.2 to 2020.1. Vitis HLS is not yet supported. Windows and macOS are not supported.

* BDT support has moved to the `Conifer <https://github.com/thesps/conifer>`__ package

Example Models
==============

We also provide and documented several example models that have been implemented in ``hls4ml`` in `this Github repository <https://github.com/fastmachinelearning/example-models>`_.

