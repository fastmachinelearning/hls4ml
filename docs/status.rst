===================
Status and Features
===================

Status
========

The latest stable release is :doc:`v0.2.0 <release_notes>`, including a validated boosted decision tree implementation (`arXiv:2002.02534 <https://arxiv.org/abs/2002.02534>`_) and binary/ternary neural networks (\ `arXiv:2003.06308 <https://arxiv.org/abs/2003.06308>`_).


Features
========

A list of suppported ML codes and architectures, including a summary table is below.  Dependences are given in the :doc:`Setup <setup>` page.

ML code support: 


* Keras/Tensorflow, PyTorch, scikit-learn
* Planned: xgboost 

Neural network architectures:


* Fully Connected NNs (multi-layer perceptron)
* Boosted Decision Trees
* Convolutional NNs (1D/2D), in beta testing
* Recurrent NN/LSTM, in prototyping

A summary of the on-going status of the ``hls4ml`` tool is in the table below.

.. list-table::
   :header-rows: 1

   * - Architectures/Toolkits
     - Keras/TensorFlow
     - PyTorch
     - scikit-learn
   * - MLP
     - ``supported``
     - ``supported``
     - -
   * - Conv1D/Conv2D
     - ``supported``
     - ``in development``
     - -
   * - BDT
     - -
     - -
     - ``supported``
   * - RNN/LSTM
     - ``in development``
     - -
     - -


Other random feature notes:


* There is a known Vivado HLS issue where the large loop unrolls create memory issues during synthesis.  We are working to solve this issue but you may see errors related to this depending on the memory of your machine.  Please feel free to email the ``hls4ml`` team if you have any further questions.

Example Models
==============

We also provide and documented several example models that have been implemented in ``hls4ml`` in `this Github repository <https://github.com/hls-fpga-machine-learning/models>`_.
