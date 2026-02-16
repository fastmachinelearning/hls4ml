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

A list of supported ML frameworks (Frontends), HLS backends, and neural network architectures, including a summary table is below.  Dependencies are given in the :doc:`Setup <setup>` page.

Frontend support:

* Keras

  * Keras v2

    * QKeras
    * HGQ
  * Keras v3

    * HGQ2
* PyTorch
* ONNX

  * QONNX

Neural network architectures:

* Fully connected NN (multilayer perceptron, MLP)
* Convolutional NN (1D and 2D)
* Recurrent NN (RNN, LSTM, GRU)
* GarNet
* Einsum and EinsumDense (Einsum)
* Multi-head attention (MHA) (experimental)

HLS backends:

* Vivado HLS
* Intel HLS
* Vitis HLS
* Catapult HLS
* oneAPI (experimental)

A summary of the on-going status of the ``hls4ml`` tool is in the table below.

.. table:: hls4ml Supported Features

+-----------------------+-----+-----+--------------+--------+--------+-----+
| Frontend/Backend      | MLP | CNN | RNN/LSTM/GRU | GarNet | Einsum | MHA |
+=======================+=====+=====+==============+========+========+=====+
| Keras v2              | ✅  | ✅  | ✅           | ✅     | ❌     | ❌  |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| QKeras                | ✅  | ✅  | ✅           | ✅     | N/A    | N/A |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| HGQ                   | ✅  | ✅  | N/A          | N/A    | N/A    | N/A |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| Keras v3              | ✅  | ✅  | ✅           | N/A    | ✅     | ❌  |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| HGQ2                  | ✅  | ✅  | N/A          | N/A    | ✅     | ✅  |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| Torch                 | ✅  | ✅  | ✅           | ❌     | ✅     | ❌  |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| ONNX                  | ✅  | ✅  | ❌           | ❌     | ❌     | ❌  |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| QONNX                 | ✅  | ✅  | ❌           | N/A    | N/A    | N/A |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| Vivado/Vitis HLS      | ✅  | ✅  | ✅           | ❌     | ✅     | ✅  |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| Intel HLS             | ✅  | ✅  | ✅           | ❌     | ❌     | ❌  |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| Catapult HLS          | ✅  | ✅  | ✅           | ❌     | ❌     | ❌  |
+-----------------------+-----+-----+--------------+--------+--------+-----+
| oneAPI (experimental) | ✅  | ✅  | ✅           | ❌     | ❌     | ❌  |
+-----------------------+-----+-----+--------------+--------+--------+-----+

Other feature notes:

* ``hls4ml`` is tested on the following platforms. Newer versions might work just fine, but try at your own risk.

  - Vivado HLS 2020.1. Older versions may work, but use at your own risk.
  - Intel HLS versions 20.1 to 21.4, versions > 21.4 have not been tested.
  - Vitis HLS versions 2022.2 to 2024.1. Versions > 2024.1 are less tested.
  - Catapult HLS versions 2024.1_1 to 2024.2
  - oneAPI versions 2024.1 to 2025.0. Any future versions are known to not work.

* ``hls4ml`` supports Linux [*]_ and requires python >=3.10. hls4ml does not require a specific Linux distribution version and we recommend following the requirements of the HLS tool you are using.
* Windows and macOS are not supported. Setting up ``hls4ml`` on these platforms, for example using the Windows Subsystem for Linux (WSL), should be possible, but we do not provide support for such use cases.
* BDT support has moved to the `Conifer <https://github.com/thesps/conifer>`__ package

.. [*] For compiling the projects for simulation or actual HLS. Otherwise, the code **may** be used on other platforms and it will likely to work. However, please note that Windows or other platforms are **not supported** in general and are not tested.

Example Models
==============

We also provide and document several example ``hls4ml`` models in `this GitHub repository <https://github.com/fastmachinelearning/example-models>`_, which is included as a submodule.
You can check it out by doing ``git submodule update --init --recursive`` from the top level directory of ``hls4ml``.
