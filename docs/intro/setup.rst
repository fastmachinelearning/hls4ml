=====================
Setup and Quick Start
=====================

Getting started with ``hls4ml`` is very easy. There are several installation options available and once installed,
it takes only a few lines of code to run your first synthesis.

Installation
============

The latest release of ``hls4ml`` can be installed with ``pip``:

.. code-block::

   pip install hls4ml

If you want to use our :doc:`profiling <../advanced/profiling>` toolbox, you might need to install extra dependencies:

.. code-block::

   pip install hls4ml[profiling]

``hls4ml`` is also available as a ``conda`` package in the ``conda-forge`` repository. To install, run:

.. warning::
   Version of hls4ml available on ``conda-forge`` is outdated, we recommend installing with ``pip`` to get the latest version.

.. code-block::

   conda install -c conda-forge hls4ml

Development version
-------------------

``hls4ml`` is rapidly evolving and many experimental features and bugfixes are available on the development branch. Development
version can be installed directly from ``git``:

.. code-block::

   pip install git+https://github.com/fastmachinelearning/hls4ml@main


Dependencies
============

The ``hls4ml`` library requires python 3.10 or later, and depends on a number of Python packages and external tools for synthesis and simulation. Python dependencies are automatically managed
by ``pip`` or ``conda``.

* `TensorFlow <https://pypi.org/project/tensorflow/>`_ (version 2.8 to 2.14) and `QKeras <https://pypi.org/project/qkeras/>`_ are required by the Keras converter. One may want to install newer versions of QKeras from GitHub. Newer versions of TensorFlow can be used, but QKeras and hl4ml do not currently support Keras v3.

* `ONNX <https://pypi.org/project/onnx/>`_ (version 1.4.0 and newer) is required by the ONNX converter.

* `PyTorch <https://pytorch.org/get-started>`_ package is optional. If not installed, the PyTorch converter will not be available.

Running C simulation from Python requires a C++11-compatible compiler. On Linux, a GCC C++ compiler ``g++`` is required. Any version from a recent
Linux should work. On MacOS, the *clang*-based ``g++`` is enough. For the oneAPI backend, one must have oneAPI installed, along with the FPGA compiler,
to run C/SYCL simulations.

To run FPGA synthesis, installation of following tools is required:

* Xilinx Vivado HLS 2018.2 to 2020.1 for synthesis for Xilinx FPGAs using the ``Vivado`` backend.

* Vitis HLS 2022.2 or newer is required for synthesis for Xilinx FPGAs using the ``Vitis`` backend.

* Intel Quartus 20.1 to 21.4 for the synthesis for Intel/Altera FPGAs using the ``Quartus`` backend.

* oneAPI 2024.1 to 2025.0 with the FPGA compiler and recent Intel/Altera Quartus for Intel/Altera FPGAs using the ``oneAPI`` backend.

Catapult HLS 2024.1_1 or 2024.2 can be used to synthesize both for ASICs and FPGAs.


Quick Start
=============

For basic concepts to understand the tool, please visit the :doc:`Concepts <../api/concepts>` chapter.
Here we give line-by-line instructions to demonstrate the general workflow.

.. code-block:: python

   import hls4ml

   # Fetch a keras model from our example repository
   # This will download our example model to your working directory and return an example configuration file
   config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')

   # You can print it to see some default parameters
   print(config)

   # Convert it to a hls project
   hls_model = hls4ml.converters.keras_to_hls(config)

   # Print full list of example model if you want to explore more
   hls4ml.utils.fetch_example_list()

After that, you can use :code:`Vivado HLS` to synthesize the model:

.. code-block:: python

   # Use Vivado HLS to synthesize the model
   # This might take several minutes
   hls_model.build()

   # Print out the report if you want
   hls4ml.report.read_vivado_report('my-hls-test')

Done! You've built your first project using ``hls4ml``! To learn more about our various API functionalities, check out our tutorials `here <https://github.com/fastmachinelearning/hls4ml-tutorial>`__.

If you want to configure your model further, check out our :doc:`Configuration <../api/configuration>` page.

..
   Apart from our main API, we also support model conversion using a command line interface, check out our next section to find out more:

   Getting started with hls4ml CLI (deprecated)
   --------------------------------------------

   As an alternative to the recommended Python PI, the command-line interface is provided via the ``hls4ml`` command.

   To follow this tutorial, you must first download our ``example-models`` repository:

   .. code-block:: bash

      git clone https://github.com/fastmachinelearning/example-models

   Alternatively, you can clone the ``hls4ml`` repository with submodules

   .. code-block:: bash

      git clone --recurse-submodules https://github.com/fastmachinelearning/hls4ml

   The model files, along with other configuration parameters, are defined in the ``.yml`` files.
   Further information about ``.yml`` files can be found in :doc:`Configuration <api/configuration>` page.

   In order to create an example HLS project, first go to ``example-models/`` from the main directory:

   .. code-block:: bash

      cd example-models/

   And use this command to translate a Keras model:

   .. code-block:: bash

      hls4ml convert -c keras-config.yml

   This will create a new HLS project directory with an implementation of a model from the ``example-models/keras/`` directory.
   To build the HLS project, do:

   .. code-block:: bash

      hls4ml build -p my-hls-test -a

   This will create a Vivado HLS project with your model implementation!

   **NOTE:** For the last step, you can alternatively do the following to build the HLS project:

   .. code-block:: Bash

      cd my-hls-test
      vivado_hls -f build_prj.tcl

   ``vivado_hls`` can be controlled with:

   .. code-block:: bash

      vivado_hls -f build_prj.tcl "csim=1 synth=1 cosim=1 export=1 vsynth=1"

   Setting the additional parameters from ``1`` to ``0`` disables that step, but disabling ``synth`` also disables ``cosim`` and ``export``.

   Further help
   ^^^^^^^^^^^^

   * For further information about how to use ``hls4ml``\ , do: ``hls4ml --help`` or ``hls4ml -h``
   * If you need help for a particular ``command``\ , ``hls4ml command -h`` will show help for the requested ``command``
   * We provide a detailed documentation for each of the command in the :doc:`Command Help <advanced/command>` section

Existing examples
-----------------

* Training codes and examples of resources needed to train the models can be found in the `tutorial <https://github.com/fastmachinelearning/hls4ml-tutorial>`__.
* Examples of model files and weights can be found in `example_models <https://github.com/fastmachinelearning/example-models>`_ directory.

Uninstalling
------------

To uninstall ``hls4ml``:

.. code-block:: bash

   pip uninstall hls4ml

If installed with ``conda``, remove the package with:

.. code-block:: bash

   conda remove hls4ml
