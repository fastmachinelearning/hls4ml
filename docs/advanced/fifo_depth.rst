==============================
FIFO Buffer Depth Optimization
==============================

With the ``io_stream`` IO type, each layer is connected with the subsequent layer through first-in first-out (FIFO) buffers.
The implementation of the FIFO buffers contributes to the overall resource utilization of the design, impacting in particular the BRAM or LUT utilization.
Because the neural networks can have complex architectures generally, it is hard to know a priori the correct depth of each FIFO buffer.
By default ``hls4ml`` chooses the most conservative possible depth for each FIFO buffer, which can result in an unnecessary over-utilization of resources.

In order to reduce the impact on the resources used for FIFO buffer implementation, an optimization flow has been developed that correctly sizes the depth
of the FIFO buffers by analyzing the RTL co-simulation. This feature is currently available in the ``Vitis`` and ``Vivado`` backends.

In the ``Vivado`` backend, FIFO buffer resizing is implemented as a :py:class:`~hls4ml.backends.vivado.passes.fifo_depth_optimization` optimizer pass.
Through RTL simulation with large FIFO buffers (by default set to a depth of 100,000), we estimate the maximum occupation of each FIFO.
Once the maximum depth is determined, the optimizer pass sets the FIFO buffer depth to that value plus 1.

Below we show an example of the use of the FIFO depth optimization. First, we can define a simple neural network in Keras:

.. code-block:: Python

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Dense(64, input_shape=(16,), name='fc1', activation='relu'))
    model.add(Dense(32, name='fc2', activation='relu'))
    model.add(Dense(32, name='fc3', activation='relu'))
    model.add(Dense(5, name='fc4', activation='softmax'))

Then, we can convert the model, including the flow:

.. code-block:: Python

    import hls4ml

    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    config['Flows'] = ['vivado:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vivado:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)


    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           io_type='io_stream',
                                                           hls_config=config,
                                                           output_dir='hls4mlprj_fifo_depth_opt',
                                                           part='xc7z020clg400-1',
                                                           backend='Vivado')

    hls_model.build(reset=False, csim=True, synth=True, cosim=True)

For more details and results, see `H. Borras et al., "Open-source FPGA-ML codesign for the MLPerf Tiny Benchmark" (2022) <https://arxiv.org/abs/2206.11791>`_.

Similarly, the FIFO buffers can be optimized while using the ``Vitis`` backend with the following changes:

.. code-block:: Python

    config['Flows'] = ['vitis:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vitis:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                        io_type='io_stream',
                                                        hls_config=config,
                                                        output_dir='hls4mlprj_fifo_depth_opt',
                                                        part='xc7z020clg400-1',
                                                        backend='Vitis')

Trace-based optimization with FIFO-Advisor (``VitisUnified`` backend)
=====================================================================

The ``VitisUnified`` backend also provides ``vitisunified:fifo_depth_optimization_advisor``, which sizes the FIFOs between layers without RTL co-simulation.
It runs C synthesis once at the default depths, models the design with `LightningSim <https://github.com/sharc-lab/LightningSim>`_,
searches depth assignments with the solvers of `FIFO-Advisor <https://github.com/sharc-lab/fifo-advisor>`_,
and applies a point of their combined Pareto front of latency and BRAM. Enable only one of the two flows.

.. code-block:: Python

    config['Flows'] = ['vitisunified:fifo_depth_optimization_advisor']
    hls4ml.model.optimizer.get_optimizer('vitisunified:fifo_depth_optimization_advisor').configure(
        selection='min_bram',
        search_scope='compute',
    )

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                        io_type='io_stream',
                                                        hls_config=config,
                                                        output_dir='hls4mlprj_fifo_depth_advisor',
                                                        part='xczu9eg-ffvb1156-2-e',
                                                        backend='VitisUnified',
                                                        axi_mode='axi_master')

The pass accepts the following options:

* ``selection``: ``'min_bram'`` (default), ``'min_latency'``, or a function that takes the Pareto front and returns one point.
* ``search_scope``: ``'compute'`` (default) searches only the FIFOs between layers.
  ``'all'`` also searches the AXI wrapper FIFOs, which are not written back.
* ``solvers``: any of ``'heuristic'``, ``'sa'``, ``'group-sa'``, ``'random'`` and ``'group-random'`` (default: all).
* ``write_report``: also write ``fifo_depths_advisor.json`` with the Pareto front and the result of each solver (default ``True``).
* ``skip_synthesis``: analyse an existing solution synthesised at the default depths (default ``False``).

FIFO-Advisor installation
-------------------------

The flow needs the ``lightningsim`` and ``fifo_advisor`` packages, which are not hls4ml dependencies.
LightningSim is distributed through its own conda channel for linux-64, and FIFO-Advisor requires Python 3.11 or newer,
so the flow runs on linux-64 with Python 3.11 or 3.12. An environment can be created with:

.. code-block:: yaml

    name: hls4ml-fifo-advisor
    channels:
      - conda-forge
      - https://sharc-lab.github.io/LightningSim/repo
    dependencies:
      - python=3.11
      - lightningsim=0.2.6
      - numpy
      - scipy<1.16
      - pymoo
      - pandas
      - matplotlib
      - seaborn
      - h5py
      - pyyaml
      - pip

FIFO-Advisor is installed from its repository, and hls4ml from a source checkout with the ``VitisUnified`` backend:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate hls4ml-fifo-advisor
    pip install --no-deps git+https://github.com/sharc-lab/fifo-advisor.git@bc49e72
    pip install .                   # in the hls4ml source directory
    pip install tensorflow==2.14.1  # Keras 2, Python 3.11 only

With Python 3.12, install ``"keras>=3.10" "tensorflow<2.20"`` instead of ``tensorflow==2.14.1``.
TensorFlow 2.20 and newer cannot be used: LightningSim crashes while reading the design.
Vitis must be on ``PATH``. The flow is tested on Ubuntu 22.04 with Vitis 2023.2.

FIFO-Advisor limitations
------------------------

* Only the ``axi_master`` interface is supported.
* The depths come from LightningSim's model of the schedule; confirm them with a co-simulation of the rebuilt design.
