==============
ONNX and QONNX
==============

Parsing of ONNX and QONNX models is made in conjunction with the `qonnx <https://github.com/fastmachinelearning/qonnx>`_ package, even if it no quantization is used. This is a common initial parser shared with the AMD/Xilinx FINN project. The first step is to do constant folding, shape inference, etc., on the ONNX graph, commonly known as `cleaning`.  If a model has convolution layers, the model also needs to be converted to a channels-last format, since that is what hls4ml mainly supports. The ``qonnx`` package also provides a number of additional transforms that may need to be used. For example, ``Gemm`` nodes need to converted to ``MatMul`` and ``Add`` nodes.

There are command-line based versions of cleaning and channels-last conversion:

.. code-block:: bash

    $ qonnx_clean filename.onnx
    $ qonnx_to_channels_last filename_clean.onnx
    $ qonnx_clean filename_clean_channels_last.onnx  # good to do a clean again as a last step

Things can similarly be done in python. This method is usually easier if you additionally need to call other transforms. An example is given below which also calls the ``GemmToMatMul`` converter:

.. code-block:: python

    model = ModelWrapper('filename.onnx')
    model = qonnx.util.cleanup.cleanup_model(model)
    model = model.transform(ConvertToChannelsLastAndClean())
    model = model.transform(GemmToMatMul())
    model = qonnx.util.cleanup.cleanup_model(model)

``ModelWrapper`` is defined in ``qonnx.core.modelwrapper``. More information on the ``qonnx`` package can be found at the `QONNX documentation page <https://qonnx.readthedocs.io/en/latest/index.html>`_.


The next steps are very similar to if you are using a Keras model:

.. code-block:: python

    config = hls4ml.utils.config.config_from_onnx_model(
        model, granularity='name', backend='Vitis', default_precision='fixed<16,6>'
    )
    # modify the config as desired
    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir='my-hls-test',
        io_type='io_stream',
        backend='Vitis',
        hls_config=config,
    )
    hls_model.compile()

Note, unlike the Keras version, "name" granularity is the default for ``config_from_onnx_model``, and it must be used for QONNX models. Unquantized ONNX models can use "model" if so desired, but generally there is no benefit.

One can subsequently call the ``predict`` function to check the performance or build the project.

Note that ``execute_onnx`` in ``qonnx.core.onnx_exec`` can be use to run the QONNX graphs directly, and it also provides the values at intermediate layers for validating the model (tracing).

Quant nodes
===========

Documentation for quant nodes is provided in the `qonnx package <https://github.com/fastmachinelearning/qonnx/tree/main/docs/qonnx-custom-ops>`_. Note that currently hls4ml only supports the `Quant operator <https://github.com/fastmachinelearning/qonnx/tree/main/docs/qonnx-custom-ops/quant_op.md>`_. Also, not all legal ``Quant`` configurations are parsable by hls4ml or synthesizable. The ``scale``, ``zeropt``, and ``bitwidth`` values must be constant (though not necessarily scalar for the ``scale`` and ``zeropt``).

Generally if the ``zeropt`` is 0 and the ``scale`` is a scalar power of 2, hls4ml uses ``ap_fixed`` or ``ac_fixed`` types (depending on the backend) to represent the quantizations. In other cases, the ``scale`` and ``zeropt`` need to be explicitly handled by hls4ml, and there is more of a chance of hls4ml not being able to process the input. (Please report any issues that you find.)
