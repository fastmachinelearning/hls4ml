====================
PyTorch and Brevitas
====================

The PyTorch frontend in ``hls4ml`` is implemented by parsing the symbolic trace of the ``torch.fx`` framework. This ensures the proper execution graph is captured. Therefore, only models that can be traced with the FX framework can be parsed by ``hls4ml``.

Provided the underlying operation is supported in ``hls4ml``, we generally aim to support the use of both ``torch.nn`` classes and ``torch.nn.functional`` functions in the construction of PyTorch models. Generally, the use of classes is more thoroughly
tested. Please reach out if you experience any issues with either case.

The PyTorch/Brevitas parser is under heavy development and doesn't yet have the same feature set of the Keras parsers. Feel free to reach out to developers if you find a missing feature that is present in Keras parser and would like it implemented.

.. note::
    The direct ingestion of models quantized with brevitas is not supported currently. Instead, brevitas models shoud be exported in the ONNX format (see `here <https://xilinx.github.io/brevitas/tutorials/onnx_export.html>`_) and read with the ``hls4ml``
    QONNX frontend. Issues may arise, for example when non power-of-2 or non-scalar quantization scales are used. Please reach out if you encounter any problems with this workflow.

For multi-dimensional tensors, ``hls4ml`` follows the channels-last convention adopted by Keras, whereas PyTorch uses channels-first. By default, ``hls4ml`` will automaticlly transpose any tensors associated with weights and biases of the internal layers
of the model. If the ``io_parallel`` I/O type (see :ref:`Concepts`) is used, a transpose node will be added to the model that also adjusts the input tensors. This is not available in the ``io_stream`` case and inputs must be transposed by the user.
Outputs are not transposed back by default, but in ``io_parallel`` case, a transpose node can be added. If not needed, these adjustments can also be switched off. See :py:class:`~hls4ml.utils.config.config_from_pytorch_model` for details.

The equivalent of Keras extension API is not yet available for PyTorch parser, and will be provided in the future.
