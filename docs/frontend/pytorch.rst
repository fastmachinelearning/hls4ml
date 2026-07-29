====================
PyTorch and Brevitas
====================

The PyTorch frontend in ``hls4ml`` is implemented by parsing the symbolic trace of the ``torch.fx`` framework. This ensures the proper execution graph is captured. Therefore, only models that can be traced with the FX framework can be parsed by ``hls4ml``.

Provided the underlying operation is supported in ``hls4ml``, we generally aim to support the use of both ``torch.nn`` classes and ``torch.nn.functional`` functions in the construction of PyTorch models. Generally, the use of classes is more thoroughly
tested. Please reach out if you experience any issues with either case.

The PyTorch/Brevitas parser is under heavy development and doesn't yet have the same feature set of the Keras parsers. Feel free to reach out to developers if you find a missing feature that is present in Keras parser and would like it implemented.

Brevitas
========

Models quantized with `Brevitas <https://xilinx.github.io/brevitas/>`_ can be ingested directly, without an intermediate ONNX export. The quantized weights are taken from the Brevitas ``QuantTensor`` and the corresponding
``ap_fixed`` precision is derived from the quantizer's bit width and scale. Quantization attached to the input or output of a layer is turned into a ``Quant`` node, which the QONNX optimizer passes then fold into the surrounding layers.

The following Brevitas modules are supported:

* ``QuantLinear``
* ``QuantConv1d``, ``QuantConv2d``
* ``QuantReLU``, ``QuantSigmoid``, ``QuantTanh``
* ``QuantIdentity``
* ``QuantEltwiseAdd``
* ``QuantUpsample``, ``QuantUpsamplingNearest2d``, ``QuantUpsamplingBilinear2d``
* ``QuantRNN``
* ``QuantDropout`` (skipped, as in the unquantized PyTorch parser)

Unquantized ``torch.nn`` layers can be mixed freely into the model; pooling in particular is handled by the ordinary ``nn.MaxPool1d``/``nn.MaxPool2d``/``nn.AvgPool1d``/``nn.AvgPool2d`` parsers.

Three limitations apply:

* Only power-of-2 quantization scales are supported. A layer with a non power-of-2 scale raises an exception at parse time; export to QONNX (see `here <https://xilinx.github.io/brevitas/tutorials/onnx_export.html>`_) and use the ``hls4ml`` QONNX frontend for those models.
* The ``QuantUpsample*`` layers are only available with the ``io_parallel`` I/O type.
* Brevitas' quantized pooling layers (``TruncAvgPool2d``, ``TruncAdaptiveAvgPool2d``) and ``QuantLSTM`` are not supported yet.

.. note::
    Call ``model.eval()`` before converting. Brevitas activation quantizers collect statistics in training mode, so the scale a model reports can change with every forward pass while it is still in training mode, and ``hls4ml`` would capture whichever value happened to be current.

For multi-dimensional tensors, ``hls4ml`` follows the channels-last convention adopted by Keras, whereas PyTorch uses channels-first. By default, ``hls4ml`` will automaticlly transpose any tensors associated with weights and biases of the internal layers
of the model. If the ``io_parallel`` I/O type (see :ref:`Concepts`) is used, a transpose node will be added to the model that also adjusts the input tensors. This is not available in the ``io_stream`` case and inputs must be transposed by the user.
Outputs are not transposed back by default, but in ``io_parallel`` case, a transpose node can be added. If not needed, these adjustments can also be switched off. See :py:class:`~hls4ml.utils.config.config_from_pytorch_model` for details.

The equivalent of Keras extension API is not yet available for PyTorch parser, and will be provided in the future.

.. note::
    Experimental spiking layer support is available for selected modules. See :doc:`../advanced/snn` for details.
