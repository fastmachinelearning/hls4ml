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
``ap_fixed`` precision is derived from the quantizer's bit width and scale. Quantization attached to the input or output of a layer becomes a ``FixedPointQuantizer`` node where it maps onto a plain fixed-point type, and a QONNX
``Quant`` node otherwise (a non-zero zero-point, or a scale that is not a power of two).

Bit-exactness
-------------

``FixedPointQuantizer`` nodes enable the model-wide :py:class:`~hls4ml.model.optimizer.passes.bit_exact.BitExact` flow, which derives every precision in the model by quantized interval arithmetic. Where a model quantizes its
own inputs and outputs this makes the generated firmware reproduce Brevitas *exactly*, with no precision tuning by hand:

* The input quantizer is folded into the model's input port, so an incoming value is rounded once onto the quantizer's grid rather than being truncated to the input type and then rounded again.
* Accumulators are widened to whatever the dot product actually needs, so they cannot overflow.

Two things to be aware of:

* A layer that has no input quantizer (only ``weight_quant``, say) is compared against a Brevitas model that is still computing in float32 for that tensor, so exact agreement is not possible in principle. Add a ``QuantIdentity``
  or an ``input_quant`` if you need it.
* The flow has no handler for some layers, ``SimpleRNN`` (from ``QuantRNN``) and ``Resize`` (from ``QuantUpsample*``) among them. Models containing those keep the ``Quant`` representation and the ordinary precision inference, so
  they behave as before. Setting ``BitExact`` to ``True`` or ``False`` in the model config overrides this choice.

The following Brevitas modules are supported:

* ``QuantLinear``
* ``QuantConv1d``, ``QuantConv2d``
* ``QuantReLU``, ``QuantSigmoid``, ``QuantTanh``
* ``QuantIdentity``
* ``QuantEltwiseAdd``
* ``QuantUpsample``, ``QuantUpsamplingNearest2d``, ``QuantUpsamplingBilinear2d``
* ``QuantRNN``, ``QuantLSTM``
* ``QuantDropout`` (skipped, as in the unquantized PyTorch parser)

Brevitas provides no ``QuantGRU``, so hls4ml's GRU layer is reachable only through the unquantized ``torch.nn.GRU``.

.. warning::
    Brevitas' recurrent layers take their initial states as separate arguments — ``QuantRNN.forward(inp, hx=None)`` and
    ``QuantLSTM.forward(inp, hx=None, cx=None)`` — not as the tuple ``torch.nn.LSTM`` expects. Writing
    ``self.rnn(x, (h0, c0))`` binds the whole tuple to ``hx`` and leaves ``cx`` unset, which silently produces a
    different result rather than raising.

Unquantized ``torch.nn`` layers can be mixed freely into the model; pooling in particular is handled by the ordinary ``nn.MaxPool1d``/``nn.MaxPool2d``/``nn.AvgPool1d``/``nn.AvgPool2d`` parsers.

Three limitations apply:

* Only power-of-2 quantization scales are supported. A layer with a non power-of-2 scale raises an exception at parse time; export to QONNX (see `here <https://xilinx.github.io/brevitas/tutorials/onnx_export.html>`_) and use the ``hls4ml`` QONNX frontend for those models.
* The ``QuantUpsample*`` layers are only available with the ``io_parallel`` I/O type.
* Brevitas' quantized pooling layers (``TruncAvgPool2d``, ``TruncAdaptiveAvgPool2d``) are not supported yet, nor are recurrent layers with ``num_layers > 1`` or ``bidirectional=True``.
* Recurrent layers are not bit-exact. hls4ml evaluates the cell in its accumulator precision with lookup-table activations and does not model brevitas' per-gate accumulator, sigmoid, tanh and cell-state quantizers.

.. note::
    Call ``model.eval()`` before converting. Brevitas activation quantizers collect statistics in training mode, so the scale a model reports can change with every forward pass while it is still in training mode, and ``hls4ml`` would capture whichever value happened to be current.

For multi-dimensional tensors, ``hls4ml`` follows the channels-last convention adopted by Keras, whereas PyTorch uses channels-first. By default, ``hls4ml`` will automaticlly transpose any tensors associated with weights and biases of the internal layers
of the model. If the ``io_parallel`` I/O type (see :ref:`Concepts`) is used, a transpose node will be added to the model that also adjusts the input tensors. This is not available in the ``io_stream`` case and inputs must be transposed by the user.
Outputs are not transposed back by default, but in ``io_parallel`` case, a transpose node can be added. If not needed, these adjustments can also be switched off. See :py:class:`~hls4ml.utils.config.config_from_pytorch_model` for details.

The equivalent of Keras extension API is not yet available for PyTorch parser, and will be provided in the future.

.. note::
    Experimental spiking layer support is available for selected modules. See :doc:`../advanced/snn` for details.
