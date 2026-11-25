====================
PyTorch and Brevitas
====================

The PyTorch frontend in ``hls4ml`` is implemented by parsing the symbolic trace of the ``torch.fx`` framework. This ensures the proper execution graph is captured. Therefore, only models that can be traced with the FX framework can be parsed by ``hls4ml``.

Provided the underlying operation is supported in ``hls4ml``, we generally aim to support the use of both ``torch.nn`` classes and ``torch.nn.functional`` functions in the construction of PyTorch models. Generally, the use of classes is more thoroughly
tested. Please reach out if you experience any issues with either case.

The PyTorch/Brevitas parser is under heavy development and doesn't yet have the same feature set as the Keras parsers. Feel free to reach out to developers if you find a missing feature that is present in the Keras parser and would like it implemented.

.. note::
    The direct ingestion of models quantized with Brevitas is not supported currently. Instead, Brevitas models should be exported in the ONNX format (see `here <https://xilinx.github.io/brevitas/tutorials/onnx_export.html>`_) and read with the ``hls4ml``
    QONNX frontend. Issues may arise, for example, when non-power-of-2 or non-scalar quantization scales are used. Please reach out if you encounter any problems with this workflow.

Tensor layout
-------------

PyTorch stores convolutional data channels-first, ``(C, H, W)``, while ``hls4ml`` implements convolutions, pooling and
similar layers on channels-last data, ``(H, W, C)``. Layers such as ``Linear``, ``LayerNorm``, ``einsum`` and
batch-first RNNs work on the last axis, so PyTorch's own order already suits them. ``hls4ml`` decides the stored order of
every tensor from the layers that produce and use it, and inserts a transpose only where two layers disagree. Models
that mix both kinds of layer, joined by ``permute`` or ``transpose``, are supported; a ``permute`` usually costs nothing,
because the data is already stored in the order the next layer needs.

The only layout choice left to the user is at the model boundary, set with the ``input_layout`` and ``output_layout``
arguments of :py:func:`~hls4ml.utils.config.config_from_pytorch_model` (``InputLayout`` and ``OutputLayout`` in the
``Model`` section of the configuration):

* ``'source'`` (default): ``predict()`` takes inputs and returns outputs in PyTorch's order. A transpose is added at an
  input whose first layer needs channels-last data, and at an output that is not stored in PyTorch's order.
* ``'native'``: no transpose at the boundary. The data must be supplied, and is returned, in the order the adjacent
  layer uses. ``hls_model.get_io_layouts()`` reports that order for each input and output, and
  ``hls4ml.utils.transpose_utils.to_native_layout`` and ``to_source_layout`` convert NumPy arrays.

Either argument can be a dict keyed by input or output name to set the layout per tensor. Transposes in ``io_stream``
buffer a whole tensor, and ``hls4ml`` warns when it inserts one; ``'native'`` layouts avoid those at the boundary.

The ``channels_last_conversion`` and ``transpose_outputs`` arguments are deprecated. ``'full'`` corresponds to
``input_layout='source'``, ``'internal'`` to ``input_layout='native'``, and ``'off'`` has no effect;
``transpose_outputs=True`` corresponds to ``output_layout='source'`` and ``False`` to ``output_layout='native'``.
Setting ``LayoutInference`` to ``False`` in the ``Model`` section temporarily restores the previous conversion.

Custom PyTorch layers can be supported through the :ref:`Extension API`; see that page for details.

.. note::
    Experimental spiking layer support is available for selected modules. See :doc:`../advanced/snn` for details.
