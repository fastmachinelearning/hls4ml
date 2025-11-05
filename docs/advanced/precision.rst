==============================
Model-wise Precision Inference
==============================

The model-wise precision inference (implemented in :py:class:`~hls4ml.model.optimizer.passes.bit_exact.BitExact`) attempts to infer the appropriate configuration for **all** precision in the model. Unlike the automatic precision inference, this pass disregards all user-defined precision, and "trust" only data embedded in the model, i.e., the actual values of the weights and explicit quantizers defined between layers.

This pass uses modified symbolic interval arithmetic to compute the ranges and the needed quantization steps for all precision in the model graph, with the goal of eliminating any discrepancy between the quantized model and the original model. In the inference process, only the raw weight values and the explicit quantizers (either ``FixedPointQuantizer``, or ``linear/relu`` layers with ``trusted=True``) are considered as sources of precision information. All other precision information (e.g., user-defined precision in ``config_from_*`` functions) will not be used in the inference process.

Invoking of this pass is configured by the ``bit_exact`` key in the backend configuration (default: ``None``). There are two ways to enable this pass:
- When converting from ``HGQ/HGQ2`` models, this pass is automatically enabled unless ``bit_exact`` is explicitly set to ``False``.
- For other models, this pass can be enabled by setting ``bit_exact`` to ``True``. Currently, only ``QKeras`` sets this key automatically when converting from ``QKeras`` models. Support for ``QONNX`` is planned but not yet implemented.

If the original model is not properly quantized, this pass will lead to huge bitwidths in the model. In this context, properly quantized models are those that have quantizers defined between **all layers with non-trivial arithmetics** (i.e., essentially all layers other than reshape/flatten/transpose/linear-like layers only rearranging elements). The successful application of this pass should result in bit-exact model, i.e., the quantized model should produce the same outputs as the original model for all inputs [*]_.

Not all operator types are supported in this pass. If any unsupported operator is encountered during the inference, this pass will **crash** the conversion process to prevent silent failures. Please consider use `automatic precision inference <../auto.html>`_ if your model contains unsupported operators or unquantized components.

.. warning::
    Importantly, quantizers **should be used immediately after the inputs**, or the input precision may not be properly inferred. If you are using ``HGQ/HGQ2``, this is automatically taken care of in most cases. If you are using ``QKeras``, make sure to put a ``QActivation`` with ``quantized_bits`` right after the input layer such that the input precision can be derived.

.. [*] While quantized, the original model will still operate on float-point values, so there is a chance that the outputs will not be exactly the same due to float rounding errors in the original model.

.. note::
    When this functionality is used, one **should not** use the ``config_from_*`` functions to set the precision in the model. Automatic precision inference and this pass cannot be used simultaneously.
