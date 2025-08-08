==============================
Model-wise Precision Inference
==============================

The model-wise precision inference (implemented in :py:class:`~hls4ml.model.optimizer.passes.bit_exact.BitExact`) attempts to infer the appropriate for **all** precisions in the model. Unlike the automatic precision inference, this pass disregards all user-defined precisions, and "trust" only data embedded in the model, i.e., the actual values of the weights and explicit quantizers defined between layers.

Currently, this pass will only be triggered by the presence of any ``FixedPointQuantizer`` (explicit quantizer operator) layer in the model. This pass uses an modified symbolic interval arithmetic to compute the ranges and needed quantization steps for all precisions in the model graph, with the goal of eliminating the discrepency between the quantized model and the original model. Currently, only HGQ/HGQ2 models will produce such quantizers, and the pass will not be triggered for models from other frontends.

If the original model is not properly quantized, this pass will lead to huge bitwidths in the model. In this context, properly quantized models are those that have quantizers defined between **all layers with non-trivial arithmetics**. Importantly, quantizers **should be used immediately after the inputs**, or the input precision may not be properly inferred. The successful application of this pass should result in bit-exact model, i.e., the quantized model should produce the same outputs as the original model for all inputs [*]_.

.. [*] While quantized, the original model will still operate on float-point values, so there is a chance that the outputs will not be exactly the same due to float rounding errors in the original model.

.. note::
    Unlike the automatic precision inference, it is strongly recommended to **not** use the ``config_from_*`` functions to set the precisions in the model.
