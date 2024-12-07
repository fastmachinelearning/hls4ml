=============================
Automatic precision inference
=============================

The automatic precision inference (implemented in :py:class:`~hls4ml.model.optimizer.passes.infer_precision.InferPrecisionTypes`) attempts to infer the appropriate
widths for a given precision. It is initiated by setting a precision in the configuration as ``'auto'``. (Note, only layer-level precisions can be set to ``'auto'``,
not model-level.)  Functions like :py:class:`~hls4ml.utils.config.config_from_keras_model`, :py:class:`~hls4ml.utils.config.config_from_onnx_model`,
and :py:class:`~hls4ml.utils.config.config_from_pytorch_model` automatically set most precisions to ``'auto'`` if the ``'name'`` granularity is used.

.. note::
    It is recommended to pass the backend to the ``config_from_*`` functions so that they can properly extract all the configurable precisions.

The approach taken by the precision inference is to set accumulator (the internal variable used to accumulate values in the matrix multiplications) and other precisions
to never truncate, using only the bitwidths of the inputs (not the values). This is quite conservative, especially in cases where post-training quantization is used, or
if the bit widths were set fairly loosely. The recommended action in that case is to edit the configuration and explicitly set some widths in it, potentially in an iterative process
after profiling the data. Another option is to pass a maximum precision using the ``max_precison`` parameter of the ``config_form_*`` functions. Then the automatic precision
inference will never set a bitwdith larger than the bitwidth of the ``max_precision`` or an integer part larger than the integer part of the ``max_precision`` that is passed.
(The bitwidth and integer parts of the ``max_precision`` are treated separately.)

When manually setting bitdwidths, the accumulator can overflow, and the precision may need to be reduced. For the accumulator, it is usually a bad idea to explicitly
enable rounding or saturation modes since it dramatically increases the execution time. For other types (e.g. output types or weight types), however, rounding and saturation handling
can be enabled as needed.
