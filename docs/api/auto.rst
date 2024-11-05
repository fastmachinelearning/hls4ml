=============================
Automatic precision inference
=============================

The automatic precision inference (implemented in :py:class:`~hls4ml.model.optimizer.passes.infer_precision.InferPrecisionTypes`) attempts to infer the appropriate widths for a given precision.
It is initiated by configuring a precision in the configuration as 'auto'. Functions like :py:class:`~hls4ml.utils.config.config_from_keras_model` and :py:class:`~hls4ml.utils.config.config_from_onnx_model`
automatically set most precisions to 'auto' if the ``'name'`` granularity is used.

.. note::
    It is recommended to pass the backend to the ``config_from_*`` functions so that they can properly extract all the configurable precisions.

The approach taken by the precision inference is to set accumulator and other precisions to never truncate, using only the bitwidths of the inputs (not the values). This is quite conservative,
especially in cases where post-training quantization is used, or if the bit widths were set fairly loosely. The recommended action in that case is to edit the configuration and explicitly set
some widths in it, potentially in an iterative process after seeing what precisions are automatically set. Another option, currently implemented in :py:class:`~hls4ml.utils.config.config_from_keras_model`,
is to pass a maximum bitwidth using the ``max_precison`` option. Then the automatic precision inference will never set a bitwdith larger than the bitwidth or an integer part larger than the integer part of
the ``max_precision`` that is passed. (The bitwidth and integer parts are treated separately.)
