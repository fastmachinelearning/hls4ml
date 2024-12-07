====================
PyTorch and Brevitas
====================

PyTorch frontend in ``hls4ml`` is implemented by parsing the symbolic trace of the ``torch.fx`` framework. This ensures proper execution graph is captured. Therefore, only models that can be traced with the FX framework can be parsed by ``hls4ml``.

PyTorch/Brevitas parser is under heavy development and doesn't yet have the same feature set of the Keras parsers. Feel free to reach out to developers if you find a missing feature that is present in Keras parser and would like it implemented.

The equivalent of Keras extension API is not yet available for PyTorch parser, and will be provided in the future.
