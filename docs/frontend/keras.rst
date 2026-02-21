================================
Keras and its quantized variants
================================

Keras and its quantized variants are supported in ``hls4ml``. Both Keras v2 (``tf.keras``) and the new Keras v3 are supported. While the Keras v2 support is based on parsing the serialized json representation of the model, the Keras v3 support uses direct model inspection.

For Keras v2, QKeras, and HGQ, ``hls4ml`` supports most of its layers, including core layers, convolutional layers, pooling layers, recurrent layers (not implemented in HGQ), merging/reshaping layers, and activation layers. The ``(Q)BatchNormalization`` layer is also supported. Experimental support for ``LayerNormalization`` is added for vanilla Keras v2.

For Keras v3, the support for EinsumDense layer is added in addition, but without recurrent layers in general. For HGQ2, some extra layers are supported in addition, such as ``QEinsum``, ``QMultiHeadAttention``, `QUnaryFunctionLUT` (arbitrary unary function as a 1-d lookup table) and some binary operators.

keras ``Operators`` that are not layers are generally not supported in ``hls4ml``. This includes operators such as ``Add``, ``Subtract``, ``Multiply``, and ``Divide``. Please use the corresponding Keras layers instead.

Arbitrary ``Lambda`` layers are not, and are not planned to be supported in ``hls4ml`` due to the difficultness to parse generic lambda expression. For custom operations required, please refer to the :ref:`Extension API` documentation to add custom layers to the conversion process.

The ``data_format='channels_first'`` parameter of Keras layers is supported for a limited subset of layers and it is not extensively tested. All HLS implementations in ``hls4ml`` are based on ``channels_last`` data format convention and need to be converted to that format before the HLS code can be emitted. We encourage users of ``channels_first`` to report their experiences to developers on GitHub.


* `QKeras <https://github.com/fastmachinelearning/qkeras>`_
    The equivalent QKeras API and its quantizers are also supported by ``hls4ml``. QKeras is not compatible with Keras v3.
* `HGQ <https://github.com/calad0i/HGQ>`_
    The equivalent HGQ API is also supported. Still maintained but deprecated in favor of `HGQ2 <../hgq2.html>`_.
* `HGQ2 <https://github.com/calad0i/HGQ2>`_
    The equivalent HGQ2 API is also supported, plus some additional advanced operators.
