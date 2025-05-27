================================
Keras and its quantized variants
================================

Keras and the quantization library QKeras are well supported in ``hls4ml``. Both Keras v2 (``tf.keras``) and the new Keras v3 are supported. While the Keras v2 support is based on parsing the serialized json representation of the model, the Keras v3 support uses direct model inspection.

Currently, ``hls4ml`` can parse most Keras layers, including core layers, convolutional layers, pooling layers, recurrent layers, merging/reshaping layers and activation layers, implemented either via sequential or functional API. Notably missing are the attention and normalization layers. The ``Lambda`` layers don't save their state in the serialized format and are thus impossible to parse. In this case, the ``Lambda`` layers can be implemented as custom layers and parsed via the :ref:`Extension API`.

The ``data_format='channels_first'`` parameter of Keras layers is supported, but not extensively tested. All HLS implementations in ``hls4ml`` are based on ``channels_last`` data format and need to be converted to that format before the HLS code can be emitted. We encourage users of ``channels_first`` to report their experiences to developers on GitHub.


* `QKeras <https://github.com/fastmachinelearning/qkeras>`_
    The equivalent QKeras API and its quantizers are also supported by ``hls4ml``. QKeras is not compatible with Keras v3. Currently, only HGQ2 is compatible with Keras v3 (see below).
* `HGQ <https://github.com/calad0i/HGQ>`_
    The equivalent HGQ API is also supported. HGQ is not compatible with Keras v3. See `advanced/HGQ <../advanced/hgq.html>`__ for more information.
* `HGQ2 <https://github.com/calad0i/HGQ2>`_
    HGQ2 is based on Keras v3. Its support in hls4ml is currently under development.

The development team of ``hls4ml`` is currently exploring options for QKeras alternative and will provide a drop-in replacement API compatible with Keras v3.
