================
Keras and QKeras
================

Keras and the quantization library QKeras are well supported in ``hls4ml``. Currently, the Keras v2 (``tf.keras``) is the preferred version, and the future versions of ``hls4ml`` will expand support for Keras v3. The frontend is based on the parsing the serialized json representation of the model.

Currently, ``hls4ml`` can parse most Keras layers, including core layers, convolutional layers, pooling layers, recurrent layers, merging/reshaping layers and activation layers, implemented either via sequential or functional API. Notably missing are the attention and normalization layers. The equivalent QKeras API and quantizers are also supported. The ``Lambda`` layers don't save their state in the serialized format and are thus impossible to parse. In this case, the ``Lambda`` layers can be implemented as custom layers and parsed via the :ref:`Extension API`.

The ``data_format='channels_first'`` parameter of Keras layers is supported, but not extensively tested. All HLS implementations in ``hls4ml`` are based on ``channels_last`` data format and need to be converted to that format before the HLS code can be emitted. We encourage users of ``channels_first`` to report their experiences to developers on GitHub.

The development team of ``hls4ml`` is currently exploring options for QKeras alternative and will provide a drop-in replacement API compatible with Keras v3.
