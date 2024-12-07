===================================
High Granularity Quantization (HGQ)
===================================

.. image:: https://github.com/calad0i/HGQ/actions/workflows/sphinx-build.yml/badge.svg
   :target: https://calad0i.github.io/HGQ/
.. image:: https://badge.fury.io/py/hgq.svg
   :target: https://badge.fury.io/py/hgq
.. image:: https://img.shields.io/badge/arXiv-2405.00645-b31b1b.svg
   :target: https://arxiv.org/abs/2405.00645

`High Granularity Quantization (HGQ) <https://github.com/calad0i/HGQ/>`_ is a library that performs gradient-based automatic bitwidth optimization and quantization-aware training algorithm for neural networks to be deployed on FPGAs. By leveraging gradients, it allows for bitwidth optimization at arbitrary granularity, up to per-weight and per-activation level.

.. image:: https://calad0i.github.io/HGQ/_images/overview.svg
   :alt: Overview of HGQ
   :align: center

Conversion of models made with HGQ library is fully supported. The HGQ models are first converted to proxy model format, which can then be parsed by hls4ml bit-accurately. Below is an example of how to create a model with HGQ and convert it to hls4ml model.

.. code-block:: Python

   import keras
   from HGQ.layers import HDense, HDenseBatchNorm, HQuantize
   from HGQ import ResetMinMax, FreeBOPs

   model = keras.models.Sequential([
      HQuantize(beta=1.e-5),
      HDenseBatchNorm(32, beta=1.e-5, activation='relu'),
      HDenseBatchNorm(32, beta=1.e-5, activation='relu'),
      HDense(10, beta=1.e-5),
   ])

    opt = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    callbacks = [ResetMinMax(), FreeBOPs()]

    model.fit(..., callbacks=callbacks)

    from HGQ import trace_minmax, to_proxy_model
    from hls4ml.converters import convert_from_keras_model

    trace_minmax(model, x_train, cover_factor=1.0)
    proxy = to_proxy_model(model, aggressive=True)

    model_hls = convert_from_keras_model(proxy, backend='vivado',output_dir=... ,part=...)


An interactive example of HGQ can be found in the `kaggle notebook <https://www.kaggle.com/code/calad0i/small-jet-tagger-with-hgq-1>`_. Full documentation can be found at `calad0i.github.io/HGQ <https://calad0i.github.io/HGQ/>`_.
