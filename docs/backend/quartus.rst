=======
Quartus
=======

.. warning::
    The **Quartus** backend is deprecated and will be removed in a future version. Users should migrate to the **oneAPI** backend.

The **Quartus** backend of hls4ml is designed for deploying NNs on Intel/Altera FPGAs. It uses the discontinued Intel HLS compiler. The **oneAPI** backend should be preferred for new projects.
The **oneAPI** backend contains the migrated the HLS code from this backend, with significantly better io_stream support, though the **oneAPI** backend does not yet support profiling, tracing,
or the BramFactor option supported by the **Quartus** backend.  Nevertheless, little or no further development is expected for this backend.

The **Quartus** backend only implements the ``Resource`` strategy for the layers. There is no ``Latency`` implementation of any of the layers.
