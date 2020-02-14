**v0.1.6:** We just release a new version of `hls4ml`, which includes the following new features:

- `tf_to_hls` tool for converting tensorflow models (protobufs)

-  Experimental support for larger `Conv1D/2D` layers

- [QKeras](https://github.com/google/qkeras) support (limited to binary/ternary for now)

- API enhancements (custom layers, multiple backends)

- [Profiling](PROFILING) support

- `hls4ml report`command, `hls4ml build -l` for Logic Synthesis

- Support for all-in-one Keras's `h5` files (obtained with Keras's `save()` function, without the need for separate json).

- Fused Batch Normalisation into Dense layer optimsation.

---
**v0.1.5**: Per-layer precision and reuse factor

---
**v0.1.3**: Adding PyTorch support

--- 
**v0.1.2**: First beta release
   * some bug fixes for pipelining and support for layer types
---
**v0.0.2**: first alpha release
   * full translation of DNNs from Keras 
   * an example Conv1D exists
   * parallel mode is supported (serial mode, not yet)
---