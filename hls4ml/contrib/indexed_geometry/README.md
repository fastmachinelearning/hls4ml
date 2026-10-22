# indexed_geometry

hls4ml Extension API layers for Keras models over non-standard
(irregularly-indexed) pixel geometries, such as the hexagonal cameras
used in imaging atmospheric Cherenkov telescopes (IACTs).

## Overview

This package provides three custom Keras layers and their hls4ml
Extension API registration (`NeighborGatherLayer`,
`IndexedConvolutionLayer`, `IndexedPoolingLayer`) that together enable
convolutional neural network inference over irregularly-pixelated
detector images. They were originally developed for hexagonal camera
geometries, where each pixel has up to seven neighbors described by a
fixed index map (border pixels have fewer, with missing slots
represented as `-1` and zero-masked in hardware), but the layers work
with any neighbor-indexing scheme, not just hexagonal grids, so they
can be reused for other detectors or irregular sensor layouts.

The examples in `examples/` use synthetic data and a model
architecture modeled after real classifiers built with
[CTLearn](https://github.com/ctlearn-project/ctlearn), a library for
IACT event reconstruction and classification with deep learning.

The three layers implement the neighbor-indexed convolution pipeline:

- **`NeighborGatherLayer`**: for each pixel, gathers the features of
  its neighbors into a new dimension, producing a tensor of shape
  `[batch, n_pixels, n_neighbors, n_features]`. Border slots
  (`index == -1`) are zero-masked.
- **`IndexedConvolutionLayer`**: applies a `Conv2D` with
  `kernel_size=(1, n_neighbors)` independently to each pixel over the
  gathered neighbor dimension, followed by a ReLU activation.
- **`IndexedPoolingLayer`**: applies max or average pooling over the
  neighbor dimension, reducing `[batch, n_pixels, n_neighbors, n_features]`
  to `[batch, n_pixels, n_features]`.

Both the `io_parallel` and `io_stream` hls4ml backends are supported
(see "Known limitations" below for `io_parallel`'s scaling limits).
Only `use_3d_conv=False` (2D) models are supported; `use_3d_conv=True`
raises `NotImplementedError`.

## Files

- `keras_layers.py`: the three Keras layer definitions.
- `neighbor_gather.py`, `indexed_conv.py`, `indexed_pool.py`: hls4ml
  Extension API registration (IR layer, Keras parser, HLS config and
  function-call templates) for each layer. Importing this package
  (`import hls4ml.contrib.indexed_geometry`) registers all three.
- `nnet_utils/`: the three HLS kernel headers (`io_parallel` and
  `io_stream` overloads).
- `examples/`: two worked, runnable examples (see below).

## Usage

```python
import hls4ml
from hls4ml.contrib.indexed_geometry import (
    NeighborGatherLayer, IndexedConvolutionLayer, IndexedPoolingLayer,
)

# ... build a Keras model using these layers ...

hls_config = hls4ml.utils.config_from_keras_model(
    keras_model, granularity='name', backend='Vitis',
)
hmodel = hls4ml.converters.convert_from_keras_model(
    keras_model, output_dir='my_hls_project',
    backend='Vitis', io_type='io_stream', hls_config=hls_config,
)
```

## Choosing precision

hls4ml has its own built-in automatic precision inference: leave
`default_precision` unset (or pass `'auto'`) in
`config_from_keras_model()`, and its `InferPrecisionTypes` optimizer
pass infers per-layer fixed-point types from the model itself. This is
the recommended starting point for new models built with these layers.

The two examples in `examples/` instead use `examples/precision_utils.py`,
a small profiling-based helper (`get_hls_config()`) developed alongside
this package: it runs a Keras forward pass to measure per-layer
activation ranges, then picks a safety-margined `ap_fixed` width from
the observed maximum. It is kept as an example-only helper, not part of
this package, since it isn't specific to these layers. If `'auto'`
works well for your model, prefer it over both.

## `pack_neighbors`

Setting `PackNeighbors=True` (in `io_stream` only) on a
`NeighborGatherLayer` and its direct consumer flattens the neighbor and
feature dimensions into a single wide stream packet per pixel, instead
of one packet per neighbor. This reduces gather latency from roughly
`n_pixels * n_neighbors` to roughly `n_pixels` cycles. A
`NeighborGatherLayer` and its consumer (`IndexedConvolutionLayer` or
`IndexedPoolingLayer`) must have matching `PackNeighbors` settings, or
conversion fails with a clear shape-mismatch `AssertionError`.
`ReuseFactor` and `Strategy` have no effect on `IndexedConvolutionLayer`
in `io_stream` with `PackNeighbors=True`.

## Examples

- **`examples/ctlearn/`**: the full, two-block CTLearn model (163
  pixels), `io_stream` only, with optional C synthesis.
- **`examples/xcku040/`**: a smaller, single-conv model, used to
  validate real C synthesis on a Xilinx XCKU040 and to compare against
  a hand-optimized manual HLS implementation (see that example's
  README).

Both examples share `examples/precision_utils.py`. See each example's
own `README.md` for details and expected results.

## Known limitations

- **2D models only.** `use_3d_conv=True` (Conv3D) is not available in
  hls4ml and raises `NotImplementedError`.
- **`io_parallel` does not scale to realistic model sizes.** The
  `io_parallel` kernels use full `#pragma HLS UNROLL` and work
  correctly for each layer in isolation or for small/synthetic models.
  For a full-size model on a real camera geometry (163 pixels), C
  synthesis of the complete pipeline does not complete; `io_stream` is
  required at that scale (see `examples/ctlearn/` and
  `examples/xcku040/`).
- **`ReuseFactor` and `Strategy` have no effect on
  `IndexedConvolutionLayer` in `io_stream` with `PackNeighbors=True`.**
  See "`pack_neighbors`" above.
- **MaxPool and negative activations.** Border pixel slots are
  zero-masked by `NeighborGatherLayer` before pooling. For
  `IndexedPoolingLayer` with `pooling_type='max'`, this is only correct
  when upstream activations are non-negative (e.g. after a ReLU). If
  activations can be negative, use `pooling_type='average'` or ensure
  border pixels are excluded at the model level.
- **Softmax.** hls4ml Softmax support can be unreliable for custom
  layer pipelines. The recommended approach, used in both examples, is
  to exclude the Softmax from the hls4ml model and apply it in Python
  after inference.

## Testing

Tests for these layers live in `test/pytest/test_indexed_geometry.py`,
covering both `io_type`s, `pack_neighbors`, both pooling types, and a
full integration test with `BatchNormalization` and border pixels.
Run with:

```bash
python -m pytest test/pytest/test_indexed_geometry.py -v
```
