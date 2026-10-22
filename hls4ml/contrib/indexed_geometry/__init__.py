"""
indexed_geometry
==================
hls4ml Extension API contribution for non-standard (e.g. hexagonal)
detector-pixel geometries indexed by a static neighbor map.

Provides three custom Keras layers, and their hls4ml Extension API
registration for the Vivado/Vitis backends (io_parallel and io_stream):

    - ``NeighborGatherLayer``      -- gathers neighbor features per pixel
    - ``IndexedConvolutionLayer``  -- applies a Conv2D per pixel
    - ``IndexedPoolingLayer``      -- applies a MaxPool/AvgPool per pixel

Usage
-----
    from hls4ml.contrib.indexed_geometry import (
        NeighborGatherLayer, IndexedConvolutionLayer, IndexedPoolingLayer,
    )

Importing this package registers the three layers with hls4ml
(idempotent -- safe to import more than once, or alongside code that
already imported one of the submodules directly).

See README.md for a worked example, and the ``precision_utils`` helper
in each example script for automatic fixed-point precision inference.
"""

from . import (
    indexed_conv,  # noqa: F401  (import triggers registration)
    indexed_pool,  # noqa: F401  (import triggers registration)
    neighbor_gather,  # noqa: F401  (import triggers registration)
)
from .keras_layers import IndexedConvolutionLayer, IndexedPoolingLayer, NeighborGatherLayer

__all__ = ['NeighborGatherLayer', 'IndexedConvolutionLayer', 'IndexedPoolingLayer']
