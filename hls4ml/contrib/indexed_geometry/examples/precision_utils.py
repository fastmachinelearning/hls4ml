"""
precision_utils.py
====================
Utility functions for converting Keras models with non-standard pixel
geometry to hls4ml firmware, with automatic fixed-point precision
inference.

The main entry point is ``get_hls_config()``, which replaces the standard
``hls4ml.utils.config_from_keras_model()`` call. It runs a lightweight
profiling forward pass in Keras to measure the maximum activation
magnitude per layer, then automatically selects per-layer ``ap_fixed``
precisions with sufficient integer width to avoid overflow.

This is an example-only helper, not part of the ``indexed_geometry``
contrib package itself: it isn't specific to the custom layers, and
hls4ml's own built-in 'auto' precision inference is the recommended
starting point for new models (see the package README, "Choosing
precision"). It's kept here because both examples in this directory
were originally developed and validated against it.

Typical usage
-------------
    import hls4ml
    from hls4ml.contrib.indexed_geometry import (
        NeighborGatherLayer, IndexedConvolutionLayer, IndexedPoolingLayer,
    )
    from precision_utils import get_hls_config

    hls_config = get_hls_config(keras_model, x_sample)
    hmodel = hls4ml.converters.convert_from_keras_model(
        keras_model,
        output_dir='my_hls_project',
        backend='Vitis',
        io_type='io_stream',
        hls_config=hls_config,
    )

Notes
-----
- The profiling pass runs entirely in Keras/NumPy -- no HLS compilation
  is required at this stage.
- ``NeighborGatherLayer`` models can be profiled and run directly with
  real neighbor indices (``-1`` included): its zero-masking is a
  multiplicative mask applied regardless of what "garbage" index
  ``tf.gather`` substitutes internally for ``-1``, so ``predict()``
  already returns the correct, zero-masked result without any index
  substitution. The optional ``profile_model`` parameter below is for
  models that genuinely need a different network for profiling than for
  conversion; it is not needed to work around ``-1`` indices.
"""

import math

import numpy as np
import tensorflow as tf

import hls4ml


def _bits_for_range(max_abs_val, frac_bits=16, safety_margin=2):
    """
    Compute the minimum number of integer bits (including sign bit) required
    to represent ``max_abs_val`` without overflow, plus a safety margin.

    Parameters
    ----------
    max_abs_val : float
        Maximum absolute activation value observed for a layer.
    frac_bits : int
        Number of fractional bits (kept fixed across layers).
    safety_margin : int
        Additional integer bits added beyond the strict minimum.

    Returns
    -------
    (int_bits, frac_bits) : tuple of int
        Integer and fractional widths for an
        ``ap_fixed<int_bits + frac_bits, int_bits>`` type.
    """
    if max_abs_val <= 0:
        int_bits = 2
    else:
        int_bits = math.ceil(math.log2(max_abs_val + 1e-9)) + 1 + safety_margin
    int_bits = max(int_bits, 2)
    return int_bits, frac_bits


def _apfixed_str(int_bits, frac_bits):
    """Return an hls4ml precision string, e.g. ``'ap_fixed<24,8>'``."""
    return f'ap_fixed<{int_bits + frac_bits},{int_bits}>'


def _profile_activations(model, x_sample):
    """
    Run a single forward pass and record the maximum absolute activation
    value for each layer.

    Parameters
    ----------
    model : tf.keras.Model
        Model to profile. Must be executable with ``predict()`` on
        ``x_sample`` without errors (i.e. no ``-1`` neighbor indices).
    x_sample : np.ndarray
        Representative input batch.

    Returns
    -------
    dict
        Mapping ``{ layer_name: max_abs_activation }`` for every
        non-input layer in the model.
    """
    layer_outputs = []
    layer_names = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)

    debug_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)

    for layer in model.layers:
        try:
            debug_layer = debug_model.get_layer(layer.name)
            if layer.get_weights():
                debug_layer.set_weights(layer.get_weights())
        except ValueError:
            pass

    results = debug_model.predict(x_sample, verbose=0)
    if not isinstance(results, list):
        results = [results]

    return {name: float(np.abs(out).max()) for name, out in zip(layer_names, results)}


def get_hls_config(
    model,
    x_sample,
    profile_model=None,
    default_frac_bits=16,
    safety_margin=2,
    reuse_factor=1,
    backend='Vitis',
    verbose=True,
):
    """
    Build an hls4ml configuration with automatically inferred per-layer
    fixed-point precisions.

    For each layer the integer bit-width is chosen to accommodate the
    maximum absolute activation observed during a profiling forward pass,
    plus a configurable safety margin. The fractional bit-width is kept
    fixed at ``default_frac_bits`` across all layers.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to convert. May contain neighbor indices with
        ``-1`` values (border pixels): ``NeighborGatherLayer`` handles
        these correctly on its own, so no index substitution is needed.
    x_sample : np.ndarray
        A representative input batch (typically 10-50 samples). Used
        only for profiling -- no HLS compilation is performed.
    profile_model : tf.keras.Model, optional
        An alternative model used exclusively for the profiling pass,
        for cases where profiling genuinely needs a different network
        than the one being converted (not needed to work around ``-1``
        neighbor indices; see the module docstring). Precision decisions
        are still applied to ``model``.
    default_frac_bits : int
        Number of fractional bits for all ``ap_fixed`` types (default 16).
    safety_margin : int
        Extra integer bits added above the strict minimum required to
        represent the observed activation range (default 2).
    reuse_factor : int
        hls4ml reuse factor passed to ``config_from_keras_model()``
        (default 1, fully parallel).
    backend : str
        hls4ml backend passed to ``config_from_keras_model()`` (default
        ``'Vitis'``).
    verbose : bool
        If ``True``, print a per-layer precision report to stdout
        (default ``True``).

    Returns
    -------
    dict
        hls4ml configuration dictionary ready to be passed to
        ``hls4ml.converters.convert_from_keras_model()``.
    """
    if verbose:
        print('=' * 60)
        print('  precision_utils -- automatic precision inference')
        print('=' * 60)
        print(f'  Profiling {len(x_sample)} sample(s)...')

    _model_for_profiling = profile_model if profile_model is not None else model
    profile = _profile_activations(_model_for_profiling, x_sample)

    # Start from a conservatively wide default to avoid silent precision
    # issues in layers not explicitly overridden below.
    hls_config = hls4ml.utils.config_from_keras_model(
        model,
        granularity='name',
        backend=backend,
        default_precision='ap_fixed<32,16>',
        default_reuse_factor=reuse_factor,
    )

    if verbose:
        print()
        print(f'  {"Layer":<45} {"Max act":>10}  {"Precision":>20}')
        print(f'  {"-" * 45} {"-" * 10}  {"-" * 20}')

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        name = layer.name
        max_val = profile.get(name, 0.0)
        int_bits, frac_bits = _bits_for_range(
            max_val,
            frac_bits=default_frac_bits,
            safety_margin=safety_margin,
        )
        precision = _apfixed_str(int_bits, frac_bits)

        if name in hls_config['LayerName']:
            cfg = hls_config['LayerName'][name]
            cfg['Precision'] = {'result': precision}
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.BatchNormalization)):
                cfg['Precision']['accum'] = precision

        if verbose:
            display_name = name if len(name) <= 45 else name[:42] + '...'
            print(f'  {display_name:<45} {max_val:>10.3f}  {precision:>20}')

    if verbose:
        print()
        print(f'  Config ready. Reuse factor: {reuse_factor}')
        print('=' * 60)

    return hls_config
