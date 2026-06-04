"""Bit-exact precision propagation through grouped / depthwise Conv1D / Conv2D.

The ``produce_kif`` handler for ``Conv1D`` / ``Conv2D`` assumed a non-grouped
kernel (``kernel.shape[-2] == n_chan``). A grouped or depthwise convolution
stores only ``in_per_group`` input channels per filter, so the im2col buffer no
longer lines up with the full-channel input and the bit_exact pass raised, e.g.::

    ValueError: could not broadcast input array from shape (48,) into shape (3,)

Each group is an independent standard convolution over its own channel slice, so
the fix processes the groups separately and concatenates along the channel axis.

Scope: this exercises the (backend-independent) bit_exact precision-propagation
pass. The io_parallel/io_stream *codegen* for grouped convolutions is a separate
concern, so rather than comparing a compiled prediction the test asserts the
contract the pass must uphold: the output precision it assigns represents the
quantized Keras output exactly (no rounding, no saturation).
"""

from pathlib import Path

import keras
import numpy as np
import pytest

from hls4ml.converters import convert_from_keras_model

try:
    from hgq.config import QuantizerConfigScope
    from hgq.layers import QConv1D, QConv2D
    from hgq.utils import trace_minmax
except ImportError:
    pytest.skip('HGQ2 is not installed', allow_module_level=True)

from keras.layers import Input  # noqa: E402

from hls4ml.model.layers import Conv1D, Conv2D  # noqa: E402
from hls4ml.model.optimizer.passes.bit_exact import produce_kif  # noqa: E402

test_root_path = Path(__file__).parent


def _assert_exactly_representable(values, k, i, f):
    """Assert every value lands exactly on the signed fixed-point grid (k, i, f).

    ``k``/``i``/``f`` may be per-element arrays (broadcasting over ``values``) or
    scalars. A value is representable when it is a multiple of 2**-f and lies in
    the closed range [-(2**i) * k, 2**i - 2**-f]; if produce_kif under-allocated
    any of k, i or f, the regridded value differs and the assertion fails.
    """
    values = values.astype(np.float64)
    k = np.asarray(k, dtype=np.float64)
    i = np.asarray(i, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    delta = 2.0**-f
    lo = -(2.0**i) * k
    hi = 2.0**i - delta
    regridded = np.clip(np.round(values / delta) * delta, lo, hi)
    np.testing.assert_array_equal(regridded, values)


def _result_kif(node):
    """k, i (excluding sign), f of the precision bit_exact assigned to ``node``."""
    precision = node.get_output_variable().type.precision
    k = int(precision.signed)
    f = precision.fractional
    i = precision.integer - k
    return k, i, f


def _find(hls_model, cls, name):
    return next(node for node in hls_model.graph.values() if isinstance(node, cls) and node.name == name)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI'])
@pytest.mark.parametrize('n_chan, groups', [(16, 16), (16, 4), (16, 1)], ids=['depthwise', 'grouped', 'dense'])
def test_bit_exact_grouped_conv1d(test_case_id, backend, n_chan, groups):
    """A grouped / depthwise QConv1D must convert through the bit_exact flow and
    be assigned an output precision that represents the quantized Keras output
    exactly. The leading QConv1D inserts the FixedPointQuantizer that triggers
    the bit_exact pass."""
    with QuantizerConfigScope(f0=4, i0=4):
        inp = Input((16, n_chan))
        x = QConv1D(n_chan, 1, name='c0')(inp)
        out = QConv1D(n_chan, 3, padding='same', groups=groups, name='cg')(x)
        model = keras.Model(inp, out)

    data = np.random.default_rng(0).standard_normal((1000, 16, n_chan)).astype(np.float32)
    r_keras = trace_minmax(model, data, return_results=True)

    precision = 'ac_fixed<2,0>' if backend == 'oneAPI' else 'ap_fixed<1,0>'
    hls_config = {'Model': {'Precision': precision, 'ReuseFactor': 1, 'Strategy': 'latency'}}
    output_dir = str(test_root_path / test_case_id)
    # Conversion runs the bit_exact pass; this raised ValueError pre-fix for groups > 1.
    hls_model = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type='io_parallel'
    )

    conv = _find(hls_model, Conv1D, 'cg')
    # The single output precision bit_exact assigned must represent the true output.
    _assert_exactly_representable(r_keras, *_result_kif(conv))

    # Per-channel check (stronger: catches per-channel / group-ordering errors the
    # max-aggregated result_t could mask). oneAPI transposes conv weights after the
    # pass, so the channels_last produce_kif recompute is only valid for Vivado/Vitis.
    if backend != 'oneAPI':
        _assert_exactly_representable(r_keras, *produce_kif(conv))


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI'])
@pytest.mark.parametrize('n_chan, groups', [(4, 4), (4, 2), (4, 1)], ids=['depthwise', 'grouped', 'dense'])
def test_bit_exact_grouped_conv2d(test_case_id, backend, n_chan, groups):
    """2D counterpart of :func:`test_bit_exact_grouped_conv1d`."""
    with QuantizerConfigScope(f0=4, i0=4):
        inp = Input((8, 8, n_chan))
        x = QConv2D(n_chan, 1, name='c0')(inp)
        out = QConv2D(n_chan, 3, padding='same', groups=groups, name='cg')(x)
        model = keras.Model(inp, out)

    data = np.random.default_rng(1).standard_normal((500, 8, 8, n_chan)).astype(np.float32)
    r_keras = trace_minmax(model, data, return_results=True)

    precision = 'ac_fixed<2,0>' if backend == 'oneAPI' else 'ap_fixed<1,0>'
    hls_config = {'Model': {'Precision': precision, 'ReuseFactor': 1, 'Strategy': 'latency'}}
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type='io_parallel'
    )

    conv = _find(hls_model, Conv2D, 'cg')
    _assert_exactly_representable(r_keras, *_result_kif(conv))
    if backend != 'oneAPI':
        _assert_exactly_representable(r_keras, *produce_kif(conv))
