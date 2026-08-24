"""Regression tests for fastmachinelearning/hls4ml#1523.

``hgq.layers.QSoftmax`` trains two lookup tables (``exp_table`` and ``inv_table``). hls4ml
used to propagate only their sizes and types, leaving the generated C++ to rebuild the
contents from ``std::exp`` / ``1/x`` at runtime. These tests pin down that the trained
tables now reach the generated code, and that plain Keras softmax is unaffected.
"""

from contextlib import nullcontext
from pathlib import Path

import keras
import numpy as np
import pytest

hgq = pytest.importorskip('hgq')

from hgq.config import QuantizerConfigScope  # noqa: E402
from hgq.layers import QDense, QSoftmax  # noqa: E402
from hgq.utils import trace_minmax  # noqa: E402

import hls4ml  # noqa: E402
from hls4ml.backends.fpga.passes.hgq_proxy_model import generate_mask_fn  # noqa: E402
from hls4ml.converters.keras_v3.hgq2._base import extract_fixed_quantizer_config  # noqa: E402
from hls4ml.converters.keras_v3.hgq2.unary_lut import extract_lut_table, fixed_grid_by_bit_pattern  # noqa: E402

test_root_path = Path(__file__).parent


def _build_model(stable, shape=(16,), io_type='io_parallel', n_out=8, seed=42):
    keras.utils.set_random_seed(seed)
    # Heterogeneous activation quantization is io_parallel-only in hls4ml, so the datalane
    # quantizers have to be pinned to a single bitwidth to reach the io_stream kernels.
    scope = QuantizerConfigScope(place='datalane', heterogeneous_axis=()) if io_type == 'io_stream' else nullcontext()
    with scope:
        inp = keras.Input(shape)
        x = QDense(n_out)(inp)
        out = QSoftmax(axis=-1, stable=stable, name='sm')(x)
        model = keras.Model(inp, out)

    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(200,) + shape).astype(np.float32)
    trace_minmax(model, X)
    return model, X


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('stable', [True, False])
@pytest.mark.parametrize('shape', [(16,), (4, 16)], ids=['1d', 'multidim'])
def test_hgq2_softmax_uses_trained_tables(test_case_id, backend, io_type, stable, shape):
    model, X = _build_model(stable, shape=shape, io_type=io_type)
    r_keras = np.asarray(model(X))

    impl = 'stable' if stable else 'latency'
    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend=backend, io_type=io_type, output_dir=odir, part='xcvu13p-flga2577-2-e'
    )

    node = hls_model.graph['sm']
    assert node.get_attr('implementation') == impl

    # The trained tables are carried as weights ...
    assert 'exp_table' in node.weights and 'inv_table' in node.weights
    exp_var, inv_var = node.get_weights('exp_table'), node.get_weights('inv_table')

    # ... sized so that the C++ table index is a plain reinterpretation of the address word
    if stable:
        exp_addr_t = node.attributes['inp_norm_t'].precision
    else:
        exp_addr_t = node.get_input_variable().type.precision
    inv_addr_t = node.attributes['inv_inp_t'].precision
    assert len(exp_var.data) == node.get_attr('exp_table_size') == 2**exp_addr_t.width
    assert len(inv_var.data) == node.get_attr('inv_table_size') == 2**inv_addr_t.width

    # ... typed with what bit_exact derived, not with a type re-inferred from the data
    assert exp_var.type is node.attributes['exp_table_t']
    assert inv_var.type is node.attributes['inv_table_t']

    # The non-serializable builders stashed by the converter must not survive
    assert '_exp_table_fn' not in node.attributes and '_inv_table_fn' not in node.attributes

    # Table contents match an independent recomputation straight from hgq ...
    sm_layer = model.get_layer('sm')
    k = int(bool(exp_addr_t.signed))
    kif = (k, exp_addr_t.integer - k, exp_addr_t.width - exp_addr_t.integer)
    expected = extract_lut_table(sm_layer.exp_table.activation, sm_layer.exp_table.oq, kif)
    np.testing.assert_array_equal(np.asarray(exp_var.data).ravel(), np.asarray(expected).ravel())

    # ... and are not simply what the generic C++ reconstruction would produce unquantized
    naive = np.exp(fixed_grid_by_bit_pattern(*kif) * (-1.0 if stable else 1.0) * node.get_attr('exp_scale'))
    assert not np.allclose(np.asarray(exp_var.data).ravel(), naive)

    # The generated code passes the tables in
    hls_model.write()
    call = [ln for ln in open(f'{odir}/firmware/myproject.cpp') if 'nnet::softmax' in ln]
    assert len(call) == 1
    assert exp_var.name in call[0] and inv_var.name in call[0]
    assert Path(f'{odir}/firmware/weights/{exp_var.name}.h').exists()
    assert f'#include "weights/{exp_var.name}.h"' in open(f'{odir}/firmware/parameters.h').read()

    # End to end: HGQ2 + bit_exact promises exact agreement with Keras
    hls_model.compile()
    r_hls = np.asarray(hls_model.predict(X)).reshape(r_keras.shape)
    assert np.std(r_hls) > 0
    np.testing.assert_array_equal(r_hls, r_keras)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_plain_keras_softmax_keeps_runtime_tables(test_case_id, backend):
    """Negative control: non-HGQ softmax must keep the two-argument call."""
    keras.utils.set_random_seed(0)
    inp = keras.Input((8,))
    out = keras.layers.Softmax(name='sm')(inp)
    model = keras.Model(inp, out)

    odir = str(test_root_path / test_case_id)
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, backend=backend, output_dir=odir, part='xcvu13p-flga2577-2-e'
    )

    node = hls_model.graph['sm']
    assert 'exp_table' not in node.weights and 'inv_table' not in node.weights

    hls_model.write()
    (call,) = (ln for ln in open(f'{odir}/firmware/myproject.cpp') if 'nnet::softmax' in ln)
    args = call.split('>(')[1].split(')')[0]
    assert len(args.split(',')) == 2, call


def test_dead_channels_stay_zero():
    """A heterogeneous quantizer channel trained to a negative total width is a dead
    channel and must be rendered as a constant zero, not resurrected as a 1-bit channel.
    """
    from hgq.quantizer import Quantizer, QuantizerConfig

    q = Quantizer(QuantizerConfig('kif', 'weight', heterogeneous_axis=(0,)))
    x = keras.ops.arange(8.0, dtype='float32')
    q.build(x.shape)
    q(x)

    i_var, f_var, _ = q.quantizer.weights
    i_var.assign(np.array([1, 1, 1, -2, 1, 1, 1, 1], dtype=np.float32))
    f_var.assign(np.array([3, 3, 3, -4, 3, 3, 3, 3], dtype=np.float32))

    inp = keras.Input(shape=(8,), name='x')
    conf = extract_fixed_quantizer_config(q, inp, is_input=True)
    k_arr, b_arr, i_arr = (np.ravel(a) for a in conf['mask_kbi'])

    assert b_arr[3] == 0, f'dead channel clamped to {b_arr[3]}, expected 0'
    assert i_arr[3] == 0
    assert (b_arr >= 0).all()

    mask_fn = generate_mask_fn('mask', (8,), *(a.reshape(1, 8) for a in (k_arr, b_arr, i_arr)), 'RND', 'SAT', 'vivado')
    assert 'out[3] = 0;' in mask_fn
