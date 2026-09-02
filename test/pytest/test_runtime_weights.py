"""End-to-end test for runtime-selected weight banks.

Two banks with deliberately different weights are packed from the manifest, and
each is shown to decode back to its own parameters and nothing else.

The packing checks need no FPGA tooling. The wrapper-generation check does need a
prior C/RTL synthesis and is skipped when no synthesized project is available.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Conv2D, Dense, Input
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.contrib.runtime_weights import pack
from hls4ml.contrib.runtime_weights.pack import PackingUnsupported

PART = 'xcvu9p-flga2104-2L-e'
N_IN, N_OUT = 8, 4


def _banks():
    """Two banks whose values cannot be confused: distinct signs and magnitudes,
    plus a walking-bit first column so a lane/address slip is unmistakable."""
    bank0 = np.zeros((N_IN, N_OUT), dtype=np.float64)
    bank1 = np.zeros((N_IN, N_OUT), dtype=np.float64)
    for i in range(N_IN):
        bank0[i, 0] = 0.125 * (2**i)  # walking bit
        bank1[i, 0] = -0.125 * (i + 1)
        for o in range(1, N_OUT):
            bank0[i, o] = (i * N_OUT + o + 1) / 64.0
            bank1[i, o] = -((i * N_OUT + o + 1) / 32.0)
    return [bank0, bank1]


def _manifest(tmp_path, reuse_factor=2, n_out=N_OUT):
    inp = Input(shape=(N_IN,), name='input_1')
    model = Model(inp, Dense(n_out, activation='linear', name='dense_1')(inp))
    model.get_layer('dense_1').set_weights([np.zeros((N_IN, n_out), np.float32), np.zeros((n_out,), np.float32)])
    cfg = hls4ml.utils.config_from_keras_model(
        model,
        granularity='model',
        backend='Vitis',
        default_precision='ap_fixed<16,6>',
        default_reuse_factor=reuse_factor,
    )
    cfg['Model']['Strategy'] = 'Resource'
    cfg['Model']['BramFactor'] = 0
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(tmp_path),
        project_name='rw_prj',
        backend='Vitis',
        io_type='io_parallel',
        part=PART,
        clock_period=10.0,
    )
    hls_model.write()
    return json.loads((Path(tmp_path) / 'firmware' / 'weights' / 'external_parameters.json').read_text())


def _unpack(port, words, n_in, n_out):
    """Invert the packing using only the manifest, to recover raw scalar codes."""
    width = port['precision']['width']
    block_size = port['layout']['block_size']
    out = np.zeros((n_in, n_out), dtype=np.int64)
    for i in range(n_in):
        for o in range(n_out):
            f = o * n_in + i
            code = (words[f % block_size] >> (width * (f // block_size))) & ((1 << width) - 1)
            out[i, o] = code
    return out


def test_two_banks_roundtrip(tmp_path):
    """Each bank packs to a distinct image that decodes back to its own weights."""
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')
    banks = _banks()

    images = [pack.pack_tensor(port, b) for b in banks]
    assert images[0] != images[1], 'banks must not produce identical images'

    for bank, words in zip(banks, images):
        assert len(words) == port['expected_depth']
        expected = np.vectorize(lambda v: pack.quantize(v, port['precision']['width'], port['precision']['integer']))(bank)
        np.testing.assert_array_equal(_unpack(port, words, N_IN, N_OUT), expected)


def test_single_scalar_mutation_touches_one_lane(tmp_path):
    """A one-scalar change must move exactly one lane of one word."""
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')

    base = _banks()[0]
    mutated = base.copy()
    mutated[3, 2] += 0.125

    words_a = pack.pack_tensor(port, base)
    words_b = pack.pack_tensor(port, mutated)

    differing = [i for i, (a, b) in enumerate(zip(words_a, words_b)) if a != b]
    assert len(differing) == 1

    width = port['precision']['width']
    f = 2 * N_IN + 3  # index(out=2, in=3)
    block_size = port['layout']['block_size']
    assert differing[0] == f % block_size

    delta = words_a[differing[0]] ^ words_b[differing[0]]
    assert delta >> (width * (f // block_size)) < (1 << width)
    assert delta & ((1 << (width * (f // block_size))) - 1) == 0


def test_bank_image_is_depth_stacked_and_padded(tmp_path):
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')

    image, stride = pack.build_bank_image(port, _banks())
    assert stride >= port['expected_depth']
    assert stride & (stride - 1) == 0, 'stride should be a power of two'
    assert len(image) == 2 * stride
    for bank_index in range(2):
        for pad in range(port['expected_depth'], stride):
            assert image[bank_index * stride + pad] == 0


def test_bias_scalar_bank(tmp_path):
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'bias')

    codes = pack.pack_flat(port, [0.5, -0.25, 1.0, -2.0])
    assert len(codes) == N_OUT
    assert codes[0] == pack.quantize(0.5, 16, 6)
    assert all(0 <= c < (1 << 16) for c in codes)


def test_packer_refuses_unclaimed_ordering(tmp_path):
    """A port the manifest declined to describe must not be packed."""
    manifest = _manifest(tmp_path)
    port = dict(next(p for p in manifest['ports'] if p['role'] == 'weight'))
    port['layout'] = None

    with pytest.raises(PackingUnsupported):
        pack.pack_tensor(port, _banks()[0])


def test_write_mem_format(tmp_path):
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')
    words = pack.pack_tensor(port, _banks()[0])

    out = tmp_path / 'bank.mem'
    pack.write_mem(out, words, port['expected_data_width'])
    lines = out.read_text().split()
    assert len(lines) == len(words)
    assert all(len(line) == port['expected_data_width'] // 4 for line in lines)
    assert int(lines[0], 16) == words[0]


def test_package_generates_wrapper(tmp_path, synthesis_config):
    """Post-export packaging emits the wrapper and preserves the stock IP."""
    from hls4ml.contrib.runtime_weights import fingerprint_ip, package

    if not synthesis_config['run_synthesis']:
        pytest.skip('set RUN_SYNTHESIS=true to run synthesis tests')

    manifest = _manifest(tmp_path)
    assert manifest['ports']

    inp = Input(shape=(N_IN,), name='input_1')
    model = Model(inp, Dense(N_OUT, activation='linear', name='dense_1')(inp))
    model.get_layer('dense_1').set_weights([_banks()[0].astype(np.float32), np.zeros((N_OUT,), np.float32)])
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity='model', backend='Vitis', default_precision='ap_fixed<16,6>', default_reuse_factor=2
    )
    cfg['Model']['Strategy'] = 'Resource'
    cfg['Model']['BramFactor'] = 0
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(tmp_path),
        project_name='rw_prj',
        backend='Vitis',
        io_type='io_parallel',
        part=PART,
        clock_period=10.0,
    )
    hls_model.write()
    hls_model.build(**synthesis_config['build_args']['Vitis'], log_to_stdout=False)

    before = fingerprint_ip(str(tmp_path), 'rw_prj')
    summary = package(str(tmp_path), 'rw_prj', n_banks=2)
    after = fingerprint_ip(str(tmp_path), 'rw_prj')

    assert before['combined'] == after['combined'], 'packaging must not modify the HLS IP'
    assert summary['n_banks'] == 2
    assert summary['bank_selection'].startswith('idle-time')
    assert {p['name'] for p in summary['banked_ports']} == {'w2', 'b2'}

    rtl = Path(tmp_path) / 'runtime_weights' / 'rtl'
    assert (rtl / 'rw_prj_runtime_weights.sv').exists()
    for module in ('bank_addr_mapper.sv', 'bank_select_latch.sv', 'parameter_bank.sv', 'scalar_bank_mux.sv'):
        assert (rtl / module).exists()
    assert (Path(tmp_path) / 'runtime_weights' / 'create_runtime_weights.tcl').exists()


def test_two_banks_rtl_simulation(tmp_path, synthesis_config):
    """Elaborate the generated wrapper and check numerical output per bank.

    This is the gate that pack/unpack round trips cannot provide: it catches a top
    that does not elaborate, and a bank that is selected a cycle late.
    """
    import runtime_weights_sim as sim

    from hls4ml.contrib.runtime_weights import fingerprint_ip, pack_flat, pack_tensor, package

    if not synthesis_config['run_synthesis']:
        pytest.skip('set RUN_SYNTHESIS=true to run synthesis tests')
    if not sim.have_xsim():
        pytest.skip('requires xvlog/xelab/xsim')

    banks_w = _banks()
    banks_b = [
        [0.5, -0.25, 1.0, -2.0],
        [-1.5, 0.75, -0.5, 0.25],
    ]

    # the IP is built with bank 0 baked in; the wrapper must override it at runtime
    inp = Input(shape=(N_IN,), name='input_1')
    model = Model(inp, Dense(N_OUT, activation='linear', name='dense_1')(inp))
    model.get_layer('dense_1').set_weights([banks_w[0].astype(np.float32), np.array(banks_b[0], np.float32)])
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity='model', backend='Vitis', default_precision='ap_fixed<16,6>', default_reuse_factor=2
    )
    cfg['Model']['Strategy'] = 'Resource'
    cfg['Model']['BramFactor'] = 0
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(tmp_path),
        project_name='rw_prj',
        backend='Vitis',
        io_type='io_parallel',
        part=PART,
        clock_period=10.0,
    )
    hls_model.write()
    hls_model.build(**synthesis_config['build_args']['Vitis'], log_to_stdout=False)

    before = fingerprint_ip(str(tmp_path), 'rw_prj')
    summary = package(str(tmp_path), 'rw_prj', n_banks=len(banks_w))
    assert summary['port_b_proven_unused']

    manifest = json.loads((tmp_path / 'firmware' / 'weights' / 'external_parameters.json').read_text())
    w_port = next(p for p in manifest['ports'] if p['role'] == 'weight')
    b_port = next(p for p in manifest['ports'] if p['role'] == 'bias')
    w_port = dict(w_port, actual_data_width=w_port['expected_data_width'])
    b_port = dict(b_port, actual_width=b_port['expected_data_width'])

    x = [0.5, -1.25, 0.75, 2.0, -0.5, 1.5, -2.0, 0.25]
    payload, expected = [], []
    for weights, bias in zip(banks_w, banks_b):
        payload.append((pack_tensor(w_port, weights), pack_flat(b_port, bias)))
        expected.append([sim.code_of(v) for v in sim.dense_reference(x, weights.tolist(), bias)])

    assert expected[0] != expected[1], 'banks must produce different outputs to be discriminating'

    tb = sim.write_testbench(
        str(tmp_path / 'tb_runtime_weights.sv'),
        'rw_prj',
        N_IN,
        N_OUT,
        [sim.code_of(v) for v in x],
        payload,
        expected,
        w_port,
        b_port,
    )
    passed, log = sim.run_xsim(
        str(tmp_path / 'xsim_work'),
        [str(tmp_path / 'runtime_weights' / 'rtl'), str(tmp_path / 'rw_prj_prj' / 'solution1' / 'syn' / 'verilog')],
        tb,
    )
    assert passed, f'two-bank RTL simulation failed:\n{log[-6000:]}'
    assert fingerprint_ip(str(tmp_path), 'rw_prj')['combined'] == before['combined']


# --- adapter-specific walking-bit tests -------------------------------------
# Each gives every scalar a code that uniquely identifies its logical position,
# then checks that every physical (word, lane) holds the scalar the manifest says
# it should. The expectation is recomputed in the test from the declared
# flat_order and layout, so it does not reuse the packer's own arithmetic.


def test_dense_weight_walking_bit(tmp_path):
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')

    # code f+1 at flat index f = out*n_in + in, i.e. W[i][o] = (o*n_in + i + 1)/1024
    weights = np.zeros((N_IN, N_OUT))
    for i in range(N_IN):
        for o in range(N_OUT):
            weights[i, o] = (o * N_IN + i + 1) / 1024.0

    words = pack.pack_tensor(port, weights)
    width = port['precision']['width']
    block_size = port['layout']['block_size']
    lanes = port['layout']['lanes']

    seen = {}
    for word_index, word in enumerate(words):
        for lane in range(lanes):
            code = (word >> (width * lane)) & ((1 << width) - 1)
            assert code != 0, f'lane {lane} of word {word_index} is empty'
            seen[(word_index, lane)] = code - 1  # recover the flat index

    assert len(seen) == N_IN * N_OUT
    for (word_index, lane), flat in seen.items():
        assert word_index == flat % block_size
        assert lane == flat // block_size


def test_dense_bias_walking_bit(tmp_path):
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'bias')

    values = [(i + 1) / 1024.0 for i in range(N_OUT)]
    codes = pack.pack_flat(port, values)

    assert port['layout']['mode'] == 'complete'
    assert codes == [i + 1 for i in range(N_OUT)], 'bias order must be b[i] = bias[i]'


def test_flatten_follows_declared_axis_order(tmp_path):
    """flatten is a transpose+ravel driven by flat_order, not a Dense special case."""
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')

    tensor = np.arange(N_IN * N_OUT).reshape(N_IN, N_OUT)
    flat = pack.flatten(port, tensor)

    assert port['flat_order']['tensor_axes'] == ['n_in', 'n_out']
    assert port['flat_order']['axes'] == ['n_out', 'n_in']
    assert flat == tensor.T.ravel().tolist()
    assert flat[:N_IN] == tensor[:, 0].tolist(), 'first n_in scalars are output 0'


def test_flatten_rejects_wrong_rank(tmp_path):
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')

    with pytest.raises(PackingUnsupported, match='dimensions'):
        pack.flatten(port, np.zeros((N_IN, N_OUT, 2)))


def test_unregistered_layer_cannot_be_packed(tmp_path):
    """A port the registry declined to describe has no layout to pack with."""
    inp = Input(shape=(4, 4, 2), name='input_1')
    model = Model(inp, Conv2D(3, (2, 2), padding='valid', activation='linear', name='conv2d_1')(inp))
    hls_model = _convert_conv(model, tmp_path / 'conv')

    from hls4ml.writer.external_parameters import build_manifest

    for port in build_manifest(hls_model)['ports']:
        assert port['layout'] is None
        with pytest.raises(PackingUnsupported):
            pack.pack_flat(port, [0.0] * port['n_scalars'])


def _convert_conv(model, out_dir):
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity='model', backend='Vitis', default_precision='ap_fixed<16,6>', default_reuse_factor=2
    )
    cfg['Model']['Strategy'] = 'Resource'
    cfg['Model']['BramFactor'] = 0
    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(out_dir),
        project_name='conv_prj',
        backend='Vitis',
        io_type='io_parallel',
        part=PART,
        clock_period=10.0,
    )
