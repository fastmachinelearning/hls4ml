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
from tensorflow.keras.layers import Dense, Input
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
    return json.loads((Path(tmp_path) / 'firmware' / 'weights' / 'bram_manifest.json').read_text())


def _unpack(port, words, n_in, n_out):
    """Invert the packing using only the manifest, to recover raw scalar codes."""
    width = port['precision']['width']
    lane_div = int(port['reshape']['lane_of_flat_index'].split('//')[1])
    word_div = int(port['reshape']['word_of_flat_index'].split('%')[1])
    out = np.zeros((n_in, n_out), dtype=np.int64)
    for i in range(n_in):
        for o in range(n_out):
            f = o * n_in + i
            code = (words[f % word_div] >> (width * (f // lane_div))) & ((1 << width) - 1)
            out[i, o] = code
    return out


def test_two_banks_roundtrip(tmp_path):
    """Each bank packs to a distinct image that decodes back to its own weights."""
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')
    banks = _banks()

    images = [pack.pack_weight_bank(port, b.tolist(), N_IN, N_OUT) for b in banks]
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

    words_a = pack.pack_weight_bank(port, base.tolist(), N_IN, N_OUT)
    words_b = pack.pack_weight_bank(port, mutated.tolist(), N_IN, N_OUT)

    differing = [i for i, (a, b) in enumerate(zip(words_a, words_b)) if a != b]
    assert len(differing) == 1

    width = port['precision']['width']
    f = 2 * N_IN + 3  # index(out=2, in=3)
    word_div = int(port['reshape']['word_of_flat_index'].split('%')[1])
    lane_div = int(port['reshape']['lane_of_flat_index'].split('//')[1])
    assert differing[0] == f % word_div

    delta = words_a[differing[0]] ^ words_b[differing[0]]
    assert delta >> (width * (f // lane_div)) < (1 << width)
    assert delta & ((1 << (width * (f // lane_div))) - 1) == 0


def test_bank_image_is_depth_stacked_and_padded(tmp_path):
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')

    image, stride = pack.build_bank_image(port, [b.tolist() for b in _banks()], N_IN, N_OUT)
    assert stride >= port['expected_depth']
    assert stride & (stride - 1) == 0, 'stride should be a power of two'
    assert len(image) == 2 * stride
    for bank_index in range(2):
        for pad in range(port['expected_depth'], stride):
            assert image[bank_index * stride + pad] == 0


def test_bias_scalar_bank(tmp_path):
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'bias')

    codes = pack.pack_scalar_bank(port, [0.5, -0.25, 1.0, -2.0])
    assert len(codes) == N_OUT
    assert codes[0] == pack.quantize(0.5, 16, 6)
    assert all(0 <= c < (1 << 16) for c in codes)


def test_packer_refuses_unclaimed_ordering(tmp_path):
    """A port the manifest declined to describe must not be packed."""
    manifest = _manifest(tmp_path)
    port = dict(next(p for p in manifest['ports'] if p['role'] == 'weight'))
    port['reshape'] = {'lane_of_flat_index': None, 'word_of_flat_index': None}

    with pytest.raises(PackingUnsupported):
        pack.pack_weight_bank(port, _banks()[0].tolist(), N_IN, N_OUT)


def test_write_mem_format(tmp_path):
    manifest = _manifest(tmp_path)
    port = next(p for p in manifest['ports'] if p['role'] == 'weight')
    words = pack.pack_weight_bank(port, _banks()[0].tolist(), N_IN, N_OUT)

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

    from hls4ml.contrib.runtime_weights import fingerprint_ip, pack_scalar_bank, pack_weight_bank, package

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

    manifest = json.loads((tmp_path / 'firmware' / 'weights' / 'bram_manifest.json').read_text())
    w_port = next(p for p in manifest['ports'] if p['role'] == 'weight')
    b_port = next(p for p in manifest['ports'] if p['role'] == 'bias')
    w_port = dict(w_port, actual_data_width=w_port['expected_data_width'])
    b_port = dict(b_port, actual_width=b_port['expected_data_width'])

    x = [0.5, -1.25, 0.75, 2.0, -0.5, 1.5, -2.0, 0.25]
    payload, expected = [], []
    for weights, bias in zip(banks_w, banks_b):
        payload.append((pack_weight_bank(w_port, weights.tolist(), N_IN, N_OUT), pack_scalar_bank(b_port, bias)))
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
