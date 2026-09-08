"""Runtime-selected weight banks: packing, wrapper generation, and RTL behaviour.

Most tests are plain Python. Six need FPGA tooling and are gated on
RUN_SYNTHESIS (conftest.py's synthesis_config), which generate_ci_yaml.py sets
to "true" for the GitLab pipeline (pinned there to Vivado 2020.1 / Vitis 2024.1):

  test_banks_rtl_simulation[2,4]          Vitis HLS + xsim (one shared synthesis)
  test_pointwise_two_banks_rtl_simulation Vitis HLS + xsim, one per input rank
  test_latch_rejects_out_of_range_bank    xsim only
  test_vivado_synthesizes_the_wrapper     Vitis HLS + Vivado

They were verified against Vitis 2025.1/2025.2; older versions are untested. xsim
on PATH is this file's own requirement, not a repo convention, so they skip
without it. Run them locally when changing the RTL templates or package.py's
top-generation logic: the Python tests cannot see a top that fails to elaborate,
or a bank selected a cycle late.
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
from hls4ml.writer.external_parameters import MANIFEST_FILENAME

PROJECT = 'rw_prj'
N_IN = 8
PRECISION = 'ap_fixed<16,6>'

# One model covers every synthesized case: three external weight memories with
# different widths and depths, three bias bundles of different sizes, both
# verified Dense kernel variants, and two output heads of different sizes.
#                  n_in, n_out, reuse_factor
LAYERS = {
    'trunk': (N_IN, 8, 2),  # rf < n_in        -> dense_resource_rf_leq_nin
    'head_a': (N_IN, 4, 16),  # rf > n_in, rem 0 -> dense_resource_rf_gt_nin_rem0
    'head_b': (N_IN, 6, 8),  # rf == n_in       -> dense_resource_rf_leq_nin
}
HEADS = ('head_a', 'head_b')


def _bank_tensors(n_banks):
    """Per-bank weights and biases for every layer, on the ap_fixed<16,6> grid.

    Kept small enough that no accumulator wraps, so a bank selected wrongly shows
    up as a wrong value rather than as an aliased one.
    """
    banks = []
    for k in range(n_banks):
        entry = {}
        for li, (layer, (n_in, n_out, _)) in enumerate(LAYERS.items()):
            w = np.zeros((n_in, n_out))
            for i in range(n_in):
                for o in range(n_out):
                    w[i, o] = (-1) ** (k + o) * (((i * n_out + o + 17 * k + 5 * li) % 23) + 1) / 128.0
            b = np.array([(-1) ** (o + k) * (o + 1 + 3 * k + li) / 64.0 for o in range(n_out)])
            entry[layer] = (w, b)
        banks.append(entry)
    return banks


def _convert_model(out_dir, tensors=None, bram_factor=0):
    """The single model builder. Part and clock period are hls4ml defaults."""
    inp = Input(shape=(N_IN,), name='input_1')
    trunk = Dense(LAYERS['trunk'][1], activation='linear', name='trunk')(inp)
    outs = [Dense(LAYERS[h][1], activation='linear', name=h)(trunk) for h in HEADS]
    model = Model(inp, outs)
    for layer, (n_in, n_out, _) in LAYERS.items():
        w, b = tensors[layer] if tensors else (np.zeros((n_in, n_out)), np.zeros(n_out))
        model.get_layer(layer).set_weights([np.asarray(w, np.float32), np.asarray(b, np.float32)])

    cfg = hls4ml.utils.config_from_keras_model(
        model,
        granularity='name',
        backend='Vitis',
        default_precision='ap_fixed<16,6>',
        default_reuse_factor=2,
    )
    cfg['Model']['Strategy'] = 'Resource'
    cfg['Model']['BramFactor'] = bram_factor
    for layer, (_, _, reuse) in LAYERS.items():
        cfg['LayerName'][layer]['Strategy'] = 'Resource'
        cfg['LayerName'][layer]['ReuseFactor'] = reuse
    # Name granularity defaults every precision to 'auto', which widens each
    # layer's accumulator and result (ap_fixed<36,16>, <56,26>, ...). Pin them so
    # the arithmetic stays the single type _reference and the testbench model.
    for entry in cfg['LayerName'].values():
        for key, value in entry.get('Precision', {}).items():
            if value == 'auto':
                entry['Precision'][key] = PRECISION
    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(out_dir),
        project_name=PROJECT,
        backend='Vitis',
        io_type='io_parallel',
    )


def _manifest(tmp_path):
    hls_model = _convert_model(tmp_path)
    hls_model.write()
    return json.loads((Path(tmp_path) / 'firmware' / 'weights' / MANIFEST_FILENAME).read_text())


def _port(manifest, layer, role):
    return next(p for p in manifest['ports'] if p['layer'] == layer and p['role'] == role)


def _unpack(port, words, n_in, n_out):
    """Invert the packing using only the manifest."""
    width = port['precision']['width']
    block_size = port['layout']['block_size']
    out = np.zeros((n_in, n_out), dtype=np.int64)
    for i in range(n_in):
        for o in range(n_out):
            f = o * n_in + i
            code = (words[f % block_size] >> (width * (f // block_size))) & ((1 << width) - 1)
            out[i, o] = code
    return out


def _reference(x, tensors):
    """Fixed-point forward pass, heads in model-output order."""
    import runtime_weights_sim as sim

    trunk = sim.dense_reference(x, tensors['trunk'][0].tolist(), tensors['trunk'][1].tolist())
    return [sim.dense_reference(trunk, tensors[h][0].tolist(), tensors[h][1].tolist()) for h in HEADS]


@pytest.fixture(scope='module')
def written_project(tmp_path_factory):
    """Written once (no synthesis): the pure-Python tests read it."""
    path = tmp_path_factory.mktemp('rw_manifest')
    hls_model = _convert_model(path)
    hls_model.write()
    return hls_model


# trunk weight is 8*8=64, every other parameter is 48 or fewer, so this threshold
# externalizes exactly one of them and leaves the rest fixed in the compute IP
PARTIAL_BRAM_FACTOR = 48


@pytest.fixture(scope='module')
def partial_project(tmp_path_factory):
    """Only some parameters externalized, so fixed ones exist to disagree about."""
    hls_model = _convert_model(tmp_path_factory.mktemp('rw_partial'), bram_factor=PARTIAL_BRAM_FACTOR)
    hls_model.write()
    return hls_model


@pytest.fixture(scope='module')
def manifest(written_project):
    path = Path(written_project.config.get_output_dir()) / 'firmware' / 'weights' / MANIFEST_FILENAME
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def synthesized_project(tmp_path_factory, synthesis_config):
    """One Vitis synthesis, shared by every test that needs real RTL."""
    if not synthesis_config['run_synthesis']:
        pytest.skip('set RUN_SYNTHESIS=true to run synthesis tests')

    path = tmp_path_factory.mktemp('rw_synth')
    hls_model = _convert_model(path, tensors=_bank_tensors(1)[0])
    hls_model.write()
    hls_model.build(**synthesis_config['build_args']['Vitis'], log_to_stdout=False)
    outputs = [(v.name, v.size()) for v in hls_model.get_output_variables()]
    return path, PROJECT, outputs


# --- packing -----------------------------------------------------------------


@pytest.mark.parametrize('layer', list(LAYERS))
def test_packing_places_each_scalar_at_the_declared_word_and_lane(manifest, layer):
    """Scalar f lands at (word f % block_size, lane f // block_size)."""
    n_in, n_out, _ = LAYERS[layer]
    port = _port(manifest, layer, 'weight')
    width = port['precision']['width']
    block_size = port['layout']['block_size']

    # round trip, and two banks must not collapse to one image
    banks = [t[layer][0] for t in _bank_tensors(2)]
    images = [pack.pack_tensor(port, b) for b in banks]
    assert images[0] != images[1], 'banks must not produce identical images'
    for bank, words in zip(banks, images):
        assert len(words) == port['expected_depth']
        expected = np.vectorize(lambda v: pack.quantize(v, width, port['precision']['integer']))(bank)
        np.testing.assert_array_equal(_unpack(port, words, n_in, n_out), expected)

    # one scalar changed -> one lane of one word
    mutated = banks[0].copy()
    mutated[3, 2] += 0.125
    words_a, words_b = images[0], pack.pack_tensor(port, mutated)
    differing = [i for i, (a, b) in enumerate(zip(words_a, words_b)) if a != b]
    assert len(differing) == 1

    f = 2 * n_in + 3  # index(out=2, in=3)
    assert differing[0] == f % block_size
    delta = words_a[differing[0]] ^ words_b[differing[0]]
    assert delta >> (width * (f // block_size)) < (1 << width)
    assert delta & ((1 << (width * (f // block_size))) - 1) == 0

    # walking bit: every scalar lands where declared
    unique = np.zeros((n_in, n_out))
    for i in range(n_in):
        for o in range(n_out):
            unique[i, o] = (o * n_in + i + 1) / 1024.0
    words = pack.pack_tensor(port, unique)
    seen = {}
    for wi, word in enumerate(words):
        for lane in range(port['layout']['lanes']):
            code = (word >> (width * lane)) & ((1 << width) - 1)
            assert code != 0, f'lane {lane} of word {wi} is empty'
            seen[(wi, lane)] = code - 1
    assert len(seen) == n_in * n_out
    for (wi, lane), flat in seen.items():
        assert (wi, lane) == (flat % block_size, flat // block_size)


def test_both_verified_kernel_variants_are_covered(manifest):
    """The fixture must keep exercising each variant the manifest claims."""
    variants = {_port(manifest, layer, 'weight')['kernel_variant'] for layer in LAYERS}
    assert variants == {'dense_resource_rf_leq_nin', 'dense_resource_rf_gt_nin_rem0'}

    geometries = {
        (_port(manifest, la, 'weight')['expected_data_width'], _port(manifest, la, 'weight')['expected_depth'])
        for la in LAYERS
    }
    assert len(geometries) == len(LAYERS), 'layers should not share a geometry'


def test_bank_image_is_depth_stacked_and_padded(manifest):
    port = _port(manifest, 'trunk', 'weight')

    image, stride = pack.build_bank_image(port, [t['trunk'][0] for t in _bank_tensors(2)])
    assert stride >= port['expected_depth']
    assert stride & (stride - 1) == 0, 'stride should be a power of two'
    assert len(image) == 2 * stride
    for bank_index in range(2):
        for pad in range(port['expected_depth'], stride):
            assert image[bank_index * stride + pad] == 0


def test_bias_scalar_bank(manifest):
    port = _port(manifest, 'head_a', 'bias')

    codes = pack.pack_flat(port, [0.5, -0.25, 1.0, -2.0])
    assert len(codes) == LAYERS['head_a'][1]
    assert codes[0] == pack.quantize(0.5, 16, 6)
    assert all(0 <= c < (1 << 16) for c in codes)


def test_dense_bias_walking_bit(manifest):
    n_out = LAYERS['trunk'][1]
    port = _port(manifest, 'trunk', 'bias')

    codes = pack.pack_flat(port, [(i + 1) / 1024.0 for i in range(n_out)])

    assert port['layout']['mode'] == 'complete'
    assert codes == [i + 1 for i in range(n_out)], 'bias order must be b[i] = bias[i]'


def test_flatten_follows_declared_axis_order(manifest):
    """flatten is a transpose+ravel driven by flat_order."""
    n_in, n_out, _ = LAYERS['trunk']
    port = _port(manifest, 'trunk', 'weight')

    tensor = np.arange(n_in * n_out).reshape(n_in, n_out)
    flat = pack.flatten(port, tensor)

    assert port['flat_order']['tensor_axes'] == ['n_in', 'n_out']
    assert port['flat_order']['axes'] == ['n_out', 'n_in']
    assert flat == tensor.T.ravel().tolist()
    assert flat[:n_in] == tensor[:, 0].tolist(), 'first n_in scalars are output 0'


def test_write_mem_format(manifest, tmp_path):
    port = _port(manifest, 'trunk', 'weight')
    words = pack.pack_tensor(port, _bank_tensors(1)[0]['trunk'][0])

    out = tmp_path / 'bank.mem'
    pack.write_mem(out, words, port['expected_data_width'])
    lines = out.read_text().split()
    assert len(lines) == len(words)
    assert all(len(line) == port['expected_data_width'] // 4 for line in lines)
    assert int(lines[0], 16) == words[0]


def _bank_sets(n_banks, fixed=None):
    """Per-bank {(layer, role): tensor} covering every parameter of the model."""
    sets = []
    for tensors in _bank_tensors(n_banks):
        entry = {}
        for layer, (w, b) in tensors.items():
            entry[(layer, 'weight')] = w
            entry[(layer, 'bias')] = b
        if fixed is not None:
            entry.update(fixed)
        sets.append(entry)
    return sets


def test_pack_banks_packs_every_external_parameter(written_project, manifest):
    packed = pack.pack_banks(written_project, _bank_sets(2))

    assert set(packed) == {p['name'] for p in manifest['ports']}
    for port in manifest['ports']:
        entry = packed[port['name']]
        if port['expected_interface_kind'] == 'bram':
            assert len(entry['image']) == 2 * entry['bank_stride_words']
        else:
            assert [len(c) for c in entry['codes']] == [port['n_scalars']] * 2

    # and it agrees with the primitive it is built on
    weight = _port(manifest, 'trunk', 'weight')
    assert packed['w2']['image'][: weight['expected_depth']] == pack.pack_tensor(weight, _bank_tensors(2)[0]['trunk'][0])


def test_pack_banks_checks_scalar_bundle_shapes(written_project):
    """A bias goes through the same exact-shape check as a weight."""
    sets = _bank_sets(2)
    n_out = LAYERS['head_a'][1]
    for bank in sets:
        bank[('head_a', 'bias')] = np.zeros((1, n_out))  # manifest declares (n_out,)

    with pytest.raises(PackingUnsupported, match='shape'):
        pack.pack_banks(written_project, sets)


def test_pack_banks_requires_every_bank_to_supply_each_external_parameter(written_project):
    sets = _bank_sets(2)
    del sets[1][('head_a', 'weight')]

    with pytest.raises(PackingUnsupported, match='bank 1 is not a complete parameter set'):
        pack.pack_banks(written_project, sets)


def test_pack_banks_rejects_banks_that_disagree_on_a_fixed_parameter(partial_project):
    """Only BramFactor-externalized parameters can vary; the rest are in the IP."""
    sets = _bank_sets(2)
    for key in sets[0]:  # only the trunk kernel is above the threshold
        if key != ('trunk', 'weight'):
            sets[1][key] = sets[0][key]
    pack.pack_banks(partial_project, sets)  # banks differ only where they may

    # head_a's kernel is below the threshold, so it is compiled in and fixed
    sets[1][('head_a', 'weight')] = sets[0][('head_a', 'weight')] + 1.0
    with pytest.raises(PackingUnsupported, match='fixed in the compute IP'):
        pack.pack_banks(partial_project, sets)


def test_pack_banks_needs_the_complete_parameter_set(partial_project):
    """A fixed parameter that is absent cannot be told from one that agrees."""
    sets = _bank_sets(2)
    for bank in sets:
        bank.pop(('head_b', 'bias'))

    with pytest.raises(PackingUnsupported, match='bank 0 is not a complete parameter set'):
        pack.pack_banks(partial_project, sets)


def test_pack_banks_rejects_keys_that_are_not_parameters(written_project):
    sets = _bank_sets(2, fixed={('typo', 'weight'): np.zeros((2, 2))})

    with pytest.raises(PackingUnsupported, match='not parameters of this model'):
        pack.pack_banks(written_project, sets)


# --- real RTL ----------------------------------------------------------------


def _run_bank_simulation(synthesized_project, tmp_path, n_banks):
    """Package the shared IP for n_banks, load every bank, check every output."""
    import runtime_weights_sim as sim

    from hls4ml.contrib.runtime_weights import interface, pack_flat, pack_tensor, package
    from hls4ml.contrib.runtime_weights.package import fingerprint_ip

    if not sim.have_xsim():
        pytest.skip('requires xvlog/xelab/xsim')

    project_path, project, outputs = synthesized_project
    assert [size for _, size in outputs] == [LAYERS[h][1] for h in HEADS], 'output order is assumed below'

    # The IP is synthesized once with bank 0 baked in; the wrapper must override
    # whatever is baked in, so the shared project is reused as-is.
    before = fingerprint_ip(str(project_path), project)
    package(str(project_path), n_banks=n_banks)

    # take the verified ports, whose actual_* come from the synthesized interface,
    # rather than assuming actual == expected
    manifest = json.loads((project_path / 'firmware' / 'weights' / MANIFEST_FILENAME).read_text())
    verified, _ = interface.verify(manifest, str(project_path), project)
    bram = [p for p in verified if p['expected_interface_kind'] == 'bram']
    scalar = [p for p in verified if p['expected_interface_kind'] == 'scalar_bundle']
    assert len(bram) == len(scalar) == len(LAYERS), 'every layer should reach the wrapper'

    # the testbench models one output type; an auto-widened result would silently
    # truncate against a 16-bit wire rather than fail here
    rtl = {p['name']: p for p in interface.parse_rtl_ports(str(project_path), project)}
    for name, size in outputs:
        for o in range(size):
            assert rtl[f'{name}_{o}']['width'] == sim.WIDTH, f'{name}_{o} is not {sim.WIDTH} bits'

    x = [0.5, -1.25, 0.75, 2.0, -0.5, 1.5, -2.0, 0.25]
    payloads, expected = [], []
    for tensors in _bank_tensors(n_banks):
        payload = {p['name']: pack_tensor(p, tensors[p['layer']][0]) for p in bram}
        payload.update({p['name']: pack_flat(p, tensors[p['layer']][1].tolist()) for p in scalar})
        payloads.append(payload)
        heads = _reference(x, tensors)
        expected.append({name: [sim.code_of(v) for v in head] for (name, _), head in zip(outputs, heads)})

    for a in range(n_banks):
        for b in range(a + 1, n_banks):
            assert expected[a] != expected[b], f'banks {a} and {b} are not discriminating'

    tb = sim.write_testbench(
        str(tmp_path / 'tb_runtime_weights.sv'),
        PROJECT,
        [sim.code_of(v) for v in x],
        outputs,
        bram,
        scalar,
        payloads,
        expected,
    )
    passed, log = sim.run_xsim(
        str(tmp_path / 'xsim_work'),
        [
            str(project_path / 'runtime_weights' / 'rtl'),
            str(project_path / f'{project}_prj' / 'solution1' / 'syn' / 'verilog'),
        ],
        tb,
    )
    assert passed, f'{n_banks}-bank RTL simulation failed:\n{log[-6000:]}'
    # comparing before and after is what shows the IP was left alone
    assert fingerprint_ip(str(project_path), project)['combined'] == before['combined']


@pytest.mark.parametrize('n_banks', [2, 4])
def test_banks_rtl_simulation(synthesized_project, tmp_path, n_banks):
    """Same input to every bank: any output difference is bank selection.

    Four banks re-bank the same IP, so this also covers multi-bit bank ids and the
    wider bank stride without a second synthesis.
    """
    _run_bank_simulation(synthesized_project, tmp_path, n_banks)


# --- pointwise: a Dense over a 2-D or 3-D input ------------------------------

PW_POSITIONS, PW_CHAN, PW_FILT = 3, 8, 4


def _convert_pointwise(out_dir, weights, bias, rank):
    """A Dense over a 2-D/3-D input, which hls4ml lowers to a pointwise conv."""
    shape = (PW_POSITIONS, PW_CHAN) if rank == 2 else (2, PW_POSITIONS, PW_CHAN)
    inp = Input(shape=shape, name='input_1')
    model = Model(inp, Dense(PW_FILT, activation='linear', name='pw')(inp))
    model.get_layer('pw').set_weights([np.asarray(weights, np.float32), np.asarray(bias, np.float32)])
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity='name', backend='Vitis', default_precision=PRECISION, default_reuse_factor=2
    )
    cfg['Model']['Strategy'] = 'Resource'
    cfg['Model']['BramFactor'] = 0
    cfg['LayerName']['pw']['Strategy'] = 'Resource'
    for entry in cfg['LayerName'].values():
        for key, value in entry.get('Precision', {}).items():
            if value == 'auto':
                entry['Precision'][key] = PRECISION
    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(out_dir),
        project_name=PROJECT,
        backend='Vitis',
        io_type='io_parallel',
    )


def _pointwise_banks(n_banks):
    banks = []
    for k in range(n_banks):
        w = np.zeros((PW_CHAN, PW_FILT))
        for i in range(PW_CHAN):
            for o in range(PW_FILT):
                w[i, o] = (-1) ** (k + o) * (((i * PW_FILT + o + 13 * k) % 19) + 1) / 256.0
        b = np.array([(-1) ** (o + k) * (o + 1 + 2 * k) / 128.0 for o in range(PW_FILT)])
        banks.append((w, b))
    return banks


@pytest.mark.parametrize('rank', [2, 3])
def test_pointwise_two_banks_rtl_simulation(tmp_path, synthesis_config, rank):
    """The pointwise IP reads its weights through BOTH BRAM ports.

    The bank lends port B to the loader only while quiescent, so this is the test
    that the sharing works: same input to both banks, and any output difference is
    bank selection. Matching the reference also proves the unreshaped
    scalar-per-word layout the adapter claims is the real one.
    """
    import runtime_weights_sim as sim

    from hls4ml.contrib.runtime_weights import interface, pack_flat, pack_tensor, package

    if not synthesis_config['run_synthesis']:
        pytest.skip('set RUN_SYNTHESIS=true to run synthesis tests')
    if not sim.have_xsim():
        pytest.skip('requires xvlog/xelab/xsim')

    banks = _pointwise_banks(2)
    hls_model = _convert_pointwise(tmp_path, *banks[0], rank=rank)
    hls_model.write()
    hls_model.build(**synthesis_config['build_args']['Vitis'], log_to_stdout=False)
    outputs = [(v.name, v.size()) for v in hls_model.get_output_variables()]

    package(str(tmp_path), n_banks=2)
    man = json.loads((tmp_path / 'firmware' / 'weights' / MANIFEST_FILENAME).read_text())
    verified, _ = interface.verify(man, str(tmp_path), PROJECT)
    w_port = next(p for p in verified if p['role'] == 'weight')
    b_port = next(p for p in verified if p['role'] == 'bias')

    # the claim under test: one scalar per word, as deep as there are scalars
    assert w_port['kernel_variant'] == 'pointwise_unreshaped'
    assert w_port['actual_data_width'] == w_port['precision']['width']
    assert w_port['expected_depth'] == PW_CHAN * PW_FILT

    n_pos = PW_POSITIONS * (1 if rank == 2 else 2)
    x = [[(i + 1) / 8.0 - (p + 1) / 4.0 for i in range(PW_CHAN)] for p in range(n_pos)]
    payloads, expected = [], []
    for w, b in banks:
        tensor = w.reshape(*w_port['flat_order']['shape'])
        payloads.append({w_port['name']: pack_tensor(w_port, tensor), b_port['name']: pack_flat(b_port, b.tolist())})
        # a pointwise conv is the same dense applied at every position
        flat = []
        for pos in range(n_pos):
            flat += [sim.code_of(v) for v in sim.dense_reference(x[pos], w.tolist(), b.tolist())]
        expected.append({outputs[0][0]: flat})

    assert expected[0] != expected[1], 'banks must produce different outputs to be discriminating'

    tb = sim.write_testbench(
        str(tmp_path / 'tb_pointwise.sv'),
        PROJECT,
        [sim.code_of(v) for row in x for v in row],
        outputs,
        [w_port],
        [b_port],
        payloads,
        expected,
    )
    passed, log = sim.run_xsim(
        str(tmp_path / 'xsim_pw'),
        [
            str(tmp_path / 'runtime_weights' / 'rtl'),
            str(tmp_path / f'{PROJECT}_prj' / 'solution1' / 'syn' / 'verilog'),
        ],
        tb,
    )
    assert passed, f'pointwise ({rank}-D) two-bank RTL simulation failed:\n{log[-6000:]}'


def test_latch_rejects_out_of_range_bank(tmp_path, synthesis_config):
    """Needs three banks: two cannot express an invalid id."""
    import runtime_weights_sim as sim

    if not synthesis_config['run_synthesis']:
        pytest.skip('set RUN_SYNTHESIS=true to run synthesis tests')
    if not sim.have_xsim():
        pytest.skip('requires xvlog/xelab/xsim')

    rtl = Path(__file__).parent.parent.parent / 'hls4ml' / 'contrib' / 'runtime_weights' / 'templates'
    tb = sim.write_latch_testbench(str(tmp_path / 'tb_latch.sv'))
    passed, log = sim.run_xsim(str(tmp_path / 'xsim_latch'), [str(rtl)], tb)
    assert passed, f'latch bench failed:\n{log[-4000:]}'


def test_vivado_synthesizes_the_wrapper(synthesized_project):
    """The only check that exercises the generated Tcl and synthesizability.

    Packages IP from the hls4ml Vitis backend; Vivado is the implementation tool,
    not a supported HLS backend.
    """
    import runtime_weights_sim as sim

    from hls4ml.contrib.runtime_weights import package

    if not sim.have_vivado():
        pytest.skip('requires vivado')

    project_path, _, _ = synthesized_project
    summary = package(str(project_path), n_banks=2)

    rw = Path(project_path) / 'runtime_weights'
    # Generated ROM data is read by $readmemh from this directory at elaboration,
    # so it has to be packaged next to the Tcl.
    assert summary['rom_data_files'], 'this fixture should produce an out_index ROM'
    for name in summary['rom_data_files']:
        assert (rw / name).exists(), f'{name} was not packaged'

    ok, log = sim.run_vivado_batch('create_runtime_weights.tcl', rw, 'runtime-weights wrapper synthesized')
    assert ok, f'vivado synthesis did not complete:\n{log[-4000:]}'

    # Vivado deletes a data file that is added as a source; it must still be here
    for name in summary['rom_data_files']:
        assert (rw / name).exists(), f'{name} did not survive synthesis'
    assert 'is read successfully' in log, 'ROM data was not read during synthesis'
    for report in ('utilization.rpt', 'timing.rpt', 'drc.rpt'):
        assert (rw / report).exists(), f'missing {report}'

    # the clock must have reached synthesis, not just been written to a file
    timing = (rw / 'timing.rpt').read_text()
    assert 'ap_clk' in timing, 'clock constraint did not reach synthesis'


# --- fail-closed behaviour ---------------------------------------------------


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
    )


def test_unregistered_layer_cannot_be_packed(tmp_path):
    inp = Input(shape=(4, 4, 2), name='input_1')
    model = Model(inp, Conv2D(3, (2, 2), padding='valid', activation='linear', name='conv2d_1')(inp))
    hls_model = _convert_conv(model, tmp_path / 'conv')

    from hls4ml.writer.external_parameters import build_manifest

    for port in build_manifest(hls_model)['ports']:
        assert port['layout'] is None
        with pytest.raises(PackingUnsupported):
            pack.pack_flat(port, [0.0] * port['n_scalars'])


def test_unsupported_backend_claims_nothing(tmp_path):
    """The Vivado backend emits the same HLS source, but its RTL ABI is unverified.

    The adapter registry is keyed on backend, so this refuses at the manifest
    rather than reusing the Vitis layout.
    """
    from hls4ml.contrib.runtime_weights import package
    from hls4ml.writer.external_parameters import build_manifest

    inp = Input(shape=(N_IN,), name='input_1')
    model = Model(inp, Dense(4, activation='linear', name='dense_1')(inp))
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity='model', backend='Vivado', default_precision='ap_fixed<16,6>', default_reuse_factor=2
    )
    cfg['Model']['Strategy'] = 'Resource'
    cfg['Model']['BramFactor'] = 0
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(tmp_path / 'vivado'),
        project_name='vivado_prj',
        backend='Vivado',
        io_type='io_parallel',
    )
    hls_model.write()

    ports = build_manifest(hls_model)['ports']
    assert ports
    for port in ports:
        assert port['expected_interface_kind'] is None
        assert port['layout'] is None
        assert 'no adapter for' in port['note']

    with pytest.raises(ValueError, match='backend'):
        package(str(tmp_path / 'vivado'), n_banks=2)


def test_packer_refuses_unclaimed_ordering(manifest):
    port = dict(_port(manifest, 'trunk', 'weight'))
    port['layout'] = None

    with pytest.raises(PackingUnsupported):
        pack.pack_tensor(port, _bank_tensors(1)[0]['trunk'][0])


def test_flatten_rejects_wrong_shape(manifest):
    """A transposed tensor would silently mis-pack."""
    n_in, n_out, _ = LAYERS['trunk']
    port = _port(manifest, 'trunk', 'weight')

    for bad in (np.zeros((n_out, n_in + 1)), np.zeros((n_in, n_out, 2)), np.zeros((n_in, n_out + 1))):
        with pytest.raises(PackingUnsupported, match='shape'):
            pack.flatten(port, bad)


def test_pack_flat_rejects_wrong_scalar_count(manifest):
    for port in (_port(manifest, 'trunk', 'weight'), _port(manifest, 'trunk', 'bias')):
        n = port['n_scalars']
        for bad in ([0.0] * (n - 1), [0.0] * (n + 1)):
            with pytest.raises(PackingUnsupported, match='scalars'):
                pack.pack_flat(port, bad)


def test_build_bank_image_rejects_scalar_bundle(manifest):
    bias = _port(manifest, 'trunk', 'bias')
    n = bias['n_scalars']

    with pytest.raises(PackingUnsupported, match='bram'):
        pack.build_bank_image(bias, [[0.0] * n, [0.0] * n])


def test_unregistered_flattener_adapter_is_rejected(manifest):
    port = dict(_port(manifest, 'trunk', 'weight'))
    port['flat_order'] = dict(port['flat_order'], adapter='not_registered')

    with pytest.raises(PackingUnsupported, match='no custom flatteners'):
        pack.flatten(port, _bank_tensors(1)[0]['trunk'][0])


@pytest.mark.parametrize(
    'field,value,match',
    [
        ('backend', 'Vivado', 'backend'),
        ('project_name', '', 'does not name a project'),
        ('schema', 'someone.else/v1', 'schema'),
        ('schema_version', 99, 'version'),
    ],
)
def test_package_enforces_manifest_contract(tmp_path, field, value, match):
    from hls4ml.contrib.runtime_weights import package

    _manifest(tmp_path)
    path = tmp_path / 'firmware' / 'weights' / 'external_parameters.json'
    good = json.loads(path.read_text())
    path.write_text(json.dumps({**good, field: value}))

    with pytest.raises(ValueError, match=match):
        package(str(tmp_path), n_banks=2)


def test_bram_signal_names_come_from_the_rtl():
    """package.py consumes these names, so a renamed port must fail here."""
    from hls4ml.contrib.runtime_weights import interface

    ports = [{'name': f'w2_{s}', 'width': 8, 'dir': 'input'} for s in interface.BRAM_SIGNAL_SUFFIXES.values()]
    signals = interface.bram_signals(ports, 'w2')
    assert set(signals) == set(interface.BRAM_SIGNAL_SUFFIXES)
    assert signals['addr_a']['name'] == 'w2_Addr_A'

    renamed = [p for p in ports if p['name'] != 'w2_Addr_A']
    renamed.append({'name': 'w2_address0', 'width': 8, 'dir': 'input'})
    with pytest.raises(interface.InterfaceMismatch, match='naming has changed'):
        interface.bram_signals(renamed, 'w2')


def test_read_only_proof_rejects_a_live_write_enable(tmp_path):
    """The loader shares port B with the IP, so the IP must never write the memory.

    Whether the IP *reads* port B is deliberately not checked -- Dense leaves it
    idle and a pointwise convolution uses it, and both are supported.
    """
    from hls4ml.contrib.runtime_weights import interface

    syn = tmp_path / 'p_prj' / 'solution1' / 'syn' / 'verilog'
    syn.mkdir(parents=True)
    signals = {role: f'w2_{suffix}' for role, suffix in interface.BRAM_SIGNAL_SUFFIXES.items()}

    tied = "assign w2_WEN_A = 1'b0;\nassign w2_WEN_B = 1'b0;\n"

    # neither a live port-B read nor a driven Din disqualifies the memory: the write
    # enables are what make it read-only
    (syn / 'p.v').write_text(tied + 'assign w2_EN_B = kernel_EN_B;\nassign w2_Din_A = kernel_din;\n')
    read_only, evidence = interface.bram_is_read_only(str(tmp_path), 'p', signals)
    assert read_only, evidence

    for live in ('w2_WEN_A', 'w2_WEN_B'):
        (syn / 'p.v').write_text(tied.replace(f'assign {live} =', f'assign {live} = drive; //'))
        read_only, evidence = interface.bram_is_read_only(str(tmp_path), 'p', signals)
        assert not read_only, f'{live} driven but still called read-only: {evidence}'


def test_rtl_port_parsing_is_exhaustive(tmp_path):
    """Synthetic Verilog: this checks the parser, not the HLS flow."""
    from hls4ml.contrib.runtime_weights import interface

    syn = tmp_path / 'p_prj' / 'solution1' / 'syn' / 'verilog'
    syn.mkdir(parents=True)
    rtl = syn / 'p.v'

    rtl.write_text(
        'module p (ap_clk, ap_rst, ap_start, din, dout);\n'
        'input   ap_clk;\n'
        'input   ap_rst;\n'
        'input   ap_start;\n'
        'input  [15:0] din;\n'
        'output [7:0] dout;\n'
        'endmodule\n'
    )
    ports = interface.parse_rtl_ports(str(tmp_path), 'p')
    assert [p['name'] for p in ports] == ['ap_clk', 'ap_rst', 'ap_start', 'din', 'dout']
    assert next(p for p in ports if p['name'] == 'din')['width'] == 16
    assert next(p for p in ports if p['name'] == 'dout')['dir'] == 'output'

    # a header port with no declaration
    rtl.write_text('module p (ap_clk, missing);\ninput ap_clk;\nendmodule\n')
    with pytest.raises(ValueError, match='no declaration found'):
        interface.parse_rtl_ports(str(tmp_path), 'p')

    # inout is not supported by the wrapper
    rtl.write_text('module p (ap_clk, bidir);\ninput ap_clk;\ninout [3:0] bidir;\nendmodule\n')
    with pytest.raises(ValueError, match='inout'):
        interface.parse_rtl_ports(str(tmp_path), 'p')
