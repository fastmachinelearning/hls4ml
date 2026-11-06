import os
import re
import shutil
import subprocess

import numpy as np
import pytest
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.contrib.parameter_banks import InterfaceUnsupported, PackingUnsupported, pack_banks, package
from hls4ml.model.external_parameters import ExternalParameterManifest

PRECISION = 'ap_fixed<16,6>'
FRACTION = 10


def _model(shape, n_out, seed=0, hidden=None):
    """Dense layers with weights on the fixed-point grid, so that every conversion of them is exact."""
    inp = Input(shape=shape, name='input_1')
    x = Dense(hidden, activation='relu', name='d1')(inp) if hidden else inp
    model = Model(inp, Dense(n_out, name='d')(x))
    rng = np.random.default_rng(seed)
    for layer in model.layers:
        if layer.get_weights():
            layer.set_weights([rng.integers(-128, 128, w.shape) / 256 for w in layer.get_weights()])
    return model


def _convert(model, out_dir, external=('weight', 'bias'), strategy='Resource', reuse_factor=2, io_type='io_parallel', **cfg):
    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', backend='Vitis', default_precision=PRECISION, default_reuse_factor=reuse_factor
    )
    config['Model']['Strategy'] = strategy
    config['Model'].update(cfg.get('model_config', {}))
    weighted = {layer.name for layer in model.layers if layer.get_weights()}
    for name, layer in config['LayerName'].items():
        for key, value in layer.get('Precision', {}).items():
            if value == 'auto':
                layer['Precision'][key] = PRECISION
        if name in weighted:
            layer['Strategy'] = strategy
            layer['ReuseFactor'] = cfg.get('reuse_factors', {}).get(name, reuse_factor)
            if external:
                layer['ExternalParameters'] = list(external)
    for name, entry in cfg.get('layer_config', {}).items():
        config['LayerName'][name].update(entry)
    return hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=str(out_dir), backend='Vitis', io_type=io_type
    )


def _params(model):
    return {
        (layer.name, role): value for layer in model.layers for role, value in zip(('weight', 'bias'), layer.get_weights())
    }


@pytest.mark.parametrize(
    'shape, reuse_factor, layer_class, lanes, depth',
    [
        ((8,), 2, 'Dense', 16, 2),  # dense_resource_rf_leq_nin
        ((8,), 16, 'Dense', 2, 16),  # dense_resource_rf_gt_nin_rem0
        ((5, 8), 2, 'PointwiseConv1D', 1, 32),
        ((2, 5, 8), 2, 'PointwiseConv2D', 1, 32),
    ],
)
def test_layouts(tmp_path, shape, reuse_factor, layer_class, lanes, depth):
    hls_model = _convert(_model(shape, 4), tmp_path, reuse_factor=reuse_factor)
    weight, bias = sorted(ExternalParameterManifest.from_model(hls_model), key=lambda p: p.role, reverse=True)

    geometry = (weight.layer_class, weight.interface_kind, weight.layout.lanes, weight.depth)
    assert geometry == (layer_class, 'bram', lanes, depth)
    assert (bias.interface_kind, bias.n_scalars) == ('scalar_bundle', 4)

    # without STABLE the dataflow region synchronizes on the parameters and inferences cannot overlap
    hls_model.write()
    top = (tmp_path / 'firmware' / 'myproject.cpp').read_text()
    assert f'#pragma HLS STABLE variable={weight.name}' in top and f'#pragma HLS STABLE variable={bias.name}' in top


@pytest.mark.parametrize(
    'strategy, reuse_factor, io_type, reason',
    [
        ('Latency', 1, 'io_parallel', 'no verified layout'),
        ('Resource', 1, 'io_parallel', 'one word'),
        ('Resource', 2, 'io_stream', 'no verified layout'),
    ],
)
def test_unbankable_parameters_warn_at_conversion(tmp_path, strategy, reuse_factor, io_type, reason):
    with pytest.warns(UserWarning, match=f'weight cannot be placed in parameter banks: .*{reason}'):
        hls_model = _convert(_model((8,), 4), tmp_path, strategy=strategy, reuse_factor=reuse_factor, io_type=io_type)
    hls_model.write()
    with pytest.raises(InterfaceUnsupported):
        package(hls_model)


@pytest.mark.parametrize(
    'cfg, error',
    [
        ({'layer_config': {'d': {'ExternalParameters': ['weight', 'gamma']}}}, "'gamma'"),
        ({'model_config': {'ExternalParameters': ['weight']}}, "not under 'Model'"),
        ({'layer_config': {'d': {'ExternalParameters': 'weight'}}}, 'external_parameters'),
    ],
)
def test_invalid_selection_fails_at_conversion(tmp_path, cfg, error):
    with pytest.raises(Exception, match=error):
        _convert(_model((8,), 4), tmp_path, external=None, **cfg)


def test_pack_banks(tmp_path):
    model = _model((8,), 4)
    hls_model = _convert(model, tmp_path, external=['weight'])
    hls_model.write()
    bank = _params(model)
    weight = bank[('d', 'weight')]

    (image,) = pack_banks(hls_model, [bank, {**bank, ('d', 'weight'): -weight}]).values()
    parameter = image.parameter
    assert len(image.image) == 2 * image.bank_stride_words
    assert image.image[: parameter.depth] == parameter.pack(weight)
    assert image.image[image.bank_stride_words : image.bank_stride_words + parameter.depth] == parameter.pack(-weight)

    with pytest.raises(PackingUnsupported, match='compiled into the compute IP'):
        pack_banks(hls_model, [bank, {**bank, ('d', 'bias'): bank[('d', 'bias')] + 1}])
    with pytest.raises(PackingUnsupported, match='not a complete parameter set'):
        pack_banks(hls_model, [{('d', 'weight'): weight}])


# --- RTL ----------------------------------------------------------------------

CASES = {
    # two Dense kernels, four banks in LUTRAM: two preloaded, one loaded at run time, one left empty
    'dense': dict(shape=(8,), n_out=4, hidden=6, reuse_factors={'d1': 4, 'd': 12}, n_banks=4, preload=2, memory='auto'),
    # a Dense over 2-D input is a pointwise convolution, which also reads port B; one bank in block RAM
    'pointwise': dict(shape=(3, 8), n_out=4, hidden=None, reuse_factors={}, n_banks=1, preload=1, memory='block'),
}
_built = {}


@pytest.fixture
def banked(request, tmp_path_factory, synthesis_config):
    """A synthesized and packaged design per case, shared by the tests that need RTL."""
    if not synthesis_config['run_synthesis']:
        pytest.skip('set RUN_SYNTHESIS=true to run synthesis tests')
    kind = request.param
    if kind not in _built:
        case = CASES[kind]
        models = [_model(case['shape'], case['n_out'], seed, case['hidden']) for seed in range(3)]
        out_dir = tmp_path_factory.mktemp(f'banked_{kind}')
        hls_model = _convert(models[0], out_dir, reuse_factors=case['reuse_factors'])
        hls_model.write()
        hls_model.build(**synthesis_config['build_args']['Vitis'], log_to_stdout=False)
        banks = [_params(m) for m in models[: case['preload']]]
        summary = package(hls_model, n_banks=case['n_banks'], banks=banks, memory=case['memory'])
        _built[kind] = (case, models, hls_model, summary, out_dir)
    return _built[kind]


def _simulate(out_dir, summary, requests, loads, work):
    """Run requests (bank, input codes) back to back through the wrapper after the loader writes
    (param id, bank, address, word); return every output port's values in order."""
    rtl = out_dir / 'parameter_banks' / 'rtl'
    header = (rtl / f'{summary["top"]}.sv').read_text()
    pattern = r'^\s*(input|output)\s+wire\s+(?:\[(\d+):0\]\s+)?(\w+)'
    ports = [(d, int(w) + 1 if w else 1, n) for d, w, n in re.findall(pattern, header, re.M)]
    names = {n for _, _, n in ports}
    outputs = [n for d, _, n in ports if d == 'output' and f'{n}_ap_vld' in names]
    width = {n: w for _, w, n in ports}

    tb = ['`timescale 1 ns / 1 ps', 'module tb;', '  reg ap_clk = 0, ap_rst = 1;', '  always #5 ap_clk = ~ap_clk;']
    tb += [f'  {"reg" if d == "input" else "wire"} [{w - 1}:0] {n};' for d, w, n in ports if n not in ('ap_clk', 'ap_rst')]
    tb += ['  integer f, errors = 0;', f'  {summary["top"]} dut ({", ".join(f".{n}({n})" for _, _, n in ports)});']
    tb += [f'  always @(posedge ap_clk) if ({n}_ap_vld) $fdisplay(f, "{n} %0d", {n});' for n in outputs]
    tb += ['  initial begin', '    f = $fopen("out.log", "w");', '    {ap_start, input_1_ap_vld, ld_req} = 0;']
    tb += ['    repeat (5) @(negedge ap_clk);', '    ap_rst = 0;']
    for pid, bank, addr, word in loads:
        tb += [
            '    @(negedge ap_clk);',
            f'    ld_req = 1; ld_param_id = {pid}; ld_bank = {bank}; ld_addr = {addr};',
            f"    ld_data = {width['ld_data']}'h{word:x};",
            '    @(negedge ap_clk);',
            '    if (ld_accept !== 1) errors = errors + 1;',
            '    ld_req = 0;',
        ]
    for i, (bank, codes) in enumerate(requests):
        packed = ''.join(f'{c:04x}' for c in reversed(codes))
        tb += ['    @(negedge ap_clk);', f"    input_1 = {width['input_1']}'h{packed}; input_1_ap_vld = 1; ap_start = 1;"]
        tb += [f'    bank_id = {bank};'] if 'bank_id' in names else []
        if i == 0 and loads:  # a write while a request is presented could change the bank in use
            tb += ['    ld_req = 1; ld_addr = 0;', "    #1 if (ld_accept !== 1'b0) errors = errors + 1;", '    ld_req = 0;']
        tb += ['    @(posedge ap_clk);', "    while (ap_ready !== 1'b1) @(posedge ap_clk);"]
    tb += ['    @(negedge ap_clk) ap_start = 0;', '    repeat (1000) @(posedge ap_clk);', '    $fclose(f);']
    tb += ['    $display("TB_DONE errors=%0d", errors);', '    $finish;', '  end', 'endmodule']

    os.makedirs(work)
    (work / 'tb.sv').write_text('\n'.join(tb) + '\n')
    hls_rtl = out_dir / 'myproject_prj' / 'solution1' / 'syn' / 'verilog'
    for data in [*rtl.glob('*.mem'), *hls_rtl.glob('*.dat')]:
        shutil.copy(data, work)
    for cmd in (
        ['xvlog', *map(str, hls_rtl.glob('*.v'))],
        ['xvlog', '-sv', *map(str, rtl.glob('*.sv')), 'tb.sv'],
        ['xelab', 'tb', '-s', 'tb_sim'],
        ['xsim', 'tb_sim', '-R'],
    ):
        result = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert 'TB_DONE errors=0' in result.stdout, result.stdout[-3000:]

    values = {n: [] for n in outputs}
    for line in (work / 'out.log').read_text().splitlines():
        name, value = line.split()
        values[name].append(int(value))
    return values


@pytest.mark.parametrize('banked', list(CASES), indirect=True)
def test_banks_match_predict(banked, tmp_path):
    """Every output of the banked RTL equals predict() of the model whose parameters its bank holds."""
    if not shutil.which('xvlog'):
        pytest.skip('requires xsim')
    case, models, hls_model, summary, out_dir = banked

    rng = np.random.default_rng(1)
    banks = [0, 0, 1, 1, 2, 0] if case['n_banks'] > 1 else [0, 0, 0]
    xs = rng.integers(-64, 64, (len(banks), *case['shape'])) / 64
    requests = [(bank, [int(c) & 0xFFFF for c in (x.ravel() * 2**FRACTION).astype(int)]) for bank, x in zip(banks, xs)]

    loads = []
    if case['n_banks'] > 2:  # bank 2 is not preloaded: write it through the loader
        param_ids = {p['name']: p['param_id'] for p in summary['banked_ports']}
        for name, image in pack_banks(hls_model, [_params(models[2])]).items():
            loads += [(param_ids[name], 2, addr, word) for addr, word in enumerate(image.per_bank[0])]
    values = _simulate(out_dir, summary, requests, loads, tmp_path / 'xsim')

    for bank in sorted(set(banks)):
        reference = _convert(models[bank], tmp_path / f'reference_{bank}', reuse_factors=case['reuse_factors'])
        reference.compile()
        expected = reference.predict(np.ascontiguousarray(xs)).reshape(len(xs), -1)
        for name, got in values.items():
            index = int(name.rsplit('_', 1)[1])
            for i, b in enumerate(banks):
                if b == bank:
                    assert got[i] == int(round(expected[i, index] * 2**FRACTION)) & 0xFFFF, (name, i, bank)
