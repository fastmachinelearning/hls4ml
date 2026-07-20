"""End-to-end tests for the Bambu accelerator layer.

`test_build_bambu.py` covers the Bambu backend itself (HLS C++ -> Verilog).
This file covers what the accelerator layer adds on top: the float I/O
wrapper, the AXI slave RTL, the PLL block, and the `manifest.json` contract
that a place-and-route flow consumes.

The tests stop at `bitstream=False`, so they need Bambu but no vendor P&R
tool -- everything asserted here is produced by hls4ml itself. That is the
whole point of the manifest seam: the artefact is complete and checkable
before any vendor tool sees it.
"""

import json
from pathlib import Path

import pytest
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

# Deliberately NOT powers of two. Bambu rounds each BRAM address port up to
# the next power of two, so 3 -> depth 4 and 5 -> depth 8: the element count
# and the BRAM depth DIFFER. That gap is the entire point of these shapes.
# With, say, 2 -> 4 the two conventions coincide and the assertions below pass
# under either one -- which is how the geometry bug reached hardware in the
# first place. `test_shapes_discriminate_the_conventions` locks this in.
N_IN = 3
N_OUT = 5


@pytest.fixture(scope='module')
def simple_model():
    model = Sequential()
    model.add(Dense(N_OUT, input_shape=(N_IN,)))
    return model


@pytest.fixture(scope='module')
def built_projects(simple_model):
    """Build once per flow; the tests below read the artefacts.

    Module-scoped because a Bambu synth run is minutes, not seconds, and
    every assertion here is a read-only look at the result.
    """
    projects = {}
    for io_type in ('io_parallel', 'io_stream'):
        output_dir = test_root_path / f'hls4mlprj_bambu_accel_{io_type}'
        config = hls4ml.utils.config_from_keras_model(simple_model, granularity='name')
        hls_model = hls4ml.converters.convert_from_keras_model(
            simple_model,
            hls_config=config,
            output_dir=str(output_dir),
            io_type=io_type,
            backend='NanoXploreAccelerator',
        )
        hls_model.build(csim=False, synth=True, bitstream=False)
        projects[io_type] = (hls_model, output_dir)
    return projects


def _manifest(output_dir):
    return json.loads((Path(output_dir) / 'manifest.json').read_text())


def test_shapes_discriminate_the_conventions(built_projects):
    """Guard the guard: if someone rounds these shapes to powers of two, the
    BRAM depth equals the element count and every N_WORDS assertion in this
    file silently stops testing anything."""
    _, output_dir = built_projects['io_parallel']
    m = _manifest(output_dir)
    assert m['bram_slots'] != m['n_words'], (
        'model shapes no longer distinguish BRAM depth from element count -- pick non-power-of-two layer sizes'
    )


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_manifest_contract(built_projects, io_type):
    """The manifest is the whole public/private interface -- check its shape."""
    _, output_dir = built_projects[io_type]
    m = _manifest(output_dir)

    assert m['manifest_version'] == 1
    assert m['top_module'] == 'myproject'
    assert m['flow'] == ('stream' if io_type == 'io_stream' else 'parallel')
    assert m['n_words'] == {'in': N_IN, 'out': N_OUT}
    assert m['clock_mhz'] == pytest.approx(1000.0 / m['clock_period_ns'])

    # bram_slots is the parallel-only BRAM depth and must be a power of two
    # (it is 2 ** address_port_width); stream has no depth at all.
    if m['flow'] == 'parallel':
        for key in ('in', 'out'):
            depth = m['bram_slots'][key]
            assert depth >= m['n_words'][key]
            assert depth & (depth - 1) == 0, f'{key} depth {depth} is not a power of two'
    else:
        assert m['bram_slots'] is None

    # Container widths are >= the ap_fixed value widths they carry.
    for key in ('in', 'out'):
        assert m['data_widths'][key] >= m['fixed_point'][key]['total']


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_rtl_file_list_is_complete(built_projects, io_type):
    """`rtl_files` is what the P&R side blindly adds. A missing entry there
    surfaces as a vendor elaboration error with no obvious cause, so assert
    every listed file actually exists next to the manifest."""
    _, output_dir = built_projects[io_type]
    m = _manifest(output_dir)

    assert m['rtl_files'], 'manifest lists no RTL files'
    missing = [f for f in m['rtl_files'] if not (Path(output_dir) / f).exists()]
    assert not missing, f'manifest lists files that were not written: {missing}'

    # The wrapper module the manifest names as top must be in the HLS output.
    hls_top_v = Path(output_dir) / f'{m["hls_top"]}.v'
    assert hls_top_v.exists()
    assert 'module myproject' in hls_top_v.read_text()


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_top_localparams_match_the_manifest(built_projects, io_type):
    """Regression test for the bsp_rc=4 DMA hang.

    The top file used to ship the reference project's hardcoded geometry, so
    the AXI slave advertised fewer read beats than the firmware requested and
    the burst hung on hardware. The generated localparams must agree with the
    manifest -- and note HLS_*_N_WORDS means different things per flow: the
    BRAM depth on parallel, the element count on stream.
    """
    _, output_dir = built_projects[io_type]
    m = _manifest(output_dir)
    top = (Path(output_dir) / f'top_{m["flow"]}.v').read_text()

    expected = m['n_words'] if m['flow'] == 'stream' else m['bram_slots']
    assert f'localparam HLS_IN_N_WORDS  = {expected["in"]};' in top
    assert f'localparam HLS_OUT_N_WORDS = {expected["out"]};' in top
    assert f'localparam HLS_IN_DATA_W   = {m["data_widths"]["in"]};' in top
    assert f'localparam HLS_OUT_DATA_W  = {m["data_widths"]["out"]};' in top

    # The marker block must survive patching, or the next build cannot patch.
    assert top.count('// HLS4ML PARAMS BEGIN') == 1
    assert top.count('// HLS4ML PARAMS END') == 1
    assert top.count('// HLS4ML PLL BEGIN') == 1
    assert top.count('// HLS4ML PLL END') == 1
