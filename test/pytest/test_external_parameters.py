import json

import numpy as np
import pytest
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Input
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.writer.external_parameters import MANIFEST_FILENAME, SCHEMA, SCHEMA_VERSION, build_manifest


def _dense_model(n_in=8, n_out=4):
    inp = Input(shape=(n_in,), name='input_1')
    model = Model(inp, Dense(n_out, activation='linear', name='dense_1')(inp))
    w = (np.arange(n_in * n_out, dtype=np.float32).reshape(n_in, n_out) + 1) / 1024.0
    b = (np.arange(n_out, dtype=np.float32) + 1) / 1024.0
    model.get_layer('dense_1').set_weights([w, b])
    return model


def _convert(model, out_dir, reuse_factor=2, bram_factor=0, strategy='Resource', io_type='io_parallel'):
    cfg = hls4ml.utils.config_from_keras_model(
        model,
        granularity='model',
        backend='Vitis',
        default_precision='ap_fixed<16,6>',
        default_reuse_factor=reuse_factor,
    )
    cfg['Model']['Strategy'] = strategy
    cfg['Model']['BramFactor'] = bram_factor
    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(out_dir),
        project_name='manifest_prj',
        backend='Vitis',
        io_type=io_type,
    )


def test_no_manifest_without_bram_factor(tmp_path):
    hls_model = _convert(_dense_model(), tmp_path / 'nobram', bram_factor=1_000_000_000)
    hls_model.write()

    assert build_manifest(hls_model)['ports'] == []
    assert not (tmp_path / 'nobram' / 'firmware' / 'weights' / MANIFEST_FILENAME).exists()


def test_manifest_describes_the_model(tmp_path):
    hls_model = _convert(_dense_model(), tmp_path / 'dense')
    hls_model.write()

    path = tmp_path / 'dense' / 'firmware' / 'weights' / MANIFEST_FILENAME
    assert path.exists()
    manifest = json.loads(path.read_text())

    assert manifest['schema'] == SCHEMA
    assert manifest['schema_version'] == SCHEMA_VERSION
    assert manifest['io_type'] == 'io_parallel'
    assert manifest['bram_factor'] == 0
    assert 'disclaimer' in manifest
    assert {p['name'] for p in manifest['ports']} == {'w2', 'b2'}

    weight = next(p for p in manifest['ports'] if p['role'] == 'weight')
    assert weight['kernel_variant'] == 'dense_resource_rf_leq_nin'
    assert weight['expected_interface_kind'] == 'bram'
    assert weight['flat_order']['tensor_axes'] == ['n_in', 'n_out']
    assert weight['flat_order']['axes'] == ['n_out', 'n_in']
    assert weight['layout'] == {'mode': 'block', 'block_size': 2, 'lanes': 16}
    assert 'description' not in weight['layout']


@pytest.mark.parametrize(
    'reuse_factor,expected_width,expected_depth',
    [
        (2, 256, 2),
        (4, 128, 4),
        (8, 64, 8),  # rf == n_in boundary
        (16, 32, 16),  # rf_gt_nin_rem0 from here
        (32, 16, 32),  # single lane
    ],
)
def test_weight_geometry_matches_reshape(tmp_path, reuse_factor, expected_width, expected_depth):
    hls_model = _convert(_dense_model(), tmp_path / f'rf{reuse_factor}', reuse_factor=reuse_factor)
    weight = next(p for p in build_manifest(hls_model)['ports'] if p['role'] == 'weight')

    assert weight['expected_interface_kind'] == 'bram'
    assert weight['expected_data_width'] == expected_width
    assert weight['expected_depth'] == expected_depth
    assert weight['expected_data_width'] % weight['precision']['width'] == 0
    lanes = weight['expected_data_width'] // weight['precision']['width']
    assert lanes * weight['expected_depth'] == weight['n_scalars']


def test_bias_is_scalar_bundle_regardless_of_size(tmp_path):
    hls_model = _convert(_dense_model(n_out=32), tmp_path / 'bigbias')
    bias = next(p for p in build_manifest(hls_model)['ports'] if p['role'] == 'bias')

    assert bias['expected_interface_kind'] == 'scalar_bundle'
    assert bias['expected_data_width'] == 16
    assert bias['expected_depth'] is None
    assert bias['n_scalars'] == 32


def test_conv2d_is_out_of_scope_and_claims_nothing(tmp_path):
    inp = Input(shape=(4, 4, 2), name='input_1')
    model = Model(inp, Conv2D(3, (2, 2), padding='valid', activation='linear', name='conv2d_1')(inp))
    hls_model = _convert(model, tmp_path / 'conv2d')

    ports = build_manifest(hls_model)['ports']
    assert ports
    for port in ports:
        assert port['layer_class'] == 'Conv2D'
        assert port['expected_interface_kind'] is None
        assert port['expected_data_width'] is None
        assert port['expected_depth'] is None
        assert port['flat_order'] is None
        assert port['layout'] is None
        assert port['kernel_variant'] is None
        assert 'no adapter for' in port['note']


def test_values_txt_omitted_when_weights_txt_disabled(tmp_path):
    hls_model = _convert(_dense_model(), tmp_path / 'notxt')
    hls_model.config.get_writer_config()['WriteWeightsTxt'] = False

    for port in build_manifest(hls_model)['ports']:
        assert port['values_txt'] is None


def test_manifest_refuses_layout_for_unencodable_precision():
    from hls4ml.writer.external_parameters import _unsupported_precision_reason

    ok = {'type': 'x', 'width': 16, 'integer': 6, 'rounding_mode': 'TRN', 'saturation_mode': 'WRAP'}
    assert _unsupported_precision_reason(ok) is None
    assert _unsupported_precision_reason({**ok, 'rounding_mode': 'RND'})
    assert _unsupported_precision_reason({**ok, 'saturation_mode': 'SAT'})
    assert _unsupported_precision_reason({'type': 'int<8>', 'width': None, 'integer': None})


@pytest.mark.parametrize(
    'shape,expected_class,axes',
    [
        ((8,), 'Dense', ['n_out', 'n_in']),
        ((4, 8), 'PointwiseConv1D', ['n_filt', 'filt_width', 'n_chan']),
        ((2, 4, 8), 'PointwiseConv2D', ['n_filt', 'filt_height', 'filt_width', 'n_chan']),
    ],
)
def test_dense_over_any_input_rank_is_described(tmp_path, shape, expected_class, axes):
    """hls4ml rewrites a Dense over 2-D/3-D input into a pointwise convolution.

    Those have their own adapter: unlike a Dense the kernel is not reshaped, so the
    port is one scalar wide and as deep as there are scalars.
    """
    n_out = 6
    inp = Input(shape=shape, name='input_1')
    model = Model(inp, Dense(n_out, activation='linear', name='d')(inp))
    hls_model = _convert(model, tmp_path / f'nd{len(shape)}')

    ports = build_manifest(hls_model)['ports']
    assert {p['layer_class'] for p in ports} == {expected_class}

    weight = next(p for p in ports if p['role'] == 'weight')
    bias = next(p for p in ports if p['role'] == 'bias')
    assert weight['expected_interface_kind'] == 'bram'
    assert bias['expected_interface_kind'] == 'scalar_bundle'
    assert weight['flat_order']['axes'] == axes

    if expected_class == 'Dense':
        assert weight['kernel_variant'] == 'dense_resource_rf_leq_nin'
        assert weight['layout']['lanes'] > 1  # reshaped into wide words
    else:
        assert weight['kernel_variant'] == 'pointwise_unreshaped'
        assert weight['layout']['lanes'] == 1  # one scalar per word
        assert weight['expected_data_width'] == weight['precision']['width']
        assert weight['expected_depth'] == weight['n_scalars'] == 8 * n_out


def test_a_wider_kernel_is_not_pointwise(tmp_path):
    """The pointwise adapter must not capture a genuine convolution."""
    inp = Input(shape=(6, 8), name='input_1')
    model = Model(inp, Conv1D(6, 3, activation='linear', name='c')(inp))
    hls_model = _convert(model, tmp_path / 'conv1d_k3')

    ports = build_manifest(hls_model)['ports']
    assert {p['layer_class'] for p in ports} == {'Conv1D'}
    for port in ports:
        assert port['expected_interface_kind'] is None
        assert 'no adapter for' in port['note']


def test_registry_covers_only_verified_combinations():
    from hls4ml.writer.external_parameters import described_combinations

    keys = set(described_combinations())
    assert ('Vitis', 'io_parallel', 'resource', 'Dense', 'weight') in keys
    assert ('Vitis', 'io_parallel', 'resource', 'Dense', 'bias') in keys

    assert ('Vitis', 'io_parallel', 'resource', 'PointwiseConv1D', 'weight') in keys
    assert ('Vitis', 'io_parallel', 'resource', 'PointwiseConv2D', 'weight') in keys

    for backend, io_type, strategy, layer_class, _role in keys:
        assert backend == 'Vitis'
        assert io_type == 'io_parallel'
        assert strategy == 'resource'
        assert layer_class in ('Dense', 'PointwiseConv1D', 'PointwiseConv2D')
