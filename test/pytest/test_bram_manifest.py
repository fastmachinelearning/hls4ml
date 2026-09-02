"""Unit tests for the external-BRAM parameter manifest.

C-level only: no HLS synthesis, so these run in ordinary CI.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Conv2D, Dense, Input
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.writer.bram_manifest import SCHEMA, SCHEMA_VERSION, build_bram_manifest

test_root_path = Path(__file__).parent
PART = 'xcvu9p-flga2104-2L-e'


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
        part=PART,
        clock_period=10.0,
    )


def test_no_manifest_without_bram_factor(tmp_path):
    """Default BramFactor exposes nothing externally, so no manifest is written."""
    hls_model = _convert(_dense_model(), tmp_path / 'nobram', bram_factor=1_000_000_000)
    hls_model.write()

    assert build_bram_manifest(hls_model)['ports'] == []
    assert not (tmp_path / 'nobram' / 'firmware' / 'weights' / 'bram_manifest.json').exists()


def test_manifest_written_and_wellformed(tmp_path):
    hls_model = _convert(_dense_model(), tmp_path / 'dense')
    hls_model.write()

    path = tmp_path / 'dense' / 'firmware' / 'weights' / 'bram_manifest.json'
    assert path.exists()
    manifest = json.loads(path.read_text())

    assert manifest['schema'] == SCHEMA
    assert manifest['schema_version'] == SCHEMA_VERSION
    assert manifest['io_type'] == 'io_parallel'
    assert manifest['bram_factor'] == 0
    assert 'disclaimer' in manifest
    assert {p['name'] for p in manifest['ports']} == {'w2', 'b2'}


@pytest.mark.parametrize(
    'reuse_factor,expected_width,expected_depth',
    [
        (2, 256, 2),  # dense_resource_rf_leq_nin
        (4, 128, 4),  # dense_resource_rf_leq_nin
        (8, 64, 8),  # dense_resource_rf_leq_nin, rf == n_in boundary
        (16, 32, 16),  # dense_resource_rf_gt_nin_rem0
        (32, 16, 32),  # dense_resource_rf_gt_nin_rem0, single lane
    ],
)
def test_weight_geometry_matches_reshape(tmp_path, reuse_factor, expected_width, expected_depth):
    """expected_data_width / expected_depth follow from the ARRAY_RESHAPE block factor."""
    hls_model = _convert(_dense_model(), tmp_path / f'rf{reuse_factor}', reuse_factor=reuse_factor)
    weight = next(p for p in build_bram_manifest(hls_model)['ports'] if p['role'] == 'weight')

    assert weight['expected_interface_kind'] == 'bram'
    assert weight['expected_data_width'] == expected_width
    assert weight['expected_depth'] == expected_depth
    # data width must be an exact multiple of the scalar width
    assert weight['expected_data_width'] % weight['precision']['width'] == 0
    # every scalar must be addressable
    lanes = weight['expected_data_width'] // weight['precision']['width']
    assert lanes * weight['expected_depth'] == weight['n_scalars']


def test_weight_packing_claimed_for_verified_kernels(tmp_path):
    hls_model = _convert(_dense_model(), tmp_path / 'packing', reuse_factor=2)
    weight = next(p for p in build_bram_manifest(hls_model)['ports'] if p['role'] == 'weight')

    assert weight['kernel_variant'] == 'dense_resource_rf_leq_nin'
    assert weight['logical_scalar_order'] == 'index(out,in) = out*n_in + in'
    assert weight['reshape']['lane_of_flat_index'] == 'f // 2'
    assert weight['reshape']['word_of_flat_index'] == 'f % 2'
    assert weight['reshape']['verified_for'] == ['dense_resource_rf_leq_nin']


def test_bias_is_scalar_bundle_regardless_of_size(tmp_path):
    """ARRAY_PARTITION variable=biases complete wins over the BRAM interface."""
    hls_model = _convert(_dense_model(n_out=32), tmp_path / 'bigbias')
    bias = next(p for p in build_bram_manifest(hls_model)['ports'] if p['role'] == 'bias')

    assert bias['expected_interface_kind'] == 'scalar_bundle'
    assert bias['expected_data_width'] == 16
    assert bias['expected_depth'] is None
    assert bias['n_scalars'] == 32


def test_conv2d_is_out_of_scope_and_claims_nothing(tmp_path):
    """The writer sees every BRAM weight, so unverified layers must stay silent."""
    inp = Input(shape=(4, 4, 2), name='input_1')
    model = Model(inp, Conv2D(3, (2, 2), padding='valid', activation='linear', name='conv2d_1')(inp))
    hls_model = _convert(model, tmp_path / 'conv2d')

    ports = build_bram_manifest(hls_model)['ports']
    assert ports, 'expected Conv2D weights to be exposed as external BRAM'
    for port in ports:
        assert port['layer_class'] == 'Conv2D'
        assert port['expected_interface_kind'] is None
        assert port['expected_data_width'] is None
        assert port['expected_depth'] is None
        assert port['logical_scalar_order'] is None
        assert port['kernel_variant'] is None
        assert 'outside schema v1 scope' in port['note']


def test_values_txt_omitted_when_weights_txt_disabled(tmp_path):
    hls_model = _convert(_dense_model(), tmp_path / 'notxt')
    hls_model.config.get_writer_config()['WriteWeightsTxt'] = False

    for port in build_bram_manifest(hls_model)['ports']:
        assert port['values_txt'] is None
