import json

import numpy as np
import pytest
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Input
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.model.external_parameters import MANIFEST_FILENAME, SCHEMA, SCHEMA_VERSION, ExternalParameterManifest
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType
from hls4ml.writer.external_parameters import build_manifest


def _ports(hls_model):
    """The manifest's port entries as written to JSON: these tests check the format."""
    return build_manifest(hls_model).serialize_state()['ports']


def _dense_model(n_in=8, n_out=4):
    inp = Input(shape=(n_in,), name='input_1')
    model = Model(inp, Dense(n_out, activation='linear', name='dense_1')(inp))
    w = (np.arange(n_in * n_out, dtype=np.float32).reshape(n_in, n_out) + 1) / 1024.0
    b = (np.arange(n_out, dtype=np.float32) + 1) / 1024.0
    model.get_layer('dense_1').set_weights([w, b])
    return model


def _convert(model, out_dir, reuse_factor=2, external=True, strategy='Resource', io_type='io_parallel'):
    cfg = hls4ml.utils.config_from_keras_model(
        model,
        granularity='model',
        backend='Vitis',
        default_precision='ap_fixed<16,6>',
        default_reuse_factor=reuse_factor,
    )
    cfg['Model']['Strategy'] = strategy
    if external:
        cfg['LayerName'] = {
            layer.name: {'ExternalParameters': ['weight', 'bias']} for layer in model.layers if layer.get_weights()
        }
    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(out_dir),
        project_name='manifest_prj',
        backend='Vitis',
        io_type=io_type,
    )


def test_no_manifest_without_external_parameters(tmp_path):
    hls_model = _convert(_dense_model(), tmp_path / 'nobram', external=False)
    hls_model.write()

    assert build_manifest(hls_model).parameters == []
    assert not (tmp_path / 'nobram' / 'firmware' / 'weights' / MANIFEST_FILENAME).exists()


def _convert_with(model, out_dir, layer_config=None, model_config=None, type_config=None):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis', default_reuse_factor=2)
    cfg['Model']['Strategy'] = 'Resource'
    cfg['Model'].update(model_config or {})
    if layer_config:
        cfg['LayerName'] = layer_config
    if type_config:
        cfg['LayerType'] = type_config
    return hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=str(out_dir), backend='Vitis', io_type='io_parallel'
    )


def test_external_parameters_selects_roles_explicitly(tmp_path):
    hls_model = _convert_with(_dense_model(), tmp_path / 'weight_only', {'dense_1': {'ExternalParameters': ['weight']}})
    assert [p.role for p in build_manifest(hls_model)] == ['weight']


def test_external_parameters_rejects_unknown_role(tmp_path):
    with pytest.raises(ValueError, match="'gamma'"):
        _convert_with(_dense_model(), tmp_path / 'bad', {'dense_1': {'ExternalParameters': ['weight', 'gamma']}})


def test_legacy_bram_factor_threshold_still_exposes_parameters(tmp_path):
    """BramFactor is kept for compatibility: above-threshold weights are exposed
    and described exactly as explicitly selected ones."""
    hls_model = _convert_with(_dense_model(), tmp_path / 'legacy', model_config={'BramFactor': 0})
    assert {p.name for p in build_manifest(hls_model)} == {'w2', 'b2'}


def test_manifest_describes_the_model(tmp_path):
    hls_model = _convert(_dense_model(), tmp_path / 'dense')
    hls_model.write()

    path = tmp_path / 'dense' / 'firmware' / 'weights' / MANIFEST_FILENAME
    assert path.exists()
    manifest = json.loads(path.read_text())

    assert manifest['schema'] == SCHEMA
    assert manifest['schema_version'] == SCHEMA_VERSION
    assert manifest['io_type'] == 'io_parallel'
    assert {p['name'] for p in manifest['ports']} == {'w2', 'b2'}

    weight = next(p for p in manifest['ports'] if p['role'] == 'weight')
    assert weight['kernel_variant'] == 'dense_resource_rf_leq_nin'
    assert weight['expected_interface_kind'] == 'bram'
    assert weight['flat_order']['tensor_axes'] == ['n_in', 'n_out']
    assert weight['flat_order']['axes'] == ['n_out', 'n_in']
    assert weight['layout'] == {'mode': 'block', 'block_size': 2, 'lanes': 16}
    assert weight['precision']['state']['width'] == 16

    # and it round-trips through the object model unchanged
    loaded = ExternalParameterManifest.load(str(tmp_path / 'dense'))
    assert loaded.serialize_state() == manifest
    assert loaded.get('dense_1', 'weight').layout.block_size == 2


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
    weight = next(p for p in _ports(hls_model) if p['role'] == 'weight')

    assert weight['expected_interface_kind'] == 'bram'
    assert weight['expected_data_width'] == expected_width
    assert weight['expected_depth'] == expected_depth
    assert weight['expected_data_width'] % weight['precision']['state']['width'] == 0
    lanes = weight['expected_data_width'] // weight['precision']['state']['width']
    assert lanes * weight['expected_depth'] == weight['n_scalars']


def test_bias_is_scalar_bundle_regardless_of_size(tmp_path):
    hls_model = _convert(_dense_model(n_out=32), tmp_path / 'bigbias')
    bias = next(p for p in _ports(hls_model) if p['role'] == 'bias')

    assert bias['expected_interface_kind'] == 'scalar_bundle'
    assert bias['expected_data_width'] == 16
    assert bias['expected_depth'] is None
    assert bias['n_scalars'] == 32


def test_conv2d_is_out_of_scope_and_claims_nothing(tmp_path):
    inp = Input(shape=(4, 4, 2), name='input_1')
    model = Model(inp, Conv2D(3, (2, 2), padding='valid', activation='linear', name='conv2d_1')(inp))
    hls_model = _convert(model, tmp_path / 'conv2d')

    ports = _ports(hls_model)
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


def test_manifest_round_trips_any_precision_type(tmp_path):
    """The manifest describes a parameter it cannot pack (so the refusal is clean
    later), and rebuilds the precision class it wrote rather than assuming one."""
    from hls4ml.model.external_parameters import ExternalParameter, ExternalParameterError

    rnd = ExternalParameter('w', 'l', 'Dense', 'weight', [2], 2, FixedPrecisionType(16, 6, rounding_mode='RND'))
    wrap_n = ExternalParameter('w', 'l', 'Dense', 'weight', [2], 2, FixedPrecisionType(16, 6, saturation_bits=3))
    integer = ExternalParameter('b', 'l', 'Dense', 'bias', [2], 2, IntegerPrecisionType(8))
    for original in (rnd, wrap_n, integer):
        rebuilt = ExternalParameter.deserialize(json.loads(json.dumps(original.serialize_state())))
        assert type(rebuilt.precision) is type(original.precision)
        assert rebuilt.precision == original.precision
        with pytest.raises(ExternalParameterError):
            rebuilt.quantize([0.0, 0.0])


def test_external_parameters_is_layer_config_only(tmp_path):
    """Roles belong to a layer: LayerType applies, a Model-level entry is an error."""
    by_type = _convert_with(_dense_model(), tmp_path / 'by_type', type_config={'Dense': {'ExternalParameters': ['bias']}})
    assert [p.role for p in build_manifest(by_type)] == ['bias']

    with pytest.raises(ValueError, match="not under 'Model'"):
        _convert_with(_dense_model(), tmp_path / 'global', model_config={'ExternalParameters': ['weight']})


@pytest.mark.parametrize('bad', ['weight', {'weight': True}, [1]])
def test_external_parameters_must_be_a_list_of_names(tmp_path, bad):
    with pytest.raises(ValueError, match='list or tuple of parameter names'):
        _convert_with(_dense_model(), tmp_path / 'bad', {'dense_1': {'ExternalParameters': bad}})


def test_described_parameter_must_be_consistent():
    from hls4ml.model.external_parameters import BlockLayout, ExternalParameter, ExternalParameterError, FlatOrder

    def make(**overrides):
        fields = dict(
            name='w',
            layer='l',
            layer_class='Dense',
            role='weight',
            tensor_shape=[8, 4],
            n_scalars=32,
            precision=FixedPrecisionType(16, 6),
            interface_kind='bram',
            data_width=256,
            depth=2,
            flat_order=FlatOrder(['n_in', 'n_out'], ['n_out', 'n_in'], [8, 4]),
            layout=BlockLayout(2, 16),
        )
        fields.update(overrides)
        return ExternalParameter(**fields)

    make()  # consistent
    for bad in (
        dict(n_scalars=30),  # shape holds 32
        dict(data_width=128),  # 16 lanes x 16 bits
        dict(depth=4),  # block_size 2
        dict(layout=BlockLayout(4, 16)),  # 64 capacity for 32 scalars
        dict(flat_order=None),
        dict(interface_kind='scalar_bundle'),  # block layout
        dict(interface_kind='rom'),
        dict(interface_kind=None),  # layout claimed without a kind
        dict(interface_kind=None, flat_order=None, layout=None),  # width/depth claimed without a kind
        dict(tensor_shape=[8, 5]),  # 40, not 32
    ):
        with pytest.raises(ExternalParameterError):
            make(**bad)
    with pytest.raises(ExternalParameterError):
        FlatOrder(['n_in', 'n_out'], ['n_out', 'n_out'], [8, 4])


def test_unencodable_precisions_are_named():
    """The one rule the writer and the packer both apply."""
    from hls4ml.model.external_parameters import unencodable_reason

    assert unencodable_reason(FixedPrecisionType(16, 6)) is None
    assert unencodable_reason(FixedPrecisionType(16, 6, rounding_mode='RND'))
    assert unencodable_reason(FixedPrecisionType(16, 6, saturation_mode='SAT'))
    # AP_WRAP with saturation bits is not a plain modulo wrap
    assert 'saturation bits' in unencodable_reason(FixedPrecisionType(16, 6, saturation_bits=3))
    assert unencodable_reason(IntegerPrecisionType(8))


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

    ports = _ports(hls_model)
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
        assert weight['expected_data_width'] == weight['precision']['state']['width']
        assert weight['expected_depth'] == weight['n_scalars'] == 8 * n_out


@pytest.mark.parametrize('reuse_factor', [1, 2, 8])
def test_pointwise_geometry_is_independent_of_reuse_factor(tmp_path, reuse_factor):
    """No reshape reaches the port, so the geometry does not move with the reuse factor.

    Reuse factor 1 is included deliberately: for a Dense it collapses the memory into
    a single word and is refused, and the pointwise layout must not inherit that.
    """
    inp = Input(shape=(4, 8), name='input_1')
    model = Model(inp, Dense(6, activation='linear', name='d')(inp))
    hls_model = _convert(model, tmp_path / f'pw_rf{reuse_factor}', reuse_factor=reuse_factor)

    weight = next(p for p in _ports(hls_model) if p['role'] == 'weight')
    assert weight['kernel_variant'] == 'pointwise_unreshaped'
    assert weight['expected_data_width'] == weight['precision']['state']['width']
    assert weight['expected_depth'] == weight['n_scalars'] == 8 * 6
    assert weight['layout']['lanes'] == 1


def test_a_wider_kernel_is_not_pointwise(tmp_path):
    """The pointwise adapter must not capture a genuine convolution."""
    inp = Input(shape=(6, 8), name='input_1')
    model = Model(inp, Conv1D(6, 3, activation='linear', name='c')(inp))
    hls_model = _convert(model, tmp_path / 'conv1d_k3')

    ports = _ports(hls_model)
    assert {p['layer_class'] for p in ports} == {'Conv1D'}
    for port in ports:
        assert port['expected_interface_kind'] is None
        assert 'no adapter for' in port['note']


# Everything the manifest claims about a port, minus its hls4ml-assigned name: the
# frontends name layers differently ('dense_1' vs '_0') but must agree on all of this.
_PORT_CLAIMS = (
    'layer_class',
    'role',
    'expected_interface_kind',
    'expected_data_width',
    'expected_depth',
    'n_scalars',
    'kernel_variant',
    'flat_order',
    'layout',
    'precision',
)


def _claims_by_role(hls_model):
    return {p['role']: {k: p.get(k) for k in _PORT_CLAIMS} for p in _ports(hls_model)}


def test_pytorch_frontend_produces_the_same_manifest(tmp_path):
    """The feature lives after conversion, so the frontend must not matter."""
    torch = pytest.importorskip('torch')

    n_in, n_out = 8, 4
    w = (np.arange(n_in * n_out, dtype=np.float32).reshape(n_in, n_out) + 1) / 1024.0
    b = (np.arange(n_out, dtype=np.float32) + 1) / 1024.0

    keras_claims = _claims_by_role(_convert(_dense_model(n_in, n_out), tmp_path / 'keras'))

    model = torch.nn.Sequential(torch.nn.Linear(n_in, n_out))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(w.T))  # torch keeps (n_out, n_in)
        model[0].bias.copy_(torch.tensor(b))
    cfg = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=(None, n_in),
        granularity='model',
        backend='Vitis',
        default_precision='ap_fixed<16,6>',
        default_reuse_factor=2,
    )
    cfg['Model']['Strategy'] = 'Resource'
    cfg['LayerName'] = {'0': {'ExternalParameters': ['weight', 'bias']}}
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=cfg,
        output_dir=str(tmp_path / 'torch'),
        project_name='torch_prj',
        backend='Vitis',
        io_type='io_parallel',
    )

    assert _claims_by_role(hls_model) == keras_claims


def test_registry_stays_inside_the_verified_envelope():
    from hls4ml.writer.external_parameters import described_combinations

    for backend, io_type, strategy, _layer_class, _role in described_combinations():
        assert (backend, io_type, strategy) == ('Vitis', 'io_parallel', 'resource')
