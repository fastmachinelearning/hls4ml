"""Tests of the fused strategy of the Vitis backend.

Every test that concerns the numbers compares the fused model against the same model built with the
latency strategy, rather than against Keras. The two differ only in the strategy, so the quantization is
the same for both and any difference comes from the fused kernels.
"""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import (
    ELU,
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    PReLU,
)
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent

N = 8  # width of every layer of the chain
SAMPLES = 20


def dense_chain(activation=None, n_layers=4, seed=0, n_in=N, widths=None):
    """A chain of Dense layers with an activation between each pair. The weights are scaled down so
    the values stay inside ap_fixed<16,6> and saturation does not dominate the comparison."""

    rng = np.random.default_rng(seed)
    widths = widths if widths is not None else [N] * n_layers
    n_layers = len(widths)
    inputs = Input(shape=(n_in,))
    x = inputs
    for i, width in enumerate(widths):
        x = Dense(width, name=f'fc{i}')(x)
        if activation is not None and i < n_layers - 1:
            x = activation(f'act{i}')(x)
    model = Model(inputs, x)
    for i in range(n_layers):
        layer = model.get_layer(f'fc{i}')
        w, b = layer.get_weights()
        layer.set_weights(
            [(rng.random(w.shape).astype('float32') * 2 - 1) / n_in, (rng.random(b.shape).astype('float32') - 0.5) / n_in]
        )
    return model


def convert(
    model,
    strategy,
    output_dir,
    reuse_factor=4,
    io_type='io_parallel',
    backend='Vitis',
    precisions=None,
    model_config=None,
    level='name',
):
    """Convert with the strategy set for each Dense layer by name, or with level='model' for the whole model."""

    config = hls4ml.utils.config_from_keras_model(
        model, granularity=level, backend=backend, default_precision='ap_fixed<16,6>'
    )
    if level == 'model':
        config['Model']['Strategy'] = strategy
        config['Model']['ReuseFactor'] = reuse_factor
    else:
        for name in config['LayerName']:
            if name.startswith('fc'):
                config['LayerName'][name]['Strategy'] = strategy
                config['LayerName'][name]['ReuseFactor'] = reuse_factor

    config['Model'].update(model_config or {})

    for name, precision in (precisions or {}).items():
        config['LayerName'][name]['Precision'] = dict(config['LayerName'][name].get('Precision', {}))
        config['LayerName'][name]['Precision']['result'] = precision
    return hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, backend=backend, io_type=io_type, output_dir=output_dir
    )


def forms(hls_model):
    """The form assigned to each Dense layer: 'dot', 'axpy', 'plain', or None where nothing was fused."""

    return [layer.get_attr('fused_form') for layer in hls_model.get_layers() if layer.name.startswith('fc')]


def layer_names(hls_model):
    return [layer.name for layer in hls_model.get_layers()]


def compare_with_latency(model, test_case_id, reuse_factor=4):
    """Build the model twice and run both on the same inputs. Returns the fused model and both outputs."""

    fused = convert(model, 'Fused', str(test_root_path / f'{test_case_id}_fused'), reuse_factor)
    latency = convert(model, 'Latency', str(test_root_path / f'{test_case_id}_latency'), reuse_factor)
    fused.compile()
    latency.compile()

    rng = np.random.default_rng(1)
    X = rng.random((SAMPLES, *model.input_shape[1:])).astype('float32') * 2 - 1
    y_fused = fused.predict(X).reshape(SAMPLES, -1)
    y_latency = latency.predict(X).reshape(SAMPLES, -1)
    return fused, y_fused, y_latency


def binary_tanh(x):
    return tf.math.sign(x)


# One case per activation the kernel computes. The classes matter more than the names: LeakyReLU and ELU
# are ParametrizedActivation with one parameter, hard_sigmoid is HardActivation with two, and treating
# either as a plain Activation would compute relu and lose the parameters.
FOLDED_ACTIVATIONS = [
    ('relu', lambda n: Activation('relu', name=n)),
    ('sigmoid', lambda n: Activation('sigmoid', name=n)),
    ('tanh', lambda n: Activation('tanh', name=n)),
    ('selu', lambda n: Activation('selu', name=n)),
    ('softplus', lambda n: Activation('softplus', name=n)),
    ('softsign', lambda n: Activation('softsign', name=n)),
    ('leaky_relu', lambda n: LeakyReLU(negative_slope=0.125, name=n)),
    ('elu', lambda n: ELU(alpha=1.0, name=n)),
    ('hard_sigmoid', lambda n: Activation('hard_sigmoid', name=n)),
    ('binary_tanh', lambda n: Activation(binary_tanh, name=n)),
    ('none', None),
]


@pytest.mark.parametrize('activation', FOLDED_ACTIVATIONS, ids=[case[0] for case in FOLDED_ACTIVATIONS])
def test_folded_activations(test_case_id, activation):
    """Each activation is computed inside the Dense layer and gives the same results as its own layer."""

    name, layer = activation
    model = dense_chain(layer)
    fused, y_fused, y_latency = compare_with_latency(model, test_case_id)

    assert forms(fused) == ['dot', 'axpy', 'dot', 'axpy']
    assert not any(n.startswith('act') for n in layer_names(fused)), f'the {name} layers were not folded'
    np.testing.assert_allclose(y_fused, y_latency, rtol=0, atol=1e-6)


NOT_FOLDED_ACTIVATIONS = [
    ('softmax', lambda n: Activation('softmax', name=n)),
    ('prelu', lambda n: PReLU(name=n)),
]


@pytest.mark.parametrize('activation', NOT_FOLDED_ACTIVATIONS, ids=[case[0] for case in NOT_FOLDED_ACTIVATIONS])
def test_activations_that_are_not_folded(test_case_id, activation):
    """An activation the kernel cannot compute keeps its own layer and ends the chain."""

    name, layer = activation
    model = dense_chain(layer)
    fused, y_fused, y_latency = compare_with_latency(model, test_case_id)

    assert forms(fused) == [None, None, None, None]
    assert sum(n.startswith('act') for n in layer_names(fused)) == 3, f'a {name} layer was removed'
    np.testing.assert_allclose(y_fused, y_latency, rtol=0, atol=1e-6)


CARRIED_NUMBERS = [
    ('leaky_relu', lambda n: LeakyReLU(negative_slope=0.375, name=n), 'fused_activation_param', 0.375),
    ('elu', lambda n: ELU(alpha=2.5, name=n), 'fused_activation_param', 2.5),
    ('hard_sigmoid', lambda n: Activation('hard_sigmoid', name=n), 'fused_activation_slope', None),
]


@pytest.mark.parametrize('case', CARRIED_NUMBERS, ids=[case[0] for case in CARRIED_NUMBERS])
def test_carried_numbers_reach_the_kernel(test_case_id, case):
    """The parameters of an activation reach the kernel. The values differ from the defaults, so losing
    one would change the result."""

    name, layer, attribute, expected = case
    model = dense_chain(layer)
    fused, y_fused, y_latency = compare_with_latency(model, test_case_id)

    dense = [node for node in fused.get_layers() if node.name == 'fc0'][0]
    value = dense.get_attr(attribute)
    assert value is not None, f'{name} did not pass {attribute} to the Dense layer'
    if expected is not None:
        assert float(value) == pytest.approx(expected)

    parameters = (Path(fused.config.get_output_dir()) / 'firmware' / 'parameters.h').read_text()
    assert 'nnet_dense_fused.h' in parameters
    np.testing.assert_allclose(y_fused, y_latency, rtol=0, atol=1e-6)


def test_folded_table_size(test_case_id):
    """The size of the table an activation reads is taken from the activation layer, not from a default."""

    model = dense_chain(lambda n: Activation('tanh', name=n))
    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', backend='Vitis', default_precision='ap_fixed<16,6>'
    )
    for name in config['LayerName']:
        if name.startswith('fc'):
            config['LayerName'][name]['Strategy'] = 'Fused'
            config['LayerName'][name]['ReuseFactor'] = 4
        if name.startswith('act'):
            config['LayerName'][name]['TableSize'] = 256

    fused = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, backend='Vitis', io_type='io_parallel', output_dir=str(test_root_path / test_case_id)
    )
    fused.write()

    dense = [node for node in fused.get_layers() if node.name == 'fc0'][0]
    assert dense.get_attr('fused_table_size') == 256

    defines = (Path(fused.config.get_output_dir()) / 'firmware' / 'parameters.h').read_text()
    assert 'table_size = 256' in defines


@pytest.mark.parametrize(
    'n_layers, expected',
    [
        (2, ['dot', 'axpy']),
        (3, ['plain', 'dot', 'axpy']),
        (4, ['dot', 'axpy', 'dot', 'axpy']),
        (5, ['plain', 'dot', 'axpy', 'dot', 'axpy']),
    ],
)
def test_chain_length(test_case_id, n_layers, expected):
    """Layers alternate dot and axpy, and a chain of odd length starts with a plain layer."""

    model = dense_chain(lambda n: Activation('relu', name=n), n_layers=n_layers)
    fused, y_fused, y_latency = compare_with_latency(model, test_case_id)

    assert forms(fused) == expected
    np.testing.assert_allclose(y_fused, y_latency, rtol=0, atol=1e-6)


@pytest.mark.parametrize('reuse_factor', [2, 4, 8, 16, 32, 64])
def test_reuse_factor(test_case_id, reuse_factor):
    """The results do not depend on the reuse factor. Every reuse factor below the layer width builds
    the same design, since a layer cannot use more multipliers than that."""

    model = dense_chain(lambda n: Activation('relu', name=n))
    fused, y_fused, y_latency = compare_with_latency(model, test_case_id, reuse_factor=reuse_factor)

    multipliers = [node.get_attr('fused_multipliers') for node in fused.get_layers() if node.name.startswith('fc')]
    assert all(count <= N for count in multipliers)
    assert all(count == max(1, min(N, N * N // reuse_factor)) for count in multipliers)
    np.testing.assert_allclose(y_fused, y_latency, rtol=0, atol=1e-6)


def test_layers_of_different_sizes(test_case_id):
    """A chain whose layers differ in size, at a reuse factor that divides none of them."""

    model = dense_chain(lambda n: Activation('relu', name=n), n_in=12, widths=[7, 5, 9])
    fused, y_fused, y_latency = compare_with_latency(model, test_case_id, reuse_factor=3)

    assert forms(fused) == ['plain', 'dot', 'axpy']
    multipliers = [node.get_attr('fused_multipliers') for node in fused.get_layers() if node.name.startswith('fc')]
    # The plain layer can use at most its 7 outputs; the dot and axpy pair after it gets the lower of
    # their two counts, which is the 7 inputs of the dot.
    assert multipliers == [7, 7, 7]
    np.testing.assert_allclose(y_fused, y_latency, rtol=0, atol=1e-6)


def one_layer():
    return dense_chain(n_layers=1)


def two_readers():
    """An output read by two layers. A value written to a stream can only be read once."""

    inputs = Input(shape=(N,))
    first = Dense(N, name='fc0')(inputs)
    return Model(inputs, Add(name='add')([Dense(N, name='fc1')(first), Dense(N, name='fc2')(first)]))


def not_dense_between():
    """A layer of another type between two Dense layers."""

    inputs = Input(shape=(N,))
    x = Dense(N, name='fc0')(inputs)
    x = Activation('softmax', name='soft')(x)
    return Model(inputs, Dense(N, name='fc1')(x))


def scaling_after_activation():
    """A scaling layer after each activation, where nothing merges it, so no chain forms. The activation
    layers must stay when their Dense layers are not fused."""

    inputs = Input(shape=(N,))
    x = inputs
    for i in range(3):
        x = Dense(N, name=f'fc{i}')(x)
        x = Activation('relu', name=f'act{i}')(x)
        x = BatchNormalization(name=f'scale{i}')(x)
    return Model(inputs, Dense(N, name='fc3')(x))


@pytest.mark.parametrize(
    'build',
    [one_layer, two_readers, not_dense_between, scaling_after_activation],
    ids=['single', 'two_readers', 'not_dense', 'scaling_after_activation'],
)
def test_what_ends_a_chain(test_case_id, build):
    """A layer with no neighbour to run alongside is not fused, and the results still match.

    It is built with the resource strategy, at the reuse factor it was given.
    """

    fused, y_fused, y_latency = compare_with_latency(build(), test_case_id)

    left_out = [node for node in fused.get_layers() if node.name.startswith('fc')]
    assert all(node.get_attr('fused_form') is None for node in left_out)
    assert all(node.get_attr('strategy') == 'resource' for node in left_out)
    assert all(node.get_attr('reuse_factor') == 4 for node in left_out)
    np.testing.assert_allclose(y_fused, y_latency, rtol=0, atol=1e-6)


def chain_and_a_layer_on_its_own():
    """A chain of two, then a softmax, then a Dense layer with no neighbour to fuse with."""

    inputs = Input(shape=(N,))
    x = Dense(N, name='fc0')(inputs)
    x = Activation('relu', name='act0')(x)
    x = Dense(N, name='fc1')(x)
    x = Activation('softmax', name='head')(x)
    return Model(inputs, Dense(N, name='fc2')(x))


REPORT_CASES = [
    ('reported', {}),
    ('silenced', {'FusedReport': False}),
    ('style_replaced', {'FusedReport': False, 'PipelineStyle': 'pipeline'}),
]


@pytest.mark.parametrize('case', REPORT_CASES, ids=[case[0] for case in REPORT_CASES])
def test_fusion_report(test_case_id, capsys, case):
    """The conversion prints what was built, and FusedReport turns that off. Replacing a PipelineStyle
    from the configuration is a warning and is printed either way."""

    name, model_config = case
    convert(chain_and_a_layer_on_its_own(), 'Fused', str(test_root_path / test_case_id), model_config=model_config)

    printed = capsys.readouterr().out
    reported = name == 'reported'
    assert ('fc0 (dot' in printed and 'fc1 (axpy' in printed) == reported
    assert ('asked for strategy' in printed) == reported
    assert ('pipeline style "dataflow"' in printed) == reported
    assert ('PipelineStyle "pipeline" replaced with "dataflow"' in printed) == (name == 'style_replaced')


def test_other_layer_type_ends_the_chain(test_case_id, capsys):
    """A layer type the strategy does not support is reported and built with the resource strategy."""

    inputs = Input(shape=(N, 2))
    x = Conv1D(2, 3, padding='same', name='conv')(inputs)
    x = Dense(N, name='fc0')(Flatten()(x))
    model = Model(inputs, x)

    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', backend='Vitis', default_precision='ap_fixed<16,6>'
    )
    for name in config['LayerName']:
        config['LayerName'][name]['Strategy'] = 'Fused'
        config['LayerName'][name]['ReuseFactor'] = 2

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, backend='Vitis', io_type='io_parallel', output_dir=str(test_root_path / test_case_id)
    )

    reported = capsys.readouterr().out
    assert 'conv' in reported and 'Dense layers only' in reported
    conv = [node for node in hls_model.get_layers() if node.name == 'conv'][0]
    assert conv.get_attr('strategy') != 'fused'


def test_chain_inside_a_larger_model(test_case_id):
    """A chain inside a model with other layer types, which are built as usual."""

    inputs = Input(shape=(N, 2))
    x = Conv1D(4, 3, padding='same', activation='relu', name='conv')(inputs)
    x = Flatten()(x)
    for i in range(3):
        x = Dense(N, name=f'fc{i}')(x)
        x = Activation('relu', name=f'act{i}')(x)
    model = Model(inputs, Activation('softmax', name='head')(x))

    fused, y_fused, y_latency = compare_with_latency(model, test_case_id, reuse_factor=2)

    assert forms(fused) == ['plain', 'dot', 'axpy']
    np.testing.assert_allclose(y_fused, y_latency, rtol=0, atol=1e-6)


def convert_with_interval(model, output_dir, target, names=('fc0', 'fc1', 'fc2', 'fc3'), readings=None):
    """Convert with the reuse factor read as an interval. `readings` sets the flag per layer."""

    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', backend='Vitis', default_precision='ap_fixed<16,6>'
    )
    for name in config['LayerName']:
        if name.startswith('fc'):
            config['LayerName'][name]['Strategy'] = 'Fused'
            config['LayerName'][name]['ReuseFactor'] = target[name] if isinstance(target, dict) else target
            if name in names:
                config['LayerName'][name]['ReuseFactorAsInterval'] = readings[name] if readings else True
    return hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, backend='Vitis', io_type='io_parallel', output_dir=output_dir
    )


def pads(hls_model):
    return {
        layer.name: layer.get_attr('fused_pad_cycles') for layer in hls_model.get_layers() if layer.name.startswith('fc')
    }


def test_interval_reading(test_case_id):
    """ReuseFactorAsInterval makes the reuse factor the largest interval the layer may have.

    The flag is a bool in Python and a string in JSON or YAML, so both are accepted. The expected
    multipliers and wait cycles are worked out beside the assertions below.
    """

    model = dense_chain(lambda n: Activation('relu', name=n), n_layers=2)
    latency = convert(model, 'Latency', str(test_root_path / f'{test_case_id}_latency'))
    latency.compile()
    rng = np.random.default_rng(1)
    x = rng.random((SAMPLES, N)).astype('float32') - 0.5

    default = convert(model, 'Fused', str(test_root_path / f'{test_case_id}_default'))
    default.write()
    assert not any(pads(default).values()), 'the reuse factor was read as an interval without the flag'
    parameters = (Path(default.config.get_output_dir()) / 'firmware' / 'parameters.h').read_text()
    assert parameters.count('static const unsigned pad_cycles = 0;') == 2

    for flag in (True, 'True'):
        fused = convert_with_interval(
            model, str(test_root_path / f'{test_case_id}_{flag}'), target=60, readings={'fc0': flag, 'fc1': flag}
        )
        multipliers = {
            layer.name: layer.get_attr('fused_multipliers') for layer in fused.get_layers() if layer.name.startswith('fc')
        }
        assert multipliers == {'fc0': 2, 'fc1': 2}
        # dot needs n_out * ceil(n_in / m) = 32 cycles, axpy (n_in + 1) * ceil(n_out / m) = 36, and
        # the headroom of both is 10 + ceil(8 / 2) = 14
        assert pads(fused) == {'fc0': 60 - (32 + 14), 'fc1': 60 - (36 + 14)}

        fused.compile()
        np.testing.assert_allclose(fused.predict(x), latency.predict(x), rtol=0, atol=1e-6)
        parameters = (Path(fused.config.get_output_dir()) / 'firmware' / 'parameters.h').read_text()
        assert f'static const unsigned pad_cycles = {pads(fused)["fc0"]};' in parameters

    off = convert_with_interval(
        model, str(test_root_path / f'{test_case_id}_off'), target=60, readings={'fc0': 'false', 'fc1': 'false'}
    )
    assert not any(pads(off).values()), "'false' was not read as off"


def test_precision_differs_between_layers(test_case_id):
    """Each connection inside a region carries the type of the layer that writes it.

    The precisions go on the activation layers: folding one into the Dense layer before it gives that
    layer the activation's output type.
    """

    precisions = {'act0': 'ap_fixed<12,4>', 'act1': 'ap_fixed<18,8>', 'act2': 'ap_fixed<10,3>'}
    model = dense_chain(lambda n: Activation('relu', name=n))
    fused = convert(model, 'Fused', str(test_root_path / f'{test_case_id}_fused'), precisions=precisions)
    latency = convert(model, 'Latency', str(test_root_path / f'{test_case_id}_latency'), precisions=precisions)
    fused.compile()
    latency.compile()

    assert forms(fused) == ['dot', 'axpy', 'dot', 'axpy']
    carried = {
        layer.name: (layer.get_output_variable().type.precision.width, layer.get_output_variable().type.precision.integer)
        for layer in fused.get_layers()
        if layer.name.startswith('fc')
    }
    # Each folded activation gives its type to the Dense layer before it, so the connections differ
    assert carried['fc0'] == (12, 4) and carried['fc1'] == (18, 8) and carried['fc2'] == (10, 3)

    rng = np.random.default_rng(1)
    x = rng.random((SAMPLES, N)).astype('float32') * 2 - 1
    np.testing.assert_allclose(fused.predict(x), latency.predict(x), rtol=0, atol=1e-6)


def test_scaling_layer_between_dense_layers(test_case_id):
    """A scaling layer straight after a Dense layer, where power-of-two weight quantisation places it,
    is merged into that layer and does not end the chain."""

    inputs = Input(shape=(N,))
    x = inputs
    for i in range(3):
        x = Dense(N, name=f'fc{i}')(x)
        x = BatchNormalization(name=f'scale{i}')(x)
        x = Activation('relu', name=f'act{i}')(x)
    model = Model(inputs, Dense(N, name='fc3')(x))
    fused, y_fused, y_latency = compare_with_latency(model, test_case_id)

    assert forms(fused) == ['dot', 'axpy', 'dot', 'axpy']
    np.testing.assert_allclose(y_fused, y_latency, rtol=0, atol=1e-6)


@pytest.mark.parametrize('granularity', ['model', 'layer_type'])
def test_strategy_set_for_more_than_one_layer(test_case_id, granularity):
    """The strategy and the flag can be set for the model or a layer type, not only by layer name."""

    model = dense_chain(lambda n: Activation('relu', name=n), n_layers=2)
    config = hls4ml.utils.config_from_keras_model(
        model, granularity='model', backend='Vitis', default_precision='ap_fixed<16,6>'
    )
    section = config['Model'] if granularity == 'model' else config.setdefault('LayerType', {}).setdefault('Dense', {})
    section['Strategy'] = 'Fused'
    section['ReuseFactor'] = 60
    section['ReuseFactorAsInterval'] = True

    fused = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, backend='Vitis', io_type='io_parallel', output_dir=str(test_root_path / test_case_id)
    )
    assert forms(fused) == ['dot', 'axpy']
    assert pads(fused) == {'fc0': 60 - (32 + 14), 'fc1': 60 - (36 + 14)}


CONFIGURATION_ERRORS = [
    ('interval_below_floor', dict(target=15), 'cannot achieve an interval'),
    ('interval_flag_unreadable', dict(target=60, readings={'fc0': 'yes', 'fc1': 'yes'}), 'neither true nor false'),
    ('interval_targets_differ', dict(target={'fc0': 60, 'fc1': 80}), 'must request the same one'),
    ('interval_reading_differs', dict(target=60, readings={'fc0': True, 'fc1': False}), 'use the reuse factor'),
    ('reuse_factor_one', dict(reuse_factor=1), 'set ReuseFactor to 8'),
    ('io_stream', dict(io_type='io_stream'), 'io_parallel'),
    ('io_stream_model_level', dict(io_type='io_stream', level='model'), 'io_parallel'),
    ('vivado_backend', dict(backend='Vivado'), 'fused'),
]


@pytest.mark.parametrize('case', CONFIGURATION_ERRORS, ids=[case[0] for case in CONFIGURATION_ERRORS])
def test_configuration_errors(test_case_id, case):
    """A configuration the strategy cannot carry out stops the conversion."""

    _, options, message = case
    model = dense_chain(lambda n: Activation('relu', name=n), n_layers=2)
    with pytest.raises(Exception, match=message):
        if 'target' in options:
            convert_with_interval(model, str(test_root_path / test_case_id), **options)
        else:
            convert(model, 'Fused', str(test_root_path / test_case_id), **options)
