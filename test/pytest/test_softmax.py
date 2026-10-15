from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import accuracy_score

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture()
def generate_data(input_shape):
    shape = (5000, *input_shape)
    d = np.random.normal(0, 2, shape)
    modify_entries = np.random.rand(*shape) < 0.05
    d[modify_entries] = d[modify_entries] * 5 + 10
    return np.clip(d, -32, 31)


@pytest.mark.parametrize('softmax_impl', ['activation', 'standalone'])
@pytest.mark.parametrize('backend', ['Vitis', 'Altera', 'Catapult', 'XLS'])
@pytest.mark.parametrize('implementation', ['stable', 'latency', 'argmax', 'legacy'])
@pytest.mark.parametrize(
    'input_bits,input_shape,table_bits,io_type,custom_accum',
    [
        ('16,6', (8,), 'auto', 'io_parallel', False),
        ('16,6', (8,), 'auto', 'io_stream', False),
        ('16,6', (8,), '18,8', 'io_parallel', False),
        ('16,6', (8,), '18,8', 'io_stream', False),
        ('16,6', (8,), '18,8', 'io_parallel', True),
        ('16,6', (8,), '18,8', 'io_stream', True),
        ('16,6', (8,), '9,3', 'io_parallel', False),
        ('16,6', (8,), '9,3', 'io_stream', False),
        ('9,6', (8,), '18,8', 'io_parallel', False),
        ('9,6', (8,), '18,8', 'io_stream', False),
        ('16,6', (8, 8, 3), '18,8', 'io_stream', False),
    ],
)
def test_softmax(
    test_case_id,
    softmax_impl,
    backend,
    implementation,
    generate_data,
    input_bits,
    input_shape,
    table_bits,
    io_type,
    custom_accum,
):

    if backend == 'Catapult' and implementation == 'argmax':
        pytest.skip('Argmax is not supported in cataplut')

    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')

    if backend == 'XLS' and table_bits == 'auto':
        pytest.skip('XLS backend does not support setting the table_bits to auto')

    if backend == 'XLS' and implementation == 'legacy':
        pytest.skip('XLS backend does not support the legacy implementation')

    X = generate_data
    model = tf.keras.models.Sequential()
    if softmax_impl == 'activation':
        model.add(tf.keras.layers.Activation(input_shape=input_shape, activation='softmax', name='softmax'))
    else:
        model.add(tf.keras.layers.Softmax(input_shape=input_shape, name='softmax'))
    model.compile()

    table_type = 'auto' if table_bits == 'auto' else f'ufixed<{table_bits}, RND_CONV, SAT>'

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    cfg['LayerName']['softmax']['Implementation'] = implementation
    cfg['LayerName']['softmax']['Precision']['inv_table'] = table_type
    cfg['LayerName']['softmax']['Precision']['exp_table'] = table_type
    cfg['LayerName']['softmax']['Precision']['inv_inp'] = table_type
    if custom_accum:
        if backend not in ['Vivado', 'Vitis']:
            pytest.skip('Custom accumulators are only supported for Vivado and Vitis backends')
        # W, I = map(int, input_bits.split(','))  # noqa: E741
        # cfg['LayerName']['softmax']['Precision']['inv_inp'] = f'ufixed<{W + 2},{I + 2}>'
    inp_layer_name = next(iter(cfg['LayerName'].keys()))
    cfg['LayerName'][inp_layer_name]['Precision']['result'] = f'fixed<{input_bits}, RND_CONV, SAT>'

    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)
    acc_hls4ml = accuracy_score(np.argmax(y_keras, axis=-1).ravel(), np.argmax(y_hls4ml, axis=-1).ravel())

    print(f'Accuracy hls4ml relative to keras: {acc_hls4ml}')

    if implementation in ('stable', 'argmax', 'legacy'):
        # loosened a bit because of random seed sensitivity
        assert acc_hls4ml >= 0.97
    elif table_bits == '9,3':
        # latency can perform poorly in some cases
        assert acc_hls4ml >= 0.72
    else:
        # This is for latency with larger tables
        assert acc_hls4ml >= 0.96


@pytest.mark.parametrize('softmax_impl', ['activation', 'standalone'])
@pytest.mark.parametrize('backend', ['Vitis', 'Altera', 'Catapult', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_softmax_skipped(test_case_id, softmax_impl, backend, io_type):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')
    X = np.random.rand(100, 10)
    dense = tf.keras.layers.Dense(14, input_shape=(10,), name='dense')
    if softmax_impl == 'activation':
        softmax = tf.keras.layers.Activation(activation='softmax', name='softmax')
    else:
        softmax = tf.keras.layers.Softmax(name='softmax')
    model = tf.keras.models.Sequential([dense, softmax])
    model.compile()

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    cfg['LayerName']['softmax']['skip'] = True

    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    # Verify Softmax was removed
    hls_layers = list(hls_model.get_layers())  # 0 is Input, 1 is Dense, 2 is Softmax (if not removed)
    assert len(hls_layers) == 2

    # Verify hls4ml output is equal to Dense output
    y_keras_dense = dense(X).numpy()  # type: ignore
    y_hls4ml = hls_model.predict(X).reshape(y_keras_dense.shape)  # type: ignore
    np.testing.assert_allclose(y_hls4ml, y_keras_dense, rtol=0, atol=2e-2)


@pytest.mark.parametrize('backend', ['Vitis', 'Altera', 'Catapult'])
def test_softmax_explicit_table_size(test_case_id, backend):
    """An explicitly configured exp/inv table size larger than TableSize is only capped for bit-exact (HGQ) layers."""
    model = tf.keras.models.Sequential([tf.keras.layers.Activation('softmax', input_shape=(8,), name='softmax')])
    model.compile()

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    cfg['LayerName']['softmax']['Implementation'] = 'stable'
    cfg['LayerName']['softmax']['TableSize'] = 1024
    cfg['LayerName']['softmax']['ExpTableSize'] = 2048
    cfg['LayerName']['softmax']['InvTableSize'] = 2048
    cfg['LayerName']['softmax']['inp_norm_t'] = 'ufixed<16,6>'
    cfg['LayerName']['softmax']['inv_inp_t'] = 'ufixed<16,6>'

    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type='io_parallel', output_dir=odir, backend=backend
    )

    softmax = hls_model.graph['softmax']
    assert softmax.get_attr('exp_table_size') == 2048
    assert softmax.get_attr('inv_table_size') == 2048
    if backend != 'Catapult':  # Catapult's softmax config has no separate exp/inv table sizes
        assert 'exp_table_size = 2048;' in softmax.get_attr('config_cpp')
        assert 'inv_table_size = 2048;' in softmax.get_attr('config_cpp')


@pytest.mark.parametrize('backend', ['Vitis', 'Altera'])
@pytest.mark.parametrize('implementation', ['stable', 'latency', 'legacy'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_softmax_values(test_case_id, backend, implementation, io_type):
    """Compare the softmax outputs themselves, which an argmax-based accuracy check cannot distinguish from argmax."""
    np.random.seed(0)
    X = np.random.rand(1000, 8) * 8 - 4
    model = tf.keras.models.Sequential([tf.keras.layers.Activation('softmax', input_shape=(8,), name='softmax')])
    model.compile()

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    cfg['LayerName']['softmax']['Implementation'] = implementation
    for table in ('inv_table', 'exp_table', 'inv_inp'):
        cfg['LayerName']['softmax']['Precision'][table] = 'ufixed<18,8,RND_CONV,SAT>'
    if implementation == 'latency':
        # The inverse table is addressed by the (inferred) accum_t, ufixed<13,9>, so size it to cover every value
        cfg['LayerName']['softmax']['Precision']['exp_table'] = 'ufixed<10,6,RND_CONV,SAT>'
        cfg['LayerName']['softmax']['InvTableSize'] = 8192
    inp_layer_name = next(iter(cfg['LayerName'].keys()))
    cfg['LayerName'][inp_layer_name]['Precision']['result'] = 'fixed<16,6,RND_CONV,SAT>'

    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)

    # Stable normalizes before the lookups, the others do not, so they are allowed larger errors
    max_mean_err, max_err = {'stable': (0.002, 0.02), 'latency': (0.005, 0.5), 'legacy': (0.01, 0.1)}[implementation]
    err = np.abs(y_hls4ml - y_keras)
    assert err.mean() < max_mean_err
    assert err.max() < max_err


@pytest.mark.parametrize('backend', ['Vitis', 'Altera'])
@pytest.mark.parametrize('input_precision', ['fixed<10,3,RND_CONV,SAT>', 'ufixed<1,1>'])
def test_softmax_stable_inp_norm(test_case_id, backend, input_precision):
    """x_max - x_i can reach twice the input range; the inferred inp_norm_t must hold it without wrapping,
    and must be usable as a table index even for a 1-bit input."""
    np.random.seed(0)
    if input_precision == 'ufixed<1,1>':
        X = (np.random.rand(1000, 8) > 0.5).astype(float)
    else:
        X = np.random.rand(1000, 8) * 8 - 4
    model = tf.keras.models.Sequential([tf.keras.layers.Activation('softmax', input_shape=(8,), name='softmax')])
    model.compile()

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    cfg['LayerName']['softmax']['Implementation'] = 'stable'
    inp_layer_name = next(iter(cfg['LayerName'].keys()))
    cfg['LayerName'][inp_layer_name]['Precision']['result'] = input_precision

    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type='io_parallel', output_dir=odir, backend=backend
    )
    hls_model.compile()

    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)

    err = np.abs(y_hls4ml - y_keras)
    assert err.mean() < 0.002
    assert err.max() < 0.02


@pytest.mark.parametrize('backend', ['Vitis', 'Altera'])
@pytest.mark.parametrize('default_precision', ['fixed<16,6>', 'fixed<24,12>'])
def test_softmax_stable_model_precision(test_case_id, backend, default_precision):
    """With only a model-level precision, the softmax internal types are inferred rather than set to the model default."""
    np.random.seed(0)
    X = np.random.rand(1000, 8) * 8 - 4
    model = tf.keras.models.Sequential([tf.keras.layers.Activation('softmax', input_shape=(8,), name='softmax')])
    model.compile()

    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity='model', backend=backend, default_precision=default_precision
    )
    cfg['LayerType'] = {'Softmax': {'Implementation': 'stable'}}

    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type='io_parallel', output_dir=odir, backend=backend
    )
    hls_model.compile()

    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)

    err = np.abs(y_hls4ml - y_keras)
    assert err.mean() < 0.005
    assert err.max() < 0.05
