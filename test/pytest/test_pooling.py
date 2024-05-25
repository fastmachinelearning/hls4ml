from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

atol = 5e-3

def get_precision_label(precision):
    default_p = precision['default'].replace('fixed', 'fxd').replace('<','').replace('>','').replace(',','_')
    input_p = precision['input'].replace('fixed', 'fxd').replace('<','').replace('>','').replace(',','_')
    result_p = precision['result'].replace('fixed', 'fxd').replace('<','').replace('>','').replace(',','_')
    accum_p = precision['accum'].replace('fixed', 'fxd').replace('<','').replace('>','').replace(',','_')
    return f'de_{default_p}_in_{input_p}_ac_{accum_p}_re_{result_p}'

@pytest.mark.parametrize(
    'model_type',
    [
        'max',
        'avg',
    ]
)
@pytest.mark.parametrize(
    'padding',
    [
        'same',
        'valid',
    ]
)
@pytest.mark.parametrize(
    'in_shape',
    [
        124
    ]
)
@pytest.mark.parametrize(
    'in_filt',
    [
        5,
    ]
)
@pytest.mark.parametrize(
    'pool_size',
    [
        2,
        3,
    ]
)
@pytest.mark.parametrize(
    'io_type',
    [
        'io_parallel',
    ]
)
@pytest.mark.parametrize(
    'strategy',
    [
        'Latency',
    ]
)
@pytest.mark.parametrize(
    'precision',
    [
        {'default': 'fixed<32,16>', 'input': 'fixed<32,16>', 'accum': 'fixed<32,16>', 'result': 'fixed<32,16>'},
#        {'default': 'fixed<8,2>', 'input': 'fixed<4,1>', 'accum': 'fixed<8,4>', 'result': 'fixed<16,6>'},
    ]
)
@pytest.mark.parametrize(
    'backend',
    [
        'Quartus',
        'Vitis',
        'Vivado',
        'Catapult'
    ]
)
def test_pool1d(model_type, padding, in_shape, in_filt, pool_size, io_type, strategy, precision, backend):

    model = Sequential()
    if model_type == 'avg':
        model.add(AveragePooling1D(pool_size=pool_size, input_shape=(in_shape, in_filt), padding=padding, name=f'{model_type}_pooling_1d'))
    elif model_type == 'max':
        model.add(MaxPooling1D(pool_size=pool_size, input_shape=(in_shape, in_filt), padding=padding, name=f'{model_type}_pooling_1d'))

    np.random.seed(42)
    data_1d = np.random.rand(100, in_shape, in_filt)

    config = hls4ml.utils.config_from_keras_model(model, default_precision=precision['default'], granularity='name')
    config['Model']['strategy'] = strategy

    config['LayerName'][f'{model_type}_pooling_1d_input']['Precision']['result'] = precision['input']
    config['LayerName'][f'{model_type}_pooling_1d']['Precision']['result'] = precision['result']
    config['LayerName'][f'{model_type}_pooling_1d']['Precision']['accum'] = precision['accum']

    precision_label = get_precision_label(precision)
    label = f'pool1d_{model_type}_w{in_shape}_f{in_filt}_p{pool_size}_{padding}_{backend}_{io_type}_{strategy}_{precision_label}'.lower()
#
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_{label}'),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_1d)
    y_hls = hls_model.predict(data_1d).reshape(y_keras.shape)

    plt.figure()
    min_x = min(np.amin(y_hls), np.amin(y_keras))
    max_x = max(np.amax(y_hls), np.amax(y_keras))
    #golden_min_x = np.amin(y_keras)
    #golden_max_x = np.amax(y_keras)
    plt.plot([min_x, max_x], [min_x, max_x], c='gray')
    plt.scatter(y_hls.flatten(), y_keras.flatten(), s=0.2, c='red')
    plt.title(label.replace('io_parallel', 'io-parallel').replace('_', ' '), fontsize=6)
    plt.xlabel('hls4ml')
    plt.ylabel('Keras')

    plt.savefig(f'{label}.png')

    import json
    with open(f'{label}.json', 'w') as f:
        config_json = json.dumps(config, indent=4)
        f.write(config_json)

    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)

@pytest.mark.parametrize(
    'model_type',
    [
        'max',
        'avg',
    ]
)
@pytest.mark.parametrize(
    'padding',
    [
        'same',
        'valid',
    ]
)
@pytest.mark.parametrize(
    'in_shape',
    [
        [124, 124],
        [11, 19],
    ]
)
@pytest.mark.parametrize(
    'in_filt',
    [
        5,
    ]
)
@pytest.mark.parametrize(
    'pool_size',
    [
        [2, 2],
        [3, 3]
    ]
)
@pytest.mark.parametrize(
    'io_type',
    [
        'io_parallel',
    ]
)
@pytest.mark.parametrize(
    'strategy',
    [
        'Latency',
    ]
)
@pytest.mark.parametrize(
    'precision',
    [
        {'default': 'fixed<32,16>', 'input': 'fixed<32,16>', 'accum': 'fixed<32,16>', 'result': 'fixed<32,16>'},
#        {'default': 'fixed<8,2>', 'input': 'fixed<4,1>', 'accum': 'fixed<8,4>', 'result': 'fixed<16,6>'},

    ]
)
@pytest.mark.parametrize(
    'backend',
    [
        'Quartus',
        'Vitis',
        'Vivado',
        'Catapult'
    ]
)
def test_pool2d(model_type, padding, in_shape, in_filt, pool_size, io_type, strategy, precision, backend):
    test_id = "my-function"

    model = Sequential()
    if model_type == 'avg':
        model.add(AveragePooling2D(input_shape=(in_shape[0], in_shape[1], in_filt), pool_size=pool_size, padding=padding, name=f'{model_type}_pooling_2d'))
    elif model_type == 'max':
        model.add(MaxPooling2D(input_shape=(in_shape[0], in_shape[1], in_filt), pool_size=pool_size, padding=padding, name=f'{model_type}_pooling_2d'))

    np.random.seed(42)
    data_2d = np.random.rand(100, in_shape[0], in_shape[1], in_filt)

    config = hls4ml.utils.config_from_keras_model(model, default_precision=precision['default'], granularity='name', backend=backend)
    config['Model']['strategy'] = strategy

    config['LayerName'][f'{model_type}_pooling_2d_input']['Precision']['result'] = precision['input']
    config['LayerName'][f'{model_type}_pooling_2d']['Precision']['result'] = precision['result']
    config['LayerName'][f'{model_type}_pooling_2d']['Precision']['accum'] = precision['accum']

    precision_label = get_precision_label(precision)
    label = f'pool2d_{model_type}_h{in_shape[0]}_w{in_shape[1]}_f{in_filt}_ph{pool_size[0]}_pw{pool_size[1]}_{padding}_{backend}_{io_type}_{strategy}_{precision_label}'.lower()

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_{label}'),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_2d)
    y_hls = hls_model.predict(data_2d).reshape(y_keras.shape)

    plt.figure()
    min_x = min(np.amin(y_hls), np.amin(y_keras))
    max_x = max(np.amax(y_hls), np.amax(y_keras))
    #golden_min_x = np.amin(y_keras)
    #golden_max_x = np.amax(y_keras)
    plt.plot([min_x, max_x], [min_x, max_x], c='gray')
    plt.scatter(y_hls.flatten(), y_keras.flatten(), s=0.2, c='red')
    plt.title(label.replace('io_parallel', 'io-parallel').replace('_', ' '), fontsize=6)
    plt.xlabel('hls4ml')
    plt.ylabel('Keras')

    plt.savefig(f'{label}.png')

    import json
    with open(f'{label}.json', 'w') as f:
        config_json = json.dumps(config, indent=4)
        f.write(config_json)

    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
