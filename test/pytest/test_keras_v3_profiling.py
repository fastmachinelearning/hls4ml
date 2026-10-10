"""Test numerical profiling with Keras v3 models."""

from pathlib import Path

import numpy as np
import pytest

try:
    import keras

    __keras_profiling_enabled__ = keras.__version__ >= '3.0'
except ImportError:
    __keras_profiling_enabled__ = False

if __keras_profiling_enabled__:
    from hls4ml.model.profiling import activations_keras, compare, get_ymodel_keras, numerical


def count_bars_in_figure(fig):
    """Count the number of bars in all axes of a figure."""
    count = 0
    for ax in fig.get_axes():
        count += len(ax.patches)
    return count


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
def test_keras_v3_numerical_profiling_simple_model():
    """Test numerical profiling with a simple Keras v3 Dense model."""
    model = keras.Sequential(
        [
            keras.layers.Dense(20, input_shape=(10,), activation='relu'),
            keras.layers.Dense(5, activation='softmax'),
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # Build the model so weights are initialized
    model.build((None, 10))

    # Test profiling weights only
    wp, _, _, _ = numerical(model)
    assert wp is not None
    # Should have 2 bars (one per layer, each showing weights and biases combined)
    assert count_bars_in_figure(wp) == 2


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
def test_keras_v3_numerical_profiling_with_activations():
    """Test numerical profiling with Keras v3 model including activations."""
    # Use functional API instead of Sequential to ensure input layer is properly defined
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(20, activation='relu')(inputs)
    outputs = keras.layers.Dense(5)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    # Generate test data
    X_test = np.random.rand(100, 10).astype(np.float32)

    # Test profiling with activations
    wp, _, ap, _ = numerical(model, X=X_test)
    assert wp is not None
    assert ap is not None


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
def test_keras_v3_numerical_profiling_conv_model():
    """Test numerical profiling with a Keras v3 Conv model."""
    model = keras.Sequential(
        [
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax'),
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # Build the model so weights are initialized
    model.build((None, 28, 28, 1))

    # Test profiling weights
    wp, _, _, _ = numerical(model)
    assert wp is not None
    # Conv layer has 1 bar, Dense layer has 1 bar = 2 bars total
    assert count_bars_in_figure(wp) == 2


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
def test_keras_v3_numerical_profiling_with_hls_model(test_case_id):
    """Test numerical profiling with both Keras v3 model and hls4ml model."""
    import hls4ml

    # Use functional API to ensure input layer is properly defined
    inputs = keras.Input(shape=(8,))
    x = keras.layers.Dense(16, activation='relu')(inputs)
    outputs = keras.layers.Dense(4, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Generate test data
    X_test = np.random.rand(100, 8).astype(np.float32)

    # Create hls4ml model, tracing every layer so that activations can be profiled
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Trace'] = True
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(Path(__file__).parent / test_case_id),
        backend='Vivado',
        allow_da_fallback=True,
        allow_v2_fallback=True,
    )

    # Test profiling with both models
    wp, wph, ap, aph = numerical(model, hls_model=hls_model, X=X_test)

    assert wp is not None  # Keras model weights (before optimization)
    assert wph is not None  # HLS model weights (after optimization)
    assert ap is not None  # Keras model activations (before optimization)
    assert aph is not None  # HLS model activations (after optimization)


def _multiple_inputs_models(output_dir):
    """Build a Keras model with two inputs of differing shapes, and its traced hls4ml model."""
    import hls4ml

    input_1 = keras.Input(shape=(16, 21), name='basic_input')
    input_2 = keras.Input(shape=(1,), name='jet_pt')
    x = keras.layers.Flatten()(input_1)
    x = keras.layers.Dense(8, activation='relu')(x)
    x = keras.layers.Concatenate()([x, input_2])
    outputs = keras.layers.Dense(4, activation='softmax')(x)
    model = keras.Model(inputs=[input_1, input_2], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Trace'] = True

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend='Vivado',
    )
    return model, hls_model


def _multiple_inputs_data(input_format):
    """One array per model input, with different shapes, packed as a list, tuple or dict."""
    X_test = {
        'basic_input': np.random.rand(10, 16, 21).astype(np.float32),
        'jet_pt': np.random.rand(10, 1).astype(np.float32),
    }
    if input_format == 'list':
        return list(X_test.values())
    if input_format == 'tuple':
        return tuple(X_test.values())
    return X_test


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
@pytest.mark.parametrize('input_format', ['list', 'tuple', 'dict'])
def test_keras_v3_numerical_profiling_multiple_inputs(test_case_id, input_format):
    """Test numerical profiling of a model with more than one input, of differing shapes."""
    model, hls_model = _multiple_inputs_models(str(Path(__file__).parent / test_case_id))
    X_test = _multiple_inputs_data(input_format)

    wp, wph, ap, aph = numerical(model, hls_model=hls_model, X=X_test)

    assert wp is not None  # Keras model weights (before optimization)
    assert wph is not None  # HLS model weights (after optimization)
    assert ap is not None  # Keras model activations (before optimization)
    assert aph is not None  # HLS model activations (after optimization)


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
def test_keras_v3_numerical_profiling_missing_input(test_case_id):
    """Test that a dict of test data without an entry for every model input is rejected."""
    _, hls_model = _multiple_inputs_models(str(Path(__file__).parent / test_case_id))
    X_test = _multiple_inputs_data('dict')
    del X_test['jet_pt']

    with pytest.raises(ValueError, match='jet_pt'):
        numerical(hls_model=hls_model, X=X_test)


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
def test_keras_v3_compare_multiple_inputs(test_case_id):
    """Test the layer-by-layer comparison of a model with more than one input."""
    model, hls_model = _multiple_inputs_models(str(Path(__file__).parent / test_case_id))
    X_test = _multiple_inputs_data('list')

    f = compare(model, hls_model, X_test)

    assert f is not None


def _single_layer_model(activation):
    """Build a Keras model with a single Dense layer."""
    inputs = keras.Input(shape=(8,))
    outputs = keras.layers.Dense(4, activation=activation, name='dense')(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
def test_keras_v3_activations_profiling_single_layer():
    """Test that the activations of a model with a single layer are profiled over all samples."""
    model = _single_layer_model('relu')
    X_test = np.random.rand(10, 8).astype(np.float32)

    data = activations_keras(model, X_test, fmt='longform')

    assert len(data) == np.count_nonzero(model.predict(X_test))


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
@pytest.mark.parametrize('activation', ['relu', None])
def test_keras_v3_layer_outputs_single_layer(activation):
    """Test that the layer outputs used by compare() cover all samples for a model with a single layer."""
    model = _single_layer_model(activation)
    X_test = np.random.rand(10, 8).astype(np.float32)

    ymodel = get_ymodel_keras(model, X_test)

    name = 'dense_relu' if activation else 'dense'
    np.testing.assert_allclose(ymodel[name], model.predict(X_test), rtol=1e-6)


@pytest.mark.skipif(not __keras_profiling_enabled__, reason='Keras 3.0 or higher is required')
def test_keras_v3_numerical_profiling_batch_norm():
    """Test numerical profiling with Keras v3 model containing BatchNormalization."""
    model = keras.Sequential(
        [
            keras.layers.Dense(20, input_shape=(10,)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dense(5, activation='softmax'),
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # Build the model so weights are initialized
    model.build((None, 10))

    # Test profiling weights
    wp, _, _, _ = numerical(model)
    assert wp is not None
    # Dense has 1 bar, BatchNorm has 1 bar, second Dense has 1 bar = 3 bars
    assert count_bars_in_figure(wp) == 3
