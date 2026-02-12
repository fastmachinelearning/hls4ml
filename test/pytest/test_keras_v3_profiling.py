"""Test numerical profiling with Keras v3 models."""

from pathlib import Path

import numpy as np
import pytest
from conftest import get_pytest_case_id

try:
    import keras

    __keras_profiling_enabled__ = keras.__version__ >= '3.0'
except ImportError:
    __keras_profiling_enabled__ = False

if __keras_profiling_enabled__:
    from hls4ml.model.profiling import numerical


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
@pytest.mark.skip(reason='convert_from_config needs update for Keras v3 model serialization format')
def test_keras_v3_numerical_profiling_with_hls_model(request):
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

    # Create hls4ml model
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(Path(__file__).parent / get_pytest_case_id(request)),
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
