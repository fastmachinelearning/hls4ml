"""Tests for input layer inclusion in profiling plots (Issue #404).

Verifies that profiling activations include the distribution of input data
and that activation type overlays cover input variable precisions.
"""

import numpy as np
import pytest

keras = pytest.importorskip('keras')

import hls4ml  # noqa: E402
from hls4ml.model.profiling import (  # noqa: E402
    _normalize_input_data,
    activation_types_hlsmodel,
    activations_keras,
)

# ---------------------------------------------------------------------------
# Tests for _normalize_input_data helper
# ---------------------------------------------------------------------------


class TestNormalizeInputData:
    """Tests for the private _normalize_input_data helper."""

    def test_single_ndarray(self):
        X = np.random.rand(10, 5)
        result = _normalize_input_data(X, ['my_input'])
        assert len(result) == 1
        assert result[0][0] == 'my_input'
        np.testing.assert_array_equal(result[0][1], X)

    def test_list_of_ndarrays(self):
        a, b = np.random.rand(10, 5), np.random.rand(10, 3)
        result = _normalize_input_data([a, b], ['in1', 'in2'])
        assert len(result) == 2
        assert result[0][0] == 'in1'
        assert result[1][0] == 'in2'
        np.testing.assert_array_equal(result[0][1], a)
        np.testing.assert_array_equal(result[1][1], b)

    def test_dict_of_ndarrays(self):
        a, b = np.random.rand(10, 5), np.random.rand(10, 3)
        X = {'in1': a, 'in2': b}
        result = _normalize_input_data(X, ['in1', 'in2'])
        assert len(result) == 2
        assert result[0][0] == 'in1'
        assert result[1][0] == 'in2'
        np.testing.assert_array_equal(result[0][1], a)
        np.testing.assert_array_equal(result[1][1], b)

    def test_dict_respects_input_names_order(self):
        """Keys present in input_names but missing from X are skipped."""
        a = np.random.rand(10, 5)
        result = _normalize_input_data({'in1': a}, ['in1', 'in2'])
        assert len(result) == 1
        assert result[0][0] == 'in1'


# ---------------------------------------------------------------------------
# Tests for activations_keras including input distribution
# ---------------------------------------------------------------------------


class TestActivationsKerasIncludesInput:
    """activations_keras must prepend the input distribution."""

    def test_input_in_summary_format(self):
        inputs = keras.Input(shape=(10,), name='fc1_input')
        x = keras.layers.Dense(5, name='dense1')(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        X = np.random.rand(50, 10).astype(np.float32) + 0.1

        data = activations_keras(model, X, fmt='summary', plot='boxplot')
        weights = [entry['weight'] for entry in data]

        assert 'fc1_input' in weights, 'Input distribution missing from activations'
        assert weights[0] == 'fc1_input', 'Input should be the first entry'

    def test_input_in_longform_format(self):
        inputs = keras.Input(shape=(10,), name='fc1_input')
        x = keras.layers.Dense(5, name='dense1')(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        X = np.random.rand(50, 10).astype(np.float32) + 0.1

        data = activations_keras(model, X, fmt='longform', plot='boxplot')
        unique_weights = list(dict.fromkeys(data['weight'].tolist()))

        assert 'fc1_input' in unique_weights, 'Input distribution missing from longform activations'
        assert unique_weights[0] == 'fc1_input', 'Input should appear first in longform data'


# ---------------------------------------------------------------------------
# Tests for activation_types_hlsmodel including input variable types
# ---------------------------------------------------------------------------


class TestActivationTypesIncludesInput:
    """activation_types_hlsmodel must include input variable precisions."""

    def test_input_type_present(self, tmp_path):
        inputs = keras.Input(shape=(10,), name='fc1_input')
        x = keras.layers.Dense(5, name='dense1')(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(tmp_path / 'hls4ml_prj'),
            backend='Vivado',
        )

        types_df = activation_types_hlsmodel(hls_model)

        assert 'fc1_input' in types_df['layer'].values, 'Input variable type missing from activation_types_hlsmodel'

    def test_input_type_has_valid_range(self, tmp_path):
        """The input type row should have finite low/high bounds."""
        inputs = keras.Input(shape=(10,), name='fc1_input')
        x = keras.layers.Dense(5, name='dense1')(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(tmp_path / 'hls4ml_prj'),
            backend='Vivado',
        )

        types_df = activation_types_hlsmodel(hls_model)
        input_row = types_df[types_df['layer'] == 'fc1_input']

        assert len(input_row) == 1, 'Expected exactly one row for fc1_input'
        assert np.isfinite(input_row['low'].iloc[0])
        assert np.isfinite(input_row['high'].iloc[0])
