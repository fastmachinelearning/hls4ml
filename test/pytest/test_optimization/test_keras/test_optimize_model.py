import numpy as np
import pytest
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from hls4ml.optimization.dsp_aware_pruning import optimize_keras_model_for_hls4ml
from hls4ml.optimization.dsp_aware_pruning.attributes import get_attributes_from_keras_model
from hls4ml.optimization.dsp_aware_pruning.keras import optimize_model
from hls4ml.optimization.dsp_aware_pruning.keras.config import DENSE_LAYERS
from hls4ml.optimization.dsp_aware_pruning.keras.regularizers import DenseRegularizer
from hls4ml.optimization.dsp_aware_pruning.keras.utils import get_model_sparsity, reset_optimizer_state
from hls4ml.optimization.dsp_aware_pruning.objectives import ParameterEstimator
from hls4ml.optimization.dsp_aware_pruning.objectives.vivado_objectives import VivadoDSPEstimator
from hls4ml.optimization.dsp_aware_pruning.scheduler import PolynomialScheduler
from hls4ml.utils.config import config_from_keras_model

"""
End-to-end runs of the Optimization API on a small classifier
A relative tolerance of zero accepts every pruning step, so the scheduler always reaches the final sparsity
"""

final_sparsity = 0.5

rng = np.random.default_rng(0)
projection = rng.normal(size=(8, 4))
X = rng.normal(size=(512, 8)).astype('float32')
y = np.eye(4, dtype='float32')[np.argmax(X @ projection, axis=1)]
X_train, y_train, X_val, y_val = X[:384], y[:384], X[384:], y[384:]


@pytest.mark.parametrize('dense', list(DENSE_LAYERS))
@pytest.mark.parametrize(
    'tuner, ranking_metric',
    [('Manual', 'l1'), ('Manual', 'oracle'), ('Manual', 'saliency'), ('Bayesian', 'l1'), ('Hyperband', 'l1')],
)
def test_optimize_model(tuner, ranking_metric, dense, tmp_path):
    model = Sequential()
    model.add(dense(16, input_shape=(8,), activation='relu', name='dense_1'))
    model.add(dense(4, name='dense_2'))

    optimized_model = optimize_model(
        model,
        get_attributes_from_keras_model(model),
        ParameterEstimator,
        PolynomialScheduler(2, final_sparsity=final_sparsity),
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=64,
        epochs=2,
        optimizer=Adam(),
        loss_fn=CategoricalCrossentropy(from_logits=True),
        validation_metric=CategoricalAccuracy(),
        increasing=True,
        rtol=0,
        ranking_metric=ranking_metric,
        tuner=tuner,
        directory=str(tmp_path),
    )

    assert get_model_sparsity(optimized_model)[0] == pytest.approx(final_sparsity, abs=0.01)
    for layer in optimized_model.layers:
        assert not isinstance(layer.kernel_regularizer, DenseRegularizer)

    # The optimized model is compiled with its own optimizer, so it can be trained further
    optimized_model.fit(X_train, y_train, epochs=1, verbose=0)


def test_optimize_keras_model_for_hls4ml(tmp_path):
    model = Sequential()
    model.add(Dense(16, input_shape=(8,), activation='relu', name='dense_1'))
    model.add(Dense(4, name='dense_2'))

    hls_config = config_from_keras_model(
        model, granularity='name', default_precision='ac_fixed<16, 6>', default_reuse_factor=4
    )
    hls_config['IOType'] = 'io_parallel'
    hls_config['Model']['Strategy'] = 'Resource'

    optimized_model = optimize_keras_model_for_hls4ml(
        model,
        hls_config,
        VivadoDSPEstimator,
        PolynomialScheduler(2, final_sparsity=final_sparsity),
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=64,
        epochs=2,
        optimizer=Adam(),
        loss_fn=CategoricalCrossentropy(from_logits=True),
        validation_metric=CategoricalAccuracy(),
        increasing=True,
        rtol=0,
        tuner='Manual',
        directory=str(tmp_path),
    )

    # Every DSP multiplies the same number of weights, so removing half of the DSPs removes half of the weights
    assert get_model_sparsity(optimized_model)[0] == pytest.approx(final_sparsity, abs=0.01)


def test_reset_optimizer_state():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.compile(Adam(learning_rate=0.01), 'mse')
    model.fit(np.ones((4, 3)), np.ones((4, 2)), verbose=0)

    reset_optimizer_state(model.optimizer)

    assert model.optimizer.iterations.numpy() == 0
    assert model.optimizer.learning_rate.numpy() == pytest.approx(0.01)
