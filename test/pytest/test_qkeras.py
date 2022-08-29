import pytest
import hls4ml
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Activation, BatchNormalization, Input
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu, ternary, binary, quantized_tanh, quantized_sigmoid
from qkeras.utils import _add_supported_quantized_objects; co = {}; _add_supported_quantized_objects(co)

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

test_root_path = Path(__file__).parent
example_model_path = (test_root_path / '../../example-models').resolve()

@pytest.fixture(scope='module')
def get_jettagging_data():
  '''
  Download the jet tagging dataset
  '''
  print("Fetching data from openml")
  data = fetch_openml('hls4ml_lhc_jets_hlf')
  X, y = data['data'], data['target']
  le = LabelEncoder()
  y = le.fit_transform(y)
  y = to_categorical(y, 5)
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  scaler = StandardScaler()
  X_train_val = scaler.fit_transform(X_train_val)
  X_test = scaler.transform(X_test)
  return X_train_val, X_test, y_train_val, y_test

@pytest.fixture(scope='module')
def load_jettagging_model():
  '''
  Load the 3 hidden layer QKeras example model trained on the jet tagging dataset
  '''
  model_path = example_model_path / 'keras/qkeras_3layer.json'
  with model_path.open('r') as f:
    jsons = f.read()
  model = model_from_json(jsons, custom_objects=co)
  model.load_weights(example_model_path / 'keras/qkeras_3layer_weights.h5')
  return model

# TODO - Paramaterize for Quartus (different strategies?)
@pytest.fixture
@pytest.mark.parametrize('strategy', ['latency', 'resource'])
def convert(load_jettagging_model, strategy):
  '''
  Convert a QKeras model trained on the jet tagging dataset
  '''
  model = load_jettagging_model
  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation'], rounding_mode='AP_RND', saturation_mode='AP_SAT')

  config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  config['Model']['Strategy'] = strategy
  config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
  config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
  hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir=str(test_root_path / 'hls4mlprj_qkeras_accuracy_{}'.format(strategy)),
                                                       part='xcu250-figd2104-2L-e')
  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=[])
  hls_model.compile()
  return hls_model

@pytest.mark.parametrize('strategy', ['latency', 'resource'])
def test_accuracy(convert, load_jettagging_model, get_jettagging_data, strategy):
  '''
  Test the hls4ml-evaluated accuracy of a 3 hidden layer QKeras model trained on
  the jet tagging dataset. QKeras model accuracy is required to be over 70%, and
  hls4ml accuracy required to be within 1% of the QKeras model accuracy.
  '''
  print("Test accuracy")
  from sklearn.metrics import accuracy_score

  X_train_val, X_test, y_train_val, y_test = get_jettagging_data

  hls_model = convert
  model = load_jettagging_model

  y_qkeras = model.predict(np.ascontiguousarray(X_test))
  y_hls4ml = hls_model.predict(np.ascontiguousarray(X_test))

  acc_qkeras = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_qkeras, axis=1))
  acc_hls4ml = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls4ml, axis=1))
  rel_diff = abs(acc_qkeras - acc_hls4ml) / acc_qkeras

  print('Accuracy qkeras:     {}'.format(acc_qkeras))
  print('Accuracy hls4ml:     {}'.format(acc_hls4ml))
  print('Relative difference: {}'.format(rel_diff))

  assert acc_qkeras > 0.7 and rel_diff < 0.01

def randX(batch_size, N):
  return np.random.rand(batch_size,N)

@pytest.fixture(scope='module')
def randX_100_16():
  return randX(100, 16)

# TODO: include wider bitwidths when that can be made to pass
# Note 4-bit test can still fail sometimes depending on random seed
# https://github.com/fastmachinelearning/hls4ml/issues/381
#@pytest.mark.parametrize('bits', [4, 6, 8])
@pytest.mark.parametrize('bits,alpha', [(4, 1), (4, 'auto_po2')])
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_single_dense_activation_exact(randX_100_16, bits, alpha, backend, io_type):
  '''
  Test a single Dense -> Activation layer topology for
  bit exactness with number of bits parameter
  '''
  X = randX_100_16
  model = Sequential()
  model.add(QDense(16, input_shape=(16,), name='fc1',
                  kernel_quantizer=quantized_bits(bits,0,alpha=alpha), bias_quantizer=quantized_bits(bits,0,alpha=1),
                  kernel_initializer='lecun_uniform'))
  model.add(QActivation(activation=quantized_relu(bits,0), name='relu1'))
  model.compile()

  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['relu1'], rounding_mode='AP_RND_CONV', saturation_mode='AP_SAT')
  config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  output_dir = str(test_root_path / 'hls4mlprj_qkeras_single_dense_activation_exact_{}_{}_{}_{}'.format(bits, alpha, backend, io_type))
  hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir=output_dir,
                                                       backend=backend,
                                                       io_type=io_type)
  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=[])
  hls_model.compile()

  y_qkeras = model.predict(X)
  y_hls4ml = hls_model.predict(X)
  # Goal is to get it passing with all equal
  #np.testing.assert_array_equal(y_qkeras, y_hls4ml)
  # For now allow matching within 1 bit
  np.testing.assert_allclose(y_qkeras.ravel(), y_hls4ml.ravel(), atol=2**-bits, rtol=1.0)

@pytest.fixture
def make_btnn(test_no, N, kernel_quantizer, bias_quantizer, activation_quantizer, use_batchnorm, is_xnor):
  shape = (N,)
  model = Sequential()
  model.add(QDense(10, input_shape=shape, kernel_quantizer=kernel_quantizer,
                   bias_quantizer=bias_quantizer, name='dense'))
  if use_batchnorm:
    model.add(BatchNormalization(name='bn'))
  model.add(QActivation(activation=activation_quantizer))
  model.compile()
  return model, is_xnor, test_no

@pytest.fixture(scope='module')
def randX_100_10():
  return randX(100, 10)

@pytest.mark.parametrize('test_no,N,kernel_quantizer,bias_quantizer,activation_quantizer,use_batchnorm,is_xnor',
                          [(1, 10, ternary(alpha=1), quantized_bits(5,2), 'binary_tanh', False, False),
                           (2, 10, binary(), quantized_bits(5,2), 'binary_tanh', False, True),
                           (3, 10, ternary(alpha='auto'), quantized_bits(5,2), binary(), True, True),
                           (4, 10, ternary(alpha='auto'), quantized_bits(5,2), 'ternary', True, False),
                           (5, 10, ternary(alpha='auto'), quantized_bits(5,2), ternary(threshold=0.2), True, False),
                           (6, 10, ternary(alpha='auto'), quantized_bits(5,2), ternary(threshold=0.8), True, False),
                           (7, 10, binary(), quantized_bits(5,2), binary(), False, True)])
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_btnn(make_btnn, randX_100_10, backend, io_type):
  model, is_xnor, test_no = make_btnn
  X = randX_100_10
  cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
  output_dir = str(test_root_path / 'hls4mlprj_btnn_{}_{}_{}'.format(test_no, backend, io_type))
  hls_model = hls4ml.converters.convert_from_keras_model(model, output_dir=output_dir, hls_config=cfg, backend=backend, io_type=io_type)
  hls_model.compile()
  y_hls = hls_model.predict(X)
  # hls4ml may return XNOR binary
  if is_xnor:
    y_hls = np.where(y_hls == 0, -1, 1)
  y_ker = model.predict(X)
  wrong = (y_hls != y_ker).ravel()
  assert sum(wrong) / len(wrong) < 0.005

@pytest.fixture(scope='module')
def randX_1000_1():
  return randX(1000, 1)

# TODO: include quantized_relu tests when they are made to pass
# https://github.com/fastmachinelearning/hls4ml/issues/377
@pytest.mark.parametrize('quantizer', [(quantized_bits(8,0)),
                                       (quantized_bits(8,4)),
                                       (quantized_bits(4,2)),
                                       (quantized_bits(4,0)),
                                       (quantized_bits(10,0)),
                                       (quantized_relu(4)),
                                       (quantized_relu(4,2)),
                                       (quantized_relu(8)),
                                       (quantized_relu(8,4)),
                                       (quantized_relu(10)),
                                       (quantized_relu(10,5))])
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_quantizer(randX_1000_1, quantizer, backend, io_type):
  '''
  Test a single quantizer as an Activation function.
  Checks the type inference through the conversion is correct without just
  using the same logic.
  '''
  X = randX_1000_1
  X = np.round(X * 2**10) * 2**-10 # make it an exact ap_fixed<16,6>
  model = Sequential()
  model.add(QActivation(input_shape=(1,), activation=quantizer, name='quantizer'))
  model.compile()

  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['quantizer'], rounding_mode='AP_RND_CONV', saturation_mode='AP_SAT')
  config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  output_dir = str(test_root_path / 'hls4mlprj_qkeras_quantizer_{}_{}_{}_{}_{}'.format(quantizer.__class__.__name__,
                                                            quantizer.bits, quantizer.integer, backend, io_type))
  hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir=output_dir,
                                                       backend=backend,
                                                       io_type=io_type)
  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=[])
  hls_model.compile()

  y_qkeras = model.predict(X)
  y_hls4ml = hls_model.predict(X)
  # Goal is to get it passing with all equal
  np.testing.assert_array_equal(y_qkeras, y_hls4ml)


@pytest.mark.parametrize('quantizer', [(quantized_tanh(8)),
                                       (quantized_tanh(12, use_real_tanh=True)),
                                       (quantized_sigmoid(5)),
                                       (quantized_sigmoid(7, use_real_sigmoid=True))
                                       ])
@pytest.mark.parametrize('backend', ['Vivado'])   # Vivado only for now
def test_quantizer_special(randX_1000_1, quantizer, backend):
  '''
  Test a single quantizer (tanh or sigmoid) as an Activation function.
  Checks the type inference through the conversion is correct without just
  using the same logic.
  '''
  X = randX_1000_1
  X = np.round(X * 2**10) * 2**-10 # make it an exact ap_fixed<16,6>
  model = Sequential()
  model.add(QActivation(input_shape=(1,), activation=quantizer, name='quantizer'))
  model.compile()

  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['quantizer'], rounding_mode='AP_RND_CONV', saturation_mode='AP_SAT')
  config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  output_dir = str(test_root_path / 'hls4mlprj_qkeras_quantizer_{}_{}_{}'.format(quantizer.__class__.__name__,
                                                            quantizer.bits, backend))
  hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir=output_dir,
                                                       backend=backend)
  hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=[])
  hls_model.compile()

  y_qkeras = model.predict(X)
  y_hls4ml = hls_model.predict(X)
  # Goal is to get it passing with all equal
  np.testing.assert_allclose(y_qkeras, y_hls4ml, rtol=1e-2, atol=0.02)


@pytest.mark.parametrize(
    'weight_quantizer,activation_quantizer,', [
        ('binary', 'binary'),
        ('ternary', 'ternary'),
        ('quantized_bits(4, 0, alpha=1)', 'quantized_relu(2, 0)'),
        ('quantized_bits(4, 0, alpha=1)', 'quantized_relu(4, 0)'),
        ('quantized_bits(4, 0, alpha=1)', 'quantized_relu(8, 0)')
    ]
)
def test_qactivation_kwarg(randX_100_10,
                           activation_quantizer,
                           weight_quantizer):
    if activation_quantizer in ['binary', 'ternary']:
        name = 'bnbt_qdense_alpha'
    else:
        name = 'qdense_{}'.format(
            eval(activation_quantizer).__class__.__name__)

    inputs = Input(shape=(10,))

    outputs = QDense(
        10,
        activation=activation_quantizer,
        name='qdense',
        kernel_quantizer=weight_quantizer,
        bias_quantizer=weight_quantizer,
        kernel_initializer='lecun_uniform'
    )(inputs)
    model = Model(inputs, outputs)

    hls4ml.model.optimizer.get_optimizer(
        'output_rounding_saturation_mode'
    ).configure(
        layers=[name],
        rounding_mode='AP_RND_CONV',
        saturation_mode='AP_SAT'
    )
    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity='name'
    )

    out_dir = str(
        test_root_path / 'hls4mlprj_qactivation_kwarg_{}'.format(
            activation_quantizer
        )
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=out_dir
    )
    hls4ml.model.optimizer.get_optimizer(
        'output_rounding_saturation_mode'
    ).configure(layers=[])
    hls_model.compile()

    # Verify if activation in hls_model
    assert name in [layer.name for layer in hls_model.get_layers()]

    # Output tests
    X = randX_100_10
    X = np.round(X * 2**10) * 2**-10
    y_qkeras = model.predict(X)
    y_hls4ml = hls_model.predict(X)
    if hasattr(eval(activation_quantizer), 'bits'):
        np.testing.assert_allclose(
            y_qkeras.ravel(),
            y_hls4ml.ravel(),
            atol=2**-eval(activation_quantizer).bits,
            rtol=1.0
        )
    else:
        if activation_quantizer == 'binary':
            y_hls4ml = np.where(y_hls4ml == 0, -1, 1)
        wrong = (y_hls4ml != y_qkeras).ravel()
        assert sum(wrong) / len(wrong) <= 0.005
