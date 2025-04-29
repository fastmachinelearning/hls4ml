import logging
import os
from pathlib import Path

import pytest
from tensorflow.keras.models import model_from_json

import hls4ml
from hls4ml.utils.plot import plot_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

test_root_path = Path(__file__).parent
example_model_path = (test_root_path / '../../example-models').resolve()


@pytest.fixture(scope='module')
def load_mlp_model():
    model_path = example_model_path / 'keras/KERAS_3layer.json'
    with model_path.open('r') as f:
        jsons = f.read()
    model = model_from_json(jsons)
    model.load_weights(example_model_path / 'keras/KERAS_3layer_weights.h5')
    return model


@pytest.fixture(scope='module')
def load_cnn_model():
    model_path = example_model_path / 'keras/jetTagger_Conv2D_Small.json'
    with model_path.open('r') as f:
        jsons = f.read()
    model = model_from_json(jsons)
    model.load_weights(example_model_path / 'keras/jetTagger_Conv2D_Small_weights.h5')
    return model


@pytest.fixture(scope='module')
def convert_mlp(load_mlp_model):
    model = load_mlp_model
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='vitis')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(test_root_path / 'hls4mlprj_mlp'),
        part='xcu250-figd2104-2L-e',
    )
    hls_model.compile()
    return hls_model


@pytest.fixture(scope='module')
def convert_cnn(load_cnn_model):
    model = load_cnn_model
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='vitis')
    config['LayerName']['cnn2D_1_relu']['Precision']['accum'] = 'ap_fixed<33,17>'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(test_root_path / 'hls4mlprj_cnn'),
        part='xcu250-figd2104-2L-e',
    )
    hls_model.compile()
    return hls_model


def test_plot_mlp_model(convert_mlp):
    # Test the plot_model function with a sample model.
    output_file = 'mlp_model_plot.png'
    hls_model = convert_mlp

    plot_model(
        model=hls_model,
        to_file=str(output_file),
        show_shapes=True,
        show_layer_names=True,
        show_precision=True,
        rankdir='TB',
        dpi=96,
    )

    # Assert that the output file was created and is not empty
    assert os.path.exists(output_file), f'Plot file was not created: {output_file}'
    assert os.path.getsize(output_file) > 0, f'Plot file is empty: {output_file}'

    logger.info(f'Plot file generated at: {os.path.abspath(output_file)}')


def test_plot_cnn_model(convert_cnn):
    # Test the plot_model function with a sample model.
    output_file = 'cnn_model_plot.png'
    hls_model = convert_cnn

    plot_model(
        model=hls_model,
        to_file=str(output_file),
        show_shapes=True,
        show_layer_names=True,
        show_precision=True,
        rankdir='TB',
        dpi=96,
    )

    # Assert that the output file was created and is not empty
    assert os.path.exists(output_file), f'Plot file was not created: {output_file}'
    assert os.path.getsize(output_file) > 0, f'Plot file is empty: {output_file}'

    logger.info(f'Plot file generated at: {os.path.abspath(output_file)}')
