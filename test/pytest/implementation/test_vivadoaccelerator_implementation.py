import subprocess
from pathlib import Path

from implementation_helpers import run_implementation_collection_test
from tensorflow.keras.models import model_from_json

import hls4ml

test_root_path = Path(__file__).parent
example_model_path = (test_root_path / '../../../example-models').resolve()

BACKEND = 'VivadoAccelerator'
IO_TYPE = 'io_parallel'
VIVADOACC_BOARD = 'zcu102'
VIVADOACC_PART = 'xczu9eg-ffvb1156-2-e'


def _load_keras_example_model(model_json, weights_h5):
    model_path = example_model_path / model_json
    with model_path.open('r') as f:
        model = model_from_json(f.read())
    model.load_weights(example_model_path / weights_h5)
    return model


def _example_models_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=example_model_path, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _run_example_model_implementation(
    *,
    model_name,
    model_json,
    weights_h5,
    test_case_id,
    synthesis_config,
):
    model = _load_keras_example_model(model_json, weights_h5)

    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=BACKEND)
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=hls_config,
        output_dir=output_dir,
        backend=BACKEND,
        io_type=IO_TYPE,
        board=VIVADOACC_BOARD,
        part=VIVADOACC_PART,
    )

    hls_model.compile()

    run_implementation_collection_test(
        config=synthesis_config,
        hls_model=hls_model,
        test_case_id=test_case_id,
        backend=BACKEND,
        metadata={
            'artifact_id': f'{model_name}_vivadoacc_{VIVADOACC_BOARD}',
            'model': {
                'name': model_name,
                'source': 'example-models',
                'source_commit': _example_models_commit(),
                'model_json': str(Path(model_json)),
                'weights_h5': str(Path(weights_h5)),
            },
            'board': VIVADOACC_BOARD,
            'part': VIVADOACC_PART,
        },
    )


def test_keras_1layer(test_case_id, synthesis_config):
    _run_example_model_implementation(
        model_name='keras_1layer',
        model_json='keras/KERAS_1layer.json',
        weights_h5='keras/KERAS_1layer_weights.h5',
        test_case_id=test_case_id,
        synthesis_config=synthesis_config,
    )


def test_keras_conv1d_small(test_case_id, synthesis_config):
    _run_example_model_implementation(
        model_name='keras_conv1d_small',
        model_json='keras/KERAS_conv1d_small.json',
        weights_h5='keras/KERAS_conv1d_small_weights.h5',
        test_case_id=test_case_id,
        synthesis_config=synthesis_config,
    )
