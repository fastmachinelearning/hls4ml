import re
import subprocess
import sys
from pathlib import Path

import pytest

test_root_path = Path(__file__).parent

# Two models with identical layer (and therefore pipe) names, compiled and loaded into one process. Before the pipe
# declarations were namespaced per project, loading the second library aborted the interpreter inside libsycl, so this
# runs in a subprocess to report a regression as a failure rather than a crashed worker.
_two_models_script = """
import sys

import keras
import numpy as np

import hls4ml

io_type = sys.argv[1]
X = np.random.rand(100, 8)
for output_dir in sys.argv[2:]:
    model = keras.Sequential(
        [
            keras.Input(shape=(8,), name='x'),
            keras.layers.Dense(8, name='dense'),
            keras.layers.Activation('relu', name='relu'),
        ]
    )
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend='Altera', io_type=io_type
    )
    hls_model.compile()
    np.testing.assert_allclose(hls_model.predict(X), model.predict(X, verbose=0), rtol=0, atol=0.05)
"""


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_two_models_same_layer_names(test_case_id, io_type):
    output_dirs = [str(test_root_path / f'{test_case_id}_{i}') for i in range(2)]
    result = subprocess.run(
        [sys.executable, '-c', _two_models_script, io_type, *output_dirs], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr

    namespaces = []
    for output_dir in output_dirs:
        header = Path(output_dir, 'src/firmware/myproject.h').read_text()
        namespaces.append(re.search(r'^namespace (\w+) \{', header, re.MULTILINE).group(1))
        assert 'class XPipeID;' in header
    assert namespaces[0] != namespaces[1]
