import ast
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_fetch_example_utils(backend):
    f = io.StringIO()
    with redirect_stdout(f):
        hls4ml.utils.fetch_example_list()
    out = f.getvalue()

    model_list = ast.literal_eval(out)  # Check if we indeed got a dictionary back

    assert 'qkeras_mnist_cnn.json' in model_list['keras']

    # This model has an example config that is also downloaded. Stored configurations don't set "Backend" value.
    config = hls4ml.utils.fetch_example_model('qkeras_mnist_cnn.json', backend=backend)
    config['KerasJson'] = 'qkeras_mnist_cnn.json'
    config['KerasH5']
    config['Backend'] = backend
    config['OutputDir'] = str(test_root_path / f'hls4mlprj_fetch_example_{backend}')

    hls_model = hls4ml.converters.keras_to_hls(config)
    hls_model.compile()  # For now, it is enough if it compiles, we're only testing downloading works as expected
