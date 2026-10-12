import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent


def make_model():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
    return model.eval()


def smarthls_headers_or_skip():
    shls = shutil.which('shls')
    if shls is None:
        pytest.skip('Libero tests require SmartHLS shls and its hls headers')
    smarthls_path = Path(shls).resolve().parent.parent
    header = next((path for path in smarthls_path.rglob('ap_fixpt.hpp') if path.parent.name == 'hls'), None)
    if header is None:
        pytest.skip('SmartHLS hls/ap_fixpt.hpp is unavailable')
    return smarthls_path, header


def make_hls_model(model, strategy, output_dir, smarthls_path):
    config = config_from_pytorch_model(model, (4,), granularity='name', backend='Libero')
    assert config['LayerName']['_0']['Strategy'] == 'latency'
    assert config['LayerName']['_2']['Strategy'] == 'latency'
    for name in ('_0', '_2'):
        config['LayerName'][name]['Strategy'] = strategy.lower()

    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        backend='Libero',
        io_type='io_parallel',
        smarthls_path=smarthls_path,
    )
    dense_layers = [layer for layer in hls_model.get_layers() if layer.class_name == 'Dense']
    assert [(layer.get_attr('n_in'), layer.get_attr('n_out')) for layer in dense_layers] == [(4, 3), (3, 1)]
    assert [layer.get_attr('strategy') for layer in dense_layers] == [strategy.lower()] * 2
    return hls_model


@pytest.mark.parametrize('strategy', ['Latency', 'Resource'])
def test_write(test_case_id, strategy, tmp_path):
    output_dir = test_root_path / test_case_id
    hls_model = make_hls_model(make_model(), strategy, output_dir, tmp_path)
    hls_model.write()

    expected_files = (
        'firmware/myproject.cpp',
        'firmware/myproject.h',
        'firmware/defines.h',
        'firmware/parameters.h',
        'firmware/nnet_utils/nnet_dense.h',
        'firmware/weights/w2.h',
        'firmware/weights/b2.h',
        'firmware/weights/w4.h',
        'firmware/weights/b4.h',
        'myproject_bridge.cpp',
        'myproject_test.cpp',
        'config.tcl',
        'Makefile',
        'Makefile.compile',
        'hls4ml_config.yml',
    )
    for filename in expected_files:
        assert (output_dir / filename).is_file(), filename

    for filename in ('Makefile', 'Makefile.compile'):
        makefile = (output_dir / filename).read_text()
        assert 'LOCAL_CONFIG = -shls-config=config.tcl' in makefile
        assert 'USER_CXX_FLAG =' in makefile
        assert not re.search(r'^\s*(?:include|LEVEL\s*=)', makefile, re.MULTILINE)

    for header in (output_dir / 'firmware/nnet_utils').glob('*.h'):
        assert not re.search(r'\b(?:uint|ulong|ushort)\b', header.read_text()), header

    testbench = (output_dir / 'myproject_test.cpp').read_text()
    data_path, fallback_path = testbench.split('    } else {', maxsplit=1)
    data_path = data_path.split('if (fin.is_open() && fpr.is_open()) {', maxsplit=1)[1]
    assert data_path.count('_fifo.read();') == 1
    assert fallback_path.count('_fifo.read();') == 1
    assert data_path.index('_fifo.read();') < data_path.index('if (e % CHECKPOINT == 0) {')

    parameters = (output_dir / 'firmware/parameters.h').read_text()
    for n_in, n_out in ((4, 3), (3, 1)):
        assert f'static const unsigned n_in = {n_in};' in parameters
        assert f'static const unsigned n_out = {n_out};' in parameters
    assert parameters.count(f'static const unsigned strategy = nnet::{strategy.lower()};') == 2
    assert parameters.count('static const unsigned io_type = nnet::io_parallel;') == 3

    project_cpp = (output_dir / 'firmware/myproject.cpp').read_text()
    dense_function = 'dense_latency' if strategy == 'Latency' else 'dense_resource_rf_leq_nin'
    assert project_cpp.count(f'nnet::{dense_function}<') == 2
    assert 'nnet::relu<' in project_cpp
    assert 'for (unsigned i = 0; i < 1; i++) {' in project_cpp


@pytest.mark.parametrize('strategy', ['Latency', 'Resource'])
def test_predict(test_case_id, strategy):
    smarthls_headers_or_skip()
    model = make_model()
    inputs = np.array([[0.25, -0.125, 0.5, 0.75]], dtype=np.float32)
    expected = model(torch.from_numpy(inputs)).detach().numpy()
    hls_model = make_hls_model(model, strategy, test_root_path / test_case_id, None)
    hls_model.compile()
    np.testing.assert_allclose(hls_model.predict(inputs), expected, rtol=0, atol=0.02)


def test_generated_cpp_syntax(test_case_id):
    smarthls_path, header = smarthls_headers_or_skip()
    if shutil.which('g++') is None:
        pytest.skip('g++ is required for the syntax check')
    output_dir = test_root_path / test_case_id
    make_hls_model(make_model(), 'Latency', output_dir, smarthls_path).write()
    subprocess.run(
        ['g++', '-std=c++17', '-fsyntax-only', '-I', str(header.parent.parent), 'firmware/myproject.cpp'],
        cwd=output_dir,
        check=True,
    )
