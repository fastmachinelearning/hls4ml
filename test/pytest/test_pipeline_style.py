""" Test that pipeline style is properly handled by optimizers (respected if user-defined, correctly set if 'auto'). """

from pathlib import Path

import pytest
import tensorflow as tf

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize(
    'param_group, pipeline_style, io_type, strategy, ii',
    [
        (1, 'auto', 'io_stream', 'resource', None),  # io_stream should result in DATAFLOW pragma regardless of other params
        (2, 'auto', 'io_stream', 'latency', None),
        (3, None, 'io_stream', 'resource_unrolled', None),  # None should be interpreted as 'auto'
        (4, 'auto', 'io_parallel', 'resource', None),  # Should end up with DATAFLOW pragma
        (5, 'auto', 'io_parallel', 'latency', None),  # Should end up with PIPELINE pragma
        (6, 'auto', 'io_parallel', 'resource_unrolled', None),  # Should end up with PIPELINE pragma and II
        (7, 'pipeline', 'io_stream', 'resource', None),  # Should result in a warning
        (8, 'pipeline', 'io_parallel', 'resource', None),  # Should result in a warning
        (9, 'pipeline', 'io_parallel', 'latency', None),  # No warning
        (10, 'pipeline', 'io_parallel', 'latency', 10),  # No warning, should include II=10
        (11, 'dataflow', 'io_stream', 'latency', None),  # No warning
        (12, 'dataflow', 'io_parallel', 'latency', None),  # No warning
        (13, 'dataflow', 'io_parallel', 'latency', None),  # No warning
        (14, 'wrong', 'io_parallel', 'latency', None),  # Incorrect settings should issue a warning and switch to 'auto'
        (15, 'auto', 'io_parallel', 'resource', None),  # Special case to test Conv layer. No warning
        (16, 'pipeline', 'io_parallel', 'resource', None),  # Special case to test Conv layer. Should result in two warnings
    ],
)
def test_pipeline_style(capfd, backend, param_group, pipeline_style, io_type, strategy, ii):
    def _check_top_hls_pragma(model, pragma, ii=None):
        assert model.config.pipeline_style == pragma

        pragma_to_check = f'#pragma HLS {pragma.upper()}'
        if ii is not None:
            pragma_to_check += f' II={ii}'

        with open(model.config.get_output_dir() + '/firmware/myproject.cpp') as main_file:
            contents = main_file.readlines()
            for line in contents:
                if pragma_to_check in line:
                    return True

        return False

    if param_group in [15, 16]:
        model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(8, 2, input_shape=(10, 4))])
    else:
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(8, input_shape=(10,))])

    config = hls4ml.utils.config_from_keras_model(model)
    if pipeline_style is not None:
        config['Model']['PipelineStyle'] = pipeline_style
    if ii is not None:
        config['Model']['PipelineInterval'] = ii
    config['Model']['Strategy'] = strategy
    config['Model']['ReuseFactor'] = 2

    prj_name = f'hls4mlprj_pipeline_style_{backend}_{param_group}'
    output_dir = str(test_root_path / prj_name)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.write()

    captured_warnings = [line for line in capfd.readouterr().out.split('\n') if line.startswith('WARNING')]

    if param_group in [1, 2, 3, 4]:
        assert _check_top_hls_pragma(hls_model, 'dataflow')
    elif param_group == 5:
        assert _check_top_hls_pragma(hls_model, 'pipeline')
    elif param_group == 6:
        assert _check_top_hls_pragma(hls_model, 'pipeline', ii=2)
    elif param_group in [7, 8]:
        assert _check_top_hls_pragma(hls_model, 'pipeline')
        assert any('bad QoR' in warning for warning in captured_warnings)
    elif param_group == 9:
        assert _check_top_hls_pragma(hls_model, 'pipeline')
        assert len(captured_warnings) == 0
    elif param_group == 10:
        assert _check_top_hls_pragma(hls_model, 'pipeline', ii=ii)
        assert len(captured_warnings) == 0
    elif param_group in [11, 12, 13]:
        assert _check_top_hls_pragma(hls_model, 'dataflow')
        assert len(captured_warnings) == 0
    elif param_group == 14:
        assert _check_top_hls_pragma(hls_model, 'pipeline')
        assert any('Using "auto"' in warning for warning in captured_warnings)
    elif param_group == 15:
        assert _check_top_hls_pragma(hls_model, 'dataflow')
    elif param_group == 16:
        assert _check_top_hls_pragma(hls_model, 'pipeline')
        assert any('bad QoR' in warning for warning in captured_warnings)
        assert any('Convolution' in warning for warning in captured_warnings)
