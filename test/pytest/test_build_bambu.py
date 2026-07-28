import os
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

# -----------------------------------------------------------------------------
# fixtures / helpers
# -----------------------------------------------------------------------------


@pytest.fixture(scope='module')
def simple_model():
    """Simple Keras model for build testing"""
    model = Sequential()
    model.add(Dense(8, input_shape=(2,)))
    return model


def count_files_with_extension(directory, extension):
    """Counts how many files in the directory and all its
    subdirectories have files that end with extension"""
    return sum(1 for _ in Path(directory).rglob(f'*{extension}'))


# -----------------------------------------------------------------------------
# tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('backend', ['Vitis', 'Bambu'])
def test_csimulation(test_case_id, simple_model, tmp_path, io_type, strategy, granularity, batch_size, backend):
    output_dir = str(test_root_path / test_case_id)

    model = simple_model
    X_input = np.random.rand(batch_size, 2).astype(np.float32)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()
    y_pred = hls_model.predict(X_input)

    input_data_tb = str(tmp_path / 'input.npy')
    output_data_tb = str(tmp_path / 'output.npy')
    np.save(input_data_tb, X_input)
    np.save(output_data_tb, y_pred)

    hls_model_csim = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        io_type=io_type,
        backend=backend,
        input_data_tb=input_data_tb,
        output_data_tb=output_data_tb,
    )
    hls_model_csim.compile()
    hls_model_csim.build(synth=True, csim=True, log_to_stdout=True)

    bridge_result = np.loadtxt(os.path.join(output_dir, 'tb_data', 'tb_output_predictions.dat'))
    csim_result = np.loadtxt(os.path.join(output_dir, 'tb_data', 'csim_results.log'))
    assert np.allclose(bridge_result, csim_result, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('backend', ['Vitis', 'Bambu'])
@pytest.mark.parametrize('part', ['xc7a100tcsg324-1'])
def test_cosimulation(test_case_id, simple_model, tmp_path, io_type, strategy, granularity, batch_size, backend, part):
    output_dir = str(test_root_path / test_case_id)

    model = simple_model
    X_input = np.random.rand(batch_size, 2).astype(np.float32)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        io_type=io_type,
        backend=backend,
        part=part,
        clock_period=20,
    )
    hls_model.compile()
    y_pred = hls_model.predict(X_input)

    input_data_tb = str(os.path.join(output_dir, 'input.npy'))
    output_data_tb = str(os.path.join(output_dir, 'output.npy'))
    np.save(input_data_tb, X_input)
    np.save(output_data_tb, y_pred)

    hls_model_cosim = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        io_type=io_type,
        backend=backend,
        input_data_tb=input_data_tb,
        output_data_tb=output_data_tb,
        part=part,
        clock_period=20,
    )
    hls_model_cosim.compile()
    hls_model_cosim.build(csim=False, synth=True, cosim=True, log_to_stdout=True)

    bridge_result = np.loadtxt(os.path.join(output_dir, 'tb_data', 'tb_output_predictions.dat'))
    cosim_result = np.loadtxt(os.path.join(output_dir, 'tb_data', 'rtl_cosim_results.log'))
    assert np.allclose(bridge_result, cosim_result, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('backend', ['Vitis', 'Bambu'])
def test_synth(test_case_id, simple_model, io_type, strategy, granularity, backend):
    """Test that a successful synth run produces the desired artifacts (.v file)"""
    synth_proj_dir = test_root_path / test_case_id

    model = simple_model

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(synth_proj_dir),
        io_type=io_type,
        backend=backend,
    )
    hls_model.build(csim=False, synth=True)

    # Bambu-specific artifact checks
    if backend == 'Bambu':
        proj_name = hls_model.config.get_project_name()
        assert Path(synth_proj_dir, f'{proj_name}.v').exists()

    # TODO: Vitis-specific artifact checks


@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('backend', ['Vitis', 'Bambu'])
@pytest.mark.parametrize('part', ['xc7a100tcsg324-1'])
def test_vsynth(test_case_id, simple_model, io_type, strategy, granularity, backend, part):
    """Test that a successful vsynth run produces the desired reports.

    `part` stays parametrized so a target can be added, but the sweep is the
    backend's own default 7-Series Artix. NanoXplore targets route into
    family-specific branches and belong with the accelerator-layer tests.
    """
    vsynth_proj_dir = test_root_path / test_case_id

    model = simple_model

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(vsynth_proj_dir),
        io_type=io_type,
        backend=backend,
        part=part,
        clock_period=25,
    )
    hls_model.build(csim=False, synth=True, cosim=True, vsynth=True)

    # Bambu-specific artifact checks
    if backend == 'Bambu':
        # Ensure we get bambu results file. Older Bambu versions produced
        # `bambu_results_<flow>.xml`; current versions produce
        # `bambu_results.xml` — match both.
        assert sum(1 for _ in vsynth_proj_dir.rglob('bambu_results*.xml')) >= 1

        # Ensure we get expected reports. The .rpt files come from the Vivado
        # synthesis Bambu drives; without Vivado on PATH (e.g. the Bambu-only CI
        # image) none are produced, so the check is only meaningful there.
        if hls_model.config.get_config_value('FPGAFamily') == 'Xilinx':
            num_reports = count_files_with_extension(vsynth_proj_dir / 'HLS_output', '.rpt')
            if num_reports == 0:
                pytest.skip('No Vivado reports produced (Vivado not available in this environment)')
            assert num_reports >= 15

    # TODO: Vitis-specific artifact checks


@pytest.mark.parametrize('backend', ['Bambu'])
def test_build_failure_is_reported(test_case_id, simple_model, monkeypatch, backend):
    """A non-zero Bambu exit must raise instead of falling through to report parsing.

    Without the check, a failed run returns an empty result (or, with ``reset=False``,
    a report left by a previous run) and ``build()`` looks like it succeeded.
    """
    output_dir = str(test_root_path / test_case_id)

    config = hls4ml.utils.config_from_keras_model(simple_model, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        simple_model, hls_config=config, output_dir=output_dir, io_type='io_parallel', backend=backend
    )

    class FailingProcess:
        returncode = 3

        def communicate(self):
            return ('', '')

    # Stand in for a Bambu installation that is present but exits non-zero.
    backend_module = hls4ml.backends.bambu.bambu_backend
    monkeypatch.setattr(backend_module.shutil, 'which', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(backend_module.subprocess, 'Popen', lambda *args, **kwargs: FailingProcess())

    with pytest.raises(RuntimeError, match='exit code 3'):
        hls_model.build(csim=False, synth=True)
