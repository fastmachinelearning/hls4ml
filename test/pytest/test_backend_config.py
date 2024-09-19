from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import torch

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('framework', ['keras', 'pytorch'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
@pytest.mark.parametrize('part', ['some_part', None])
@pytest.mark.parametrize('clock_period', [8, None])
@pytest.mark.parametrize('clock_unc', ['15%', None])
def test_backend_config(framework, backend, part, clock_period, clock_unc):
    if framework == 'keras':
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(
                2,
                input_shape=(1,),
                activation='relu',
            )
        )
        model.compile(optimizer='adam', loss='mse')

        config = hls4ml.utils.config_from_keras_model(model)
        convert_fn = hls4ml.converters.convert_from_keras_model
    else:
        model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.ReLU())
        config = hls4ml.utils.config_from_pytorch_model(model, input_shape=(None, 1))
        convert_fn = hls4ml.converters.convert_from_pytorch_model

    if clock_unc is not None:
        unc_str = clock_unc.replace('%', '')
    else:
        unc_str = clock_unc

    test_dir = f'hls4mlprj_backend_config_{framework}_{backend}_part_{part}_period_{clock_period}_unc_{unc_str}'
    output_dir = test_root_path / test_dir

    if framework == "keras":
        hls_model = convert_fn(
            model,
            input_shape=(None, 1),  # This serves as a test of handling unexpected values by the backend in keras converer
            hls_config=config,
            output_dir=str(output_dir),
            backend=backend,
            part=part,
            clock_period=clock_period,
            clock_uncertainty=clock_unc,
        )
    else:
        hls_model = convert_fn(
            model,
            hls_config=config,
            output_dir=str(output_dir),
            backend=backend,
            part=part,
            clock_period=clock_period,
            clock_uncertainty=clock_unc,
        )

    hls_model.write()

    # Check if config was properly parsed into the ModelGraph

    read_part = hls_model.config.get_config_value('Part')
    expected_part = part
    if backend in ['Vivado', 'Vitis'] and part is None:
        expected_part = 'xcvu13p-flga2577-2-e'
    elif backend == 'Quartus' and part is None:
        expected_part = 'Arria10'
    assert read_part == expected_part

    read_clock_period = hls_model.config.get_config_value('ClockPeriod')
    expected_period = clock_period if clock_period is not None else 5
    assert read_clock_period == expected_period

    expected_unc = clock_unc
    read_clock_unc = hls_model.config.get_config_value('ClockUncertainty')
    if backend == 'Vivado' and clock_unc is None:
        expected_unc = '12.5%'
    elif backend == 'Vitis' and clock_unc is None:
        expected_unc = '27%'
    elif backend in ['Vivado', 'Vitis']:
        expected_unc = clock_unc
    elif backend == 'Quartus':
        expected_unc = None
    assert read_clock_unc == expected_unc

    # Check if Writer properly wrote makefiles and tcl scripts
    if backend in ['Vivado', 'Vitis']:
        part_ok = period_ok = unc_ok = False

        prj_tcl_path = output_dir / 'project.tcl'
        with open(prj_tcl_path) as f:
            for line in f.readlines():
                if 'set part' in line and expected_part in line:
                    part_ok = True
                if f'set clock_period {expected_period}' in line:
                    period_ok = True
                if f'set clock_uncertainty {expected_unc}' in line:
                    unc_ok = True

        assert part_ok and period_ok and unc_ok
    elif backend == 'Quartus':
        part_ok = period_ok = False

        makefile_path = output_dir / 'Makefile'
        with open(makefile_path) as f:
            for line in f.readlines():
                if 'DEVICE   :=' in line and expected_part in line:
                    part_ok = True

        main_cpp_path = output_dir / f'firmware/{hls_model.config.get_project_name()}.cpp'
        with open(main_cpp_path) as f:
            clock_mhz = 1000 / (hls_model.config.get_config_value('ClockPeriod'))
            clock_mhz = np.ceil(clock_mhz).astype(int)
            for line in f.readlines():
                if f'hls_scheduler_target_fmax_mhz({clock_mhz})' in line:
                    period_ok = True

        assert part_ok and period_ok
