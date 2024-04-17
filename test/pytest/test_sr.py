from pathlib import Path

import numpy as np
import pytest

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def data():
    X = 2 * np.random.rand(100, 5)
    y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

    return X, y


def test_hlssr(data):
    expr = 'x0**2 + 2.5382*cos_lut(x3) - 0.5'

    lut_functions = {'cos_lut': {'math_func': 'cos', 'range_start': -4, 'range_end': 4, 'table_size': 2048}}

    output_dir = str(test_root_path / 'hls4mlprj_sr')

    hls_model = hls4ml.converters.convert_from_symbolic_expression(
        expr,
        n_symbols=5,
        precision='ap_fixed<18,6>',
        output_dir=output_dir,
        lut_functions=lut_functions,
        hls_include_path='',
        hls_libs_path='',
    )
    hls_model.write()
    hls_model.compile()

    X, y = data
    y_hls = hls_model.predict(X)
    y_hls = y_hls.reshape(y.shape)

    np.testing.assert_allclose(y, y_hls, rtol=1e-2, atol=1e-2, verbose=True)


def test_pysr_luts(data):
    try:
        from pysr import PySRRegressor
    except ImportError:
        pytest.skip('Failed to import PySR, test will be skipped.')

    function_definitions = ['cos_lut(x) = math_lut(cos, x, N=1024, range_start=-4, range_end=4)']
    hls4ml.utils.symbolic_utils.init_pysr_lut_functions(init_defaults=True, function_definitions=function_definitions)

    model = PySRRegressor(
        model_selection='best',  # Result is mix of simplicity+accuracy
        niterations=10,
        binary_operators=['+', '*'],
        unary_operators=[
            'cos_lut',
        ],
        loss='loss(x, y) = (x - y)^2',
        temp_equation_file=True,
    )

    X, y = data

    model.fit(X, y)

    eq = str(model.sympy())

    assert 'cos_lut' in eq


@pytest.mark.parametrize('part', ['some_part', None])
@pytest.mark.parametrize('clock_period', [8, None])
@pytest.mark.parametrize('clock_unc', ['15%', None])
@pytest.mark.parametrize('compiler', ['vivado_hls', 'vitis_hls'])
def test_sr_backend_config(part, clock_period, clock_unc, compiler):

    expr = 'x0**2 + 2.5382*cos_lut(x3) - 0.5'

    if clock_unc is not None:
        unc_str = clock_unc.replace('%', '')
    else:
        unc_str = clock_unc

    compiler_str = compiler.replace('_hls', '')

    test_dir = f'hls4mlprj_sr_backend_config_part_{part}_period_{clock_period}_unc_{unc_str}_{compiler_str}'
    output_dir = test_root_path / test_dir

    hls_model = hls4ml.converters.convert_from_symbolic_expression(
        expr,
        n_symbols=5,
        precision='ap_fixed<18,6>',
        output_dir=str(output_dir),
        part=part,
        clock_period=clock_period,
        clock_uncertainty=clock_unc,
        compiler=compiler,
        hls_include_path='',
        hls_libs_path='',
    )
    hls_model.write()

    # Check if config was properly parsed into the ModelGraph

    read_part = hls_model.config.get_config_value('Part')
    expected_part = part if part is not None else 'xcvu13p-flga2577-2-e'
    assert read_part == expected_part

    read_clock_period = hls_model.config.get_config_value('ClockPeriod')
    expected_period = clock_period if clock_period is not None else 5
    assert read_clock_period == expected_period

    read_clock_unc = hls_model.config.get_config_value('ClockUncertainty')
    expected_unc = clock_unc
    if expected_unc is None:
        if compiler == 'vivado_hls':
            expected_unc = '12.5%'
        else:
            expected_unc = '27%'
    assert read_clock_unc == expected_unc

    # Check if Writer properly wrote tcl scripts
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
