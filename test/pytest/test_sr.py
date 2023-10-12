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
