import subprocess
import tempfile

math_lut_julia = """
function math_lut(fun::Function, x::Float32 ; N::Integer = 1024, range_start::Real = 0, range_end::Real = 8)
    range = range_end - range_start
    step = range / N
    idx = round((x - range_start) * N / range)

    if idx < 0
        idx = 0
    elseif idx > N - 1
        idx = N - 1
    end

    x_approx = range_start + step * idx
    return Float32(fun(x_approx))
end
"""


def init_pysr_lut_functions(init_defaults=False, function_definitions=None):
    """Register LUT-based approximations with PySR.

    Functions should be in the form of::

        <func_name>(x) = math_lut(<func>, x, N=<table_size>, range_start=<start>, range_end=<end>)

    where ``<func_name>`` is a given name that can be used with PySR, ``<func>`` is the math function to approximate
    (`sin`, `cos`, `log`,...), ``<table_size>`` is the size of the lookup table, and ``<start>`` and ``<end>`` are the
    ranges in which the function will be approximated. It is **strongly** recommended to use a power-of-two as a range.

    Registered functions can be passed by name to ``PySRRegressor`` (as ``unary_operators``).

    Args:
        init_defaults (bool, optional): Register the most frequently used functions (sin, cos, tan, log, exp).
            Defaults to False.
        function_definitions (list, optional): List of strings with function definitions to register with PySR.
            Defaults to None.
    """
    from pysr.julia_helpers import init_julia

    Main = init_julia()
    Main.eval(math_lut_julia)

    if init_defaults:
        Main.eval('sin_lut(x) = math_lut(sin, x, N=1024, range_start=-4, range_end=4)')
        Main.eval('cos_lut(x) = math_lut(cos, x, N=1024, range_start=-4, range_end=4)')
        Main.eval('tan_lut(x) = math_lut(tan, x, N=1024, range_start=-4, range_end=4)')

        Main.eval('log_lut(x) = math_lut(log, x, N=1024, range_start=0, range_end=8)')
        Main.eval('exp_lut(x) = math_lut(exp, x, N=1024, range_start=0, range_end=16)')

    for func in function_definitions or []:
        register_pysr_lut_function(func, Main)


def register_pysr_lut_function(func, julia_main=None):
    if julia_main is None:
        from pysr.julia_helpers import init_julia

        Main = init_julia()
    else:
        Main = julia_main

    Main.eval(func)


class LUTFunction:
    def __init__(self, name, math_func, range_start, range_end, table_size=1024) -> None:
        self.name = name
        self.math_func = math_func
        self.range_start = range_start
        self.range_end = range_end
        self.table_size = table_size


_binary_ops = {'/': 'x / y', '*': 'x * y', '+': 'x + y', '-': 'x - y', 'pow': 'x**y', 'pow_abs': 'Abs(x) ** y'}


_unary_ops = {
    'abs': 'Abs',
    'mod': 'sympy.Mod(x, 2)',
    'erf': 'sympy.erf',
    'erfc': 'sympy.erfc',
    'log': 'sympy.log(x)',
    'log10': 'sympy.log(x, 10)',
    'log2': 'sympy.log(x, 2)',
    'log1p': 'sympy.log(x + 1)',
    'log_abs': 'sympy.log(Abs(x))',
    'log10_abs': 'sympy.log(Abs(x), 10)',
    'log2_abs': 'sympy.log(Abs(x), 2)',
    'log1p_abs': 'sympy.log(Abs(x) + 1)',
    'floor': 'sympy.floor',
    'ceil': 'sympy.ceiling',
    'sqrt': 'sympy.sqrt(x)',
    'sqrt_abs': 'sympy.sqrt(Abs(x))',
    'square': 'x**2',
    'cube': 'x**3',
    'neg': '-x',
    'cos': 'sympy.cos',
    'sin': 'sympy.sin',
    'tan': 'sympy.tan',
    'cosh': 'sympy.cosh',
    'sinh': 'sympy.sinh',
    'tanh': 'sympy.tanh',
    'exp': 'sympy.exp',
    'acos': 'sympy.acos',
    'asin': 'sympy.asin',
    'atan': 'sympy.atan',
    'acosh': 'sympy.acosh(x)',
    'acosh_abs': 'sympy.acosh(Abs(x) + 1)',
    'asinh': 'sympy.asinh',
    'atanh': 'sympy.atanh(sympy.Mod(x + 1, 2) - 1)',
    'atanh_clip': 'sympy.atanh(sympy.Mod(x + 1, 2) - 1)',
    'sign': 'sympy.sign',
}


def generate_operator_complexity(
    part, precision, unary_operators=None, binary_operators=None, hls_include_path=None, hls_libs_path=None
):
    """Generates HLS projects and synthesizes them to obtain operator complexity (clock cycles per given math operation).

    This function can be used to obtain a list of operator complexity for a given FPGA part at a given precision.

    Args:
        part (str): FPGA part number to use.
        precision (str): Precision to use.
        unary_operators (list, optional): List of unary operators to evaluate. Defaults to None.
        binary_operators (list, optional): List of binary operators to evaluate. Defaults to None.
        hls_include_path (str, optional): Path to the HLS include files. Defaults to None.
        hls_libs_path (str, optional): Path to the HLS libs. Defaults to None.

    Returns:
        dict: Dictionary of obtained operator complexities.
    """

    from sympy.parsing.sympy_parser import parse_expr as parse_sympy_expr

    from hls4ml.converters import convert_from_symbolic_expression

    if unary_operators is None:
        unary_ops = _unary_ops
    else:
        unary_ops = {fn_name: sympy_expr for fn_name, sympy_expr in _unary_ops.items() if fn_name in unary_operators}
    if binary_operators is None:
        binary_ops = _binary_ops
    else:
        binary_ops = {fn_name: sympy_expr for fn_name, sympy_expr in _binary_ops.items() if fn_name in binary_operators}

    complexity_of_operators = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for fn_name, sympy_expr in binary_ops.items():
            print(f'Estimating complexity of {fn_name}')
            equation = sympy_expr.replace('x', 'x0').replace('y', 'x1')
            expression = parse_sympy_expr(equation)
            hls_model = convert_from_symbolic_expression(
                expression,
                n_symbols=2,
                output_dir=tmp_dir,
                precision=precision,
                part=part,
                hls_include_path=hls_include_path,
                hls_libs_path=hls_libs_path,
            )
            hls_model.write()
            subprocess.run(
                ['vivado_hls', '-f', 'build_prj.tcl', '"reset=1 synth=1 csim=0 cosim=0 validation=0 export=0"'],
                cwd=tmp_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            result = subprocess.check_output(
                ['awk', 'NR==32', 'myproject_prj/solution1/syn/report/myproject_csynth.rpt'], cwd=tmp_dir
            )
            cc = result.decode('utf-8').replace(' ', '').split('|')[1]
            complexity_of_operators[fn_name] = max(int(cc), 1)

        for fn_name, sympy_expr in unary_ops.items():
            print(f'Estimating complexity of {fn_name}')
            equation = sympy_expr.replace('sympy.', '')
            if 'x' in equation and fn_name != 'exp':
                equation = equation.replace('x', 'x0')
            else:
                equation += '(x0)'
            expression = parse_sympy_expr(equation)
            hls_model = convert_from_symbolic_expression(
                expression,
                n_symbols=1,
                output_dir=tmp_dir,
                precision=precision,
                part=part,
                hls_include_path=hls_include_path,
                hls_libs_path=hls_libs_path,
            )
            hls_model.write()
            subprocess.run(
                ['vivado_hls', '-f', 'build_prj.tcl', '"reset=1 synth=1 csim=0 cosim=0 validation=0 export=0"'],
                cwd=tmp_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            result = subprocess.check_output(
                ['awk', 'NR==32', 'myproject_prj/solution1/syn/report/myproject_csynth.rpt'], cwd=tmp_dir
            )
            cc = result.decode('utf-8').replace(' ', '').split('|')[1]
            complexity_of_operators[fn_name] = max(int(cc), 1)

    return complexity_of_operators
