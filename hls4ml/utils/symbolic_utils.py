
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
    from pysr.julia_helpers import init_julia

    Main = init_julia()
    Main.eval(math_lut_julia)

    if init_defaults:
        Main.eval('sin_lut(x) = math_lut(sin, x, N=1024, range_start=-4, range_end=4)')
        Main.eval('cos_lut(x) = math_lut(cos, x, N=1024, range_start=-4, range_end=4)')
        Main.eval('tan_lut(x) = math_lut(tan, x, N=1024, range_start=-4, range_end=4)')

        Main.eval('log_lut(x) = math_lut(log, x, N=1024, range_start=0, range_end=8)')
        Main.eval('exp_lut(x) = math_lut(exp, x, N=1024, range_start=-8, range_end=8)')
    
    for func in function_definitions or []:
        register_pysr_lut_function(func, Main)

def register_pysr_lut_function(func, julia_main=None):
    if julia_main is None:
        from pysr.julia_helpers import init_julia
        Main = init_julia()
    else:
        Main = julia_main
    
    Main.eval(func)


class LUTFunction(object):
    def __init__(self, name, math_func, range_start, range_end, table_size=1024) -> None:
        self.name = name
        self.math_func = math_func
        self.range_start = range_start
        self.range_end = range_end
        self.table_size = table_size