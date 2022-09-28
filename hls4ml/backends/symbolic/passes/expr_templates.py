import re

from hls4ml.model.layers import SymbolicExpression
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

from sympy.printing.cxx import CXX11CodePrinter

# Expression templates

expr_function_template = 'y[0] = {expr_str};'

expr_include_list = ['hls_math.h']

class HLSCodePrinter(CXX11CodePrinter):
    _ns = 'hls::'

    def _print_Float(self, flt):
        float_str = super()._print_Float(flt)
        return f'model_default_t({float_str})'

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        return re.sub(r'([a-zA-Z]+)(\d+)', r'\1[\2]', name)

class ExpressionFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SymbolicExpression, include_header=expr_include_list)
        self.template = expr_function_template
    
    def format(self, node):
        params = self._default_function_params(node)
        params['expr_str'] = HLSCodePrinter().doprint(node.attributes['expression'])
        return self.template.format(**params)
