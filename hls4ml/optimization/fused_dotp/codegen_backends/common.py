from ..precision import FixedPointPrecision
from ..symbolic_variable import Variable
from ..utils import precision_from_const


class CodegenBackend:
    @staticmethod
    def type(precision: FixedPointPrecision) -> str: ...

    def to_opr_code(self, var: Variable) -> str:
        return getattr(self, var.operation)(var)

    def neg(self, var: Variable) -> str:
        return f'-{var.ancestors[0].id}'

    def add(self, var: Variable) -> str:
        line: str = ' + '.join([v.id for v in var.ancestors])  # type: ignore
        if var.const != 0:
            const_type = self.type_str_for_const(var.const)
            line += f' + ({const_type}){var.const}'
        return line

    def sub(self, var: Variable) -> str:
        assert len(var.ancestors) == 2
        assert var.const == 0
        return f'{var.ancestors[0].id} - {var.ancestors[1].id}'

    def mul(self, var: Variable) -> str:
        line: str = ' * '.join([v.id for v in var.ancestors])  # type: ignore
        if var.const != 1:
            const_type = self.type_str_for_const(var.const)
            line += f' * ({const_type}){var.const}'
        return line

    def shift(self, var: Variable) -> str:
        return f'bit_shift<{var.const}>({var.ancestors[0].id})'

    def const(self, var: Variable) -> str:
        return f'({self.type(var.precision)}){var.const}'

    def type_str_for_const(self, number: float) -> str:
        precision = precision_from_const(number)
        return self.type(precision)

    def line_code(self, var: Variable) -> str | None:
        if var.operation == 'new':
            return None
        typing = self.type(var.precision)
        name = var.id
        opr_code = self.to_opr_code(var)
        return f'{typing} {name} = {opr_code};'


class Namer:
    def __init__(self):
        self.names = dict()

    def __call__(self, name: str):
        if name == 'const':
            name = '_const'  # avoid conflict with C++ keyword
        if name in self.names:
            self.names[name] += 1
            return f'{name}_{self.names[name]}'
        else:
            self.names[name] = 0
            return name


def _codegen(variable: Variable, realized: set[Variable], lines: list, namer: Namer, backend: CodegenBackend):
    """Recursive codegen function. Generates code for a variable and its ancestors.
    Args:
        variable (Variable): The variable to generate code for.
        realized (set[Variable]): The set of variables that have already been generated.
        lines (list): A list of lines of generated code, to which the generated code will be appended.
        namer (Namer): The namer to use for generating unique variable names.
        backend (CodegenBackend): The backend to use.
    """
    if variable in realized:
        return
    for a in variable.ancestors:
        if a not in realized:
            _codegen(a, realized, lines, namer, backend)
    if variable.id is None:
        variable.id = namer(variable.operation)
    line = backend.line_code(variable)
    if line is not None:
        lines.append(line)
    realized.add(variable)


def code_gen(variables: list, backend: CodegenBackend, out_vec_name: str | None = None):
    """Generate code that produces a list of variables. If out_vec_name is not None, the variables will be assigned to the elements of the array with that name.

    Args:
        variables (list): The list of variables to generate code for.
        backend (Backend): The backend to use.
        out_vec_name (str|None): The name of the output array. Defaults to None. If defined, the variables will be assigned to the elements of the array with that name.

    """  # noqa: E501
    lines = []
    realized = set()
    namer = Namer()
    for v in variables:
        if not isinstance(v, Variable):
            continue
        _codegen(v, realized, lines, namer, backend)
    if out_vec_name is not None:
        for i, v in enumerate(variables):
            if isinstance(v, Variable):
                line = f'{out_vec_name}[{i}] = {v.id};'
            else:
                type_str = backend.type_str_for_const(v)
                line = f'{out_vec_name}[{i}] = ({type_str}){v};'
            lines.append(line)
    return lines
