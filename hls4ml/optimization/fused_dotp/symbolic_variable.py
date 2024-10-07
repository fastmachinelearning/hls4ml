from functools import singledispatchmethod
from numbers import Integral

import numpy as np

from .config import VariableOverrideContextManager, _global_config
from .precision import FixedPointPrecision, float_gcd


class fuse_associative_ops(VariableOverrideContextManager):
    target = 'fuse_associative_ops'


class backend(VariableOverrideContextManager):
    target = 'backend'


class order_metrics(VariableOverrideContextManager):
    target = 'order_metrics'


class VariableBase:
    def __new__(cls, *args, **kwargs):
        if cls is VariableBase:
            raise TypeError('VariableBase should not be instantiated directly')
        return super().__new__(cls)


class Variable(VariableBase):
    def __init__(
        self,
        precision: FixedPointPrecision,
        ancestors: tuple['Variable', ...] = (),
        operation: str = 'new',
        const: float | int = 0,
        id: str | None = None,
        depth=0,
        n_depth=0,
    ):
        """
        precision: precision of the variable. If it is a number, the Variable will define a constant.
        ancestors: ancestors of the variable. Defaults to ().
        operation: operation to create the variable. Defaults to 'origin'. Available operations are:
            - 'new': variable pop up from the thin air. Must have no ancestors
            - 'add': addition.
            - 'sub': substraction. Must have exactly two ancestors.
            - 'mul': multiplication.
            - 'shift': bit shift. Must have exactly one ancestor.
            - 'max': maximum value of the ancestors. Must have at least one ancestor.
            - 'cnst': constant value. Must have no ancestors.
        const: constant parameter used in some operations. Defaults to 0.
            - 'add': sum(ancestors) + const
            - 'mul': product(ancestors) * const
            - 'shift': ancestors[0] << const
            - 'max': max(ancestors[0], const)
            - 'cnst': = constant
        id: unique identifier of the variable. Defaults to None

        When there is exactly one ancestor, params field must present the value of the operation.
        """
        self.precision = precision + 0
        self.ancestors = ancestors
        self.operation = operation
        self.id = id
        self.const = const
        self.children: tuple[Variable, ...] = ()
        self.depth = depth
        self.n_depth = n_depth

        self._proper_precision = False

    def fix_precision(self, recursive=False):
        if not self._proper_precision:
            self.precision = self.precision.make_proper()
            self._proper_precision = True
        if recursive:
            for ancestor in self.ancestors:
                ancestor.fix_precision(recursive=True)

    @property
    def k(self) -> bool:
        'If the variable is signed.'
        return self.precision.k

    @property
    def i(self):
        'The number of bits in the integer part of the variable, excluding the sign bit.'
        if self._proper_precision:
            return self.precision.i
        return self.precision.make_proper().i

    @property
    def f(self):
        'The number of bits in the fractional part of the variable.'
        if self._proper_precision:
            return self.precision.f
        return self.precision.make_proper().f

    @property
    def b(self):
        'The number of bits in the variable.'
        if self._proper_precision:
            return self.precision.b
        return self.precision.make_proper().b

    def __repr__(self):
        if self._proper_precision:
            precision = self.precision
        else:
            precision = self.precision.make_proper()
        if self.id is not None:
            return f'{precision} {self.id} (depth={self.depth})'
        return f'{precision}: (opr={self.operation}, #a={len(self.ancestors)}, c={self.const})'

    @singledispatchmethod
    def __add__(self, other) -> 'Variable':
        operation = 'add'
        if other == 0:
            return self
        if self.operation == 'add':
            ancestors = self.ancestors
            const = self.const + other
        else:
            ancestors = (self,)
            const = other

        precision = self.precision + other
        return Variable(precision, ancestors, operation, const, n_depth=self.n_depth, depth=self.depth)

    @__add__.register(VariableBase)
    def _(self, other: 'Variable'):

        precision = self.precision + other.precision

        if _global_config.fuse_associative_ops:
            if other.operation == 'add':
                if self.operation == 'add':
                    ancestors = self.ancestors + other.ancestors
                    const = self.const + other.const
                    depth = max(x.depth for x in ancestors) + 1
                    return Variable(precision, ancestors, 'add', const, depth=depth)
                else:
                    return other + self
            if self.operation == 'add':
                ancestors = self.ancestors + (other,)
                const = self.const
                depth = max(max(x.depth for x in ancestors), other.depth) + 1
                return Variable(precision, ancestors, 'add', const, depth=self.depth + 1)

        if self.operation == other.operation == 'shift':
            if self.const == other.const:
                return (self.ancestors[0] + other.ancestors[0]) << self.const

        ancestors = (self, other)
        const = 0
        p1, p2 = self.precision, other.precision
        I1, I2 = p1.I, p2.I
        f1, f2 = p1.f, p2.f
        ddepth = max(I1, I2) + max(f1, f2)
        n_depth = max(self.n_depth, other.n_depth) + 1
        depth = max(self.depth, other.depth) + ddepth
        return Variable(precision, ancestors, 'add', const, depth=depth, n_depth=n_depth)

    @singledispatchmethod
    def __mul__(self, other) -> 'Variable|float|int':
        if other == 1:
            return self
        if other == 0:
            return 0
        if other == -1:
            return -self

        if np.log2(abs(other)) % 1 == 0:
            shift = int(np.log2(abs(other)))
            if other < 0:
                return (-self) << shift
            return self << shift

        if self.operation == 'mul':
            ancestors = self.ancestors
            const = self.const * other
        else:
            ancestors = (self,)
            const = other

        precision = self.precision * other
        if isinstance(precision, FixedPointPrecision):
            return Variable(precision, ancestors, 'mul', const, depth=self.depth + 1)
        else:
            value = precision
            return value

    @__mul__.register(VariableBase)
    def _(self, other: 'Variable'):

        if other.operation == 'neg':
            if self.operation == 'neg':
                return self.ancestors[0] * other.ancestors[0]
            return -(self.ancestors[0] * other.ancestors[0])

        precision = self.precision * other.precision
        assert isinstance(precision, FixedPointPrecision)

        if _global_config.fuse_associative_ops:
            if other.operation == 'mul':
                if self.operation == 'mul':
                    ancestors = self.ancestors + other.ancestors
                    const = self.const * other.const
                    depth = max(x.depth for x in ancestors) + 1
                    return Variable(precision, ancestors, 'mul', const, depth=depth)
                else:
                    return other * self
            if self.operation == 'mul':
                ancestors = self.ancestors + (other,)
                const = self.const
                depth = max(max(x.depth for x in ancestors), other.depth) + 1
                return Variable(precision, ancestors, 'mul', const, depth=depth)

        if self.operation == other.operation == 'shift':
            return (self.ancestors[0] * other.ancestors[0]) << (self.const + other.const)

        const = 1
        ancestors = (self, other)
        depth = max(self.depth, other.depth) + 1
        if isinstance(precision, FixedPointPrecision):
            return Variable(precision, ancestors, 'mul', const, depth=depth)
        else:
            value = precision
            return value

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self) -> 'Variable':
        if self.operation == 'neg':
            return self.ancestors[0]
        return Variable(-self.precision, (self,), 'neg', depth=self.depth + self.precision.b, n_depth=self.n_depth + 1)

    def __sub__(self, other) -> 'Variable':
        # if not isinstance(other, Variable):
        return self + (-other)
        # depth = max(self.depth, other.depth) + 1
        # return Variable(self.precision - other.precision, (self, other), 'sub', depth=depth)

    def __rsub__(self, other) -> 'Variable':
        return -self + other

    def __truediv__(self, other) -> 'Variable|float|int':
        assert not isinstance(other, Variable), 'Division is not supported between variables'
        return self * (1 / other)

    def __lshift__(self, other) -> 'Variable':
        operation = 'shift'
        assert isinstance(other, Integral), 'Shift can only be done with an integer'
        other = int(other)
        if other == 0:
            return self

        if self.operation == 'shift':
            ancestors = self.ancestors
            const = self.const + other
        else:
            ancestors = (self,)
            const = other
        if const == 0:
            return ancestors[0]  # type: ignore

        return Variable(self.precision << other, ancestors, operation, const, depth=self.depth)

    def __rshift__(self, other):
        return self.__lshift__(-other)

    def __le__(self, other):
        metrics = _global_config.order_metrics
        assert metrics != (), 'Order metric is not set in config'
        for metric in metrics:
            sm, om = getattr(self, metric), getattr(other, metric)
            if sm > om:
                return False
            if sm < om:
                return True
        return True

    def __lt__(self, other):
        metrics = _global_config.order_metrics
        assert metrics != (), 'Order metric is not set in config'
        for metric in metrics:
            sm, om = getattr(self, metric), getattr(other, metric)
            if sm > om:
                return False
            if sm < om:
                return True
        return False

    def __ge__(self, other):
        return not self < other

    def __gt__(self, other):
        return not self <= other


def take_max(*args: Variable | float | int):
    precisions = [arg.precision for arg in args if isinstance(arg, Variable)]
    const = max([arg for arg in args if not isinstance(arg, Variable)] + [-float('inf')])
    v_min = max([precision.min for precision in precisions])
    v_max = max([precision.max for precision in precisions])
    _min, _max = max(v_min, const), max(v_max, const)
    # delta = math.gcd(*[round(precision.delta * 1e10) for precision in precisions if precision.max > _min]) * 1e-10
    delta = float_gcd(*[precision.delta for precision in precisions if precision.max > _min])

    if _min == _max:
        return _min

    const = const if const > v_min else -float('inf')

    ancestors = tuple(arg for arg in args if isinstance(arg, Variable) and arg.precision.max > _min)
    partition = (_max - _min) / delta + 1
    precision = FixedPointPrecision(_min, _max, partition)
    return Variable(precision, ancestors, 'max', const)
