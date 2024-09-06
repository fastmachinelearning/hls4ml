import math
from typing import overload

import numpy as np


def const_fp_bits(*x):
    "Number of fp bits needed to represent x exactly."
    x = np.array(x, dtype=np.float32)
    l, h = -25, 25
    while l < h:  # noqa: E741
        v = (l + h) // 2
        _v = x * 2.0**v
        if np.all(np.round(_v) == _v):
            h = v
        else:
            l = v + 1  # noqa: E741
    return l


def float_gcd(*args):
    e = 0
    while any(arg % 1 for arg in args):
        args = tuple(arg * 10 for arg in args)
        e += 1
    r = math.gcd(*(int(arg) for arg in args))
    return r / 10**e


class PrecisionBase:
    def __init__(self, min: float, max: float, partition: int | float):
        assert np.all(min < max), f'{min} > {max}'
        assert np.all(partition > 0), f'{partition} <= 0'
        self._minmax = np.array([min, max])
        self._partition = partition

    @classmethod
    def cast_from(cls, other: 'PrecisionBase'):
        return cls(other.min, other.max, other.partition)

    @property
    def min(self):
        return self._minmax[0]

    @property
    def max(self):
        return self._minmax[1]

    @property
    def minmax(self):
        return self._minmax.copy()

    @property
    def partition(self):
        return self._partition

    @property
    def delta(self):
        return (self.max - self.min) / (self.partition - 1)

    @classmethod
    def from_kif(cls, k: bool | int, i: int, f: int, sym=False):
        "keep_negative, integer_bits, fractional_bits"
        max = 2.0**i - 2.0**-f
        if sym:
            min = -max
        else:
            min = -(k * 2.0**i)
        partition = 2 ** (k + i + f) - sym
        if partition == 1:
            assert min == max, f'{min} != {max} when partition == 1'
            return min
        return cls(min, max, partition)

    @classmethod
    def from_kbi(cls, k: bool | int, b: int, i: int, sym=False):
        "keep_negative, total_bits, integer_bits"
        i, f = i - k, b - i
        return cls.from_kif(k, i, f, sym)

    def copy(self):
        return self.__class__(self.min, self.max, self.partition)


class Precision(PrecisionBase):
    """Type of a variable that can only take values on `np.linspae(min, max, partition)`.
    Supports basic arithmetic operations.
    """

    def __str__(self):
        return f'Precision(min={self.min}, max={self.max}, partition={self.partition})'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):  # type: ignore
        if not isinstance(other, PrecisionBase):
            return False
        return np.all(self.minmax == other.minmax) and self.partition == other.partition

    def __mul__(self, other):
        if not isinstance(other, PrecisionBase):
            _min, _max = np.sort(self.minmax * other)
            partition = self.partition
        else:
            values = np.outer(self._minmax, other._minmax)
            _min, _max = np.min(values), np.max(values)

            delta = self.delta * other.delta
            partition = round((_max - _min) / delta + 1)
        if _min == _max:
            return float(_min)  # when min == max, the number can only be a single value

        return self.__class__(_min, _max, partition)

    def __add__(self, other):
        if not isinstance(other, PrecisionBase):
            _min, _max = self.minmax + other
            return self.__class__(_min, _max, self.partition)

        delta = float_gcd(self.delta, other.delta)
        # delta = math.gcd(round(self.delta * 2**24), int(other.delta * 2**24)) * 2.**-24
        _min, _max = self.minmax + other.minmax
        partition = (_max - _min) / delta + 1

        if abs(partition - round(partition)) > 1e-1:
            raise ValueError(
                f'Partition is not an integer: {partition}, {_min}, {_max}, {np.log2(self.delta)}, {np.log2(other.delta)}'
            )
        else:
            partition = round(partition)
        return self.__class__(_min, _max, partition)

    def __neg__(self):
        return self.__class__(-self.max, -self.min, self.partition)

    def __truediv__(self, other):
        if isinstance(other, PrecisionBase):
            raise NotImplementedError('Division between two Precision objects is not supported')
        return self * (1 / other)

    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        return self - other

    def __rsub__(self, other):
        return -self + other

    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        return self + other

    def __imul__(self, other):
        return self * other

    def __rmul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other

    def __pow__(self, other):
        if not isinstance(other, int):
            raise NotImplementedError('Only integer powers are supported')
        if other < 0:
            raise NotImplementedError('Only positive powers are supported')

        if other % 2 == 0:
            _min, _max = sorted(self.minmax**other)
            if self.min < 0:
                _min = 0
        else:
            _min, _max = self.minmax**other
        delta = self.delta**other
        partition = round((_max - _min) / delta + 1)
        return self.__class__(_min, _max, partition)

    def __ipow__(self, other):
        return self**other

    def __lshift__(self, other):
        if not isinstance(other, int):
            raise NotImplementedError('Only integer shifts are supported')

        _min, _max = self.minmax * 2**other
        return self.__class__(_min, _max, self.partition)

    def __rshift__(self, other):
        return self << -other


def short_render(k: int, b: int, I: int):  # noqa: E741
    return f'{"" if k else "u"}fixed<{b},{I}>'


class FixedPointPrecision(Precision):

    def __init__(self, min: float, max: float, partition: int | float):
        super().__init__(min, max, partition)

    def make_proper(self):
        "Cast min, max, partition to match hardware-friendly bitwidths."
        if self.partition == float('inf'):
            raise ValueError('Cannot match hardware delta for infinite partition - precision ill-defined')
        new_delta = 2.0 ** -const_fp_bits(self.delta, *self._minmax)
        _min, _max = self.minmax
        k = _min < 0
        i = int(np.ceil(np.log2(max(abs(_min), abs(_max) + new_delta))))

        _min = -(k * 2.0**i)
        _max = 2.0**i - new_delta
        partition = round((_max - _min) / new_delta + 1)
        return self.__class__(_min, _max, partition)

    @property
    def proper(self):
        "Check whether the precision is proper, i.e. it can be represented in hardware-friendly bitwidths."
        if self.partition == float('inf'):
            return False
        return (
            (self.delta == 2.0**-self.f) and (self.min == -(self.k * 2.0**self.i)) and (self.max == 2.0**self.i - self.delta)
        )

    @property
    def k(self) -> bool:
        'keep_negative'
        return self.min < 0

    @property
    def i(self):
        'integer_bits, excluding sign bit'
        m = max(abs(self.min), abs(self.max) + self.delta)
        return int(np.ceil(np.log2(m)))

    @property
    def I(self):  # noqa: E743
        'integer_bits, including sign bit'
        return self.i + self.k

    @property
    def f(self):
        'fractional_bits'
        return int(np.ceil(-np.log2(self.delta)))

    @property
    def kif(self):
        'keep_negative, integer_bits (excluding sign), fractional_bits'
        return self.k, self.i, self.f

    @property
    def kbi(self):
        'keep_negative, total_bits, integer_bits (including sign)'
        return self.k, self.b, self.I

    @property
    def b(self):
        'total_bits'
        return self.k + self.i + self.f

    def to_kif(self):
        return self.k, self.i, self.f

    def __str__(self):
        if not self.proper:
            return f'FixedPointPrecision(min={self.min}, max={self.max}, partition={self.partition}, [INVALID])'
        return short_render(self.k, self.b, self.I)

    @overload
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, x: float) -> float: ...

    def __call__(self, x):
        "Basic quantizer in numpy, assumes TRN and WRAP modes. Can only be used if the precision is proper."
        if not self.proper:
            raise ValueError('Precision is improper, convert to hardware-friendly precision first')
        x = x * 2**self.f
        x = np.floor(x)
        x += self.k * 2**self.b
        x %= 2 ** (self.b + self.k)
        x -= self.k * 2**self.b
        x = (x / 2**self.f).astype(np.float32)
        return x
