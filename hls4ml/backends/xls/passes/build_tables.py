# Typing imports
from __future__ import annotations  # makes all annotations into strings

import warnings
from copy import copy
from enum import Enum
from typing import Literal, Callable, TYPE_CHECKING

from hls4ml.backends.xls.xls_types import XLSFixedPointType, XLSLookupTable, XLSFixedPoint, float_to_significand
from hls4ml.model.types import FixedPrecisionType

if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer

from hls4ml.model.optimizer import OptimizerPass

import math


class LookupTableRange(Enum):
    FULL = 1
    NON_NEGATIVE = 2
    NEGATIVE = 3


def build_table(name: str, func: Callable[[float], float], table_size: int, input_precision: FixedPrecisionType,
                output_precision: FixedPrecisionType,
                table_range: LookupTableRange) -> XLSLookupTable:
    # Hereafter 'raw' means operations with significand values, i.e.
    # raw_x == x.significand == int(x * 2**precision.fractional)

    raw_to_float = 2 ** (-input_precision.fractional)

    def raw_func(raw_x: int) -> int:
        return float_to_significand(
            func(raw_x * raw_to_float),
            output_precision
        )

    raw_minus_inf = XLSFixedPoint.min_value(input_precision).significand.value
    raw_plus_inf = XLSFixedPoint.max_value(input_precision).significand.value
    match table_range:
        # x = -inf..+inf
        case LookupTableRange.FULL:
            raw_original_x_min = raw_minus_inf
            raw_original_x_max = raw_plus_inf
        # x = 0..+inf
        case LookupTableRange.NON_NEGATIVE:
            raw_original_x_min = 0
            raw_original_x_max = raw_plus_inf
        # x = -inf..0
        case LookupTableRange.NEGATIVE:
            raw_original_x_min = raw_minus_inf
            raw_original_x_max = -1

    raw_x_min = raw_original_x_min
    raw_x_max = raw_original_x_max

    # Build input range for lookup table.
    # If the function saturates at the table edges,
    # we adjust the range to account for that.
    recompute_range = True
    while recompute_range:
        raw_log2_step = math.ceil(math.log2((raw_x_max - raw_x_min) / (table_size - 1)))
        if raw_log2_step < 0:
            raw_log2_step = 0
        raw_step = 2 ** raw_log2_step
        f_min = raw_func(raw_x_min)
        f_max = raw_func(raw_x_max)
        raw_range = list(range(raw_x_min, raw_x_max + 1, raw_step))

        recompute_range = False
        for x in raw_range[1:]:
            if raw_func(x) == f_min:
                raw_x_min = x
                recompute_range = True
            else:
                break
        for x in reversed(raw_range[:-1]):
            if x < raw_x_min:
                break
            if raw_func(x) == f_max:
                raw_x_max = x
                recompute_range = True

    if raw_x_min != raw_original_x_min or raw_x_max != raw_original_x_max:
        warnings.warn(
            f'Lookup table {name} range has been reduced to account for saturation at the table edges. '
            f'The original significand range was {raw_original_x_min}..{raw_original_x_max}, '
            f'and the adjusted range is {raw_x_min}..{raw_x_max}.'
        )
    if len(raw_range) < table_size:
        warnings.warn(f'Lookup table {name} size has been reduced from {table_size} to {len(raw_range)}.')

    assert 0 < len(raw_range) <= table_size
    assert raw_range[0] == raw_x_min >= raw_original_x_min
    assert raw_range[-1] <= raw_x_max <= raw_original_x_max

    return XLSLookupTable(
        name=name,
        input_precision=XLSFixedPointType.from_precision(input_precision),
        output_precision=XLSFixedPointType.from_precision(output_precision),
        x_min=XLSFixedPoint(type=input_precision, significand=raw_x_min),
        log2_step=raw_log2_step - input_precision.fractional,
        raw_table=[raw_func(x) for x in raw_range])


def build_softmax_tables(node: Layer) -> list[XLSLookupTable]:
    table_size = int(node.get_attr('table_size'))
    exp_table_size = int(node.get_attr('exp_table_size', table_size))
    inv_table_size = int(node.get_attr('inv_table_size', table_size))
    implementation = node.get_attr('implementation', 'stable')
    input_precision = node.get_input_variable().type.precision
    exp_in = copy(input_precision)
    exp_out = node.get_attr('exp_table_t').precision
    match implementation:
        case 'stable':
            exp_in.width += 1
            exp_in.integer += 1
            exp_name = 'EXP_NEG_TABLE'
            exp_func = lambda x: math.exp(-x)
            # Arguments of exp_func are (x_max - x_i) > 0
            exp_table_range = LookupTableRange.NON_NEGATIVE
        case 'latency':
            exp_name = 'EXP_TABLE'
            exp_func = math.exp
            # Arguments of exp_func are x_i, which can be both positive and negative
            exp_table_range = LookupTableRange.FULL
        case _:
            raise ValueError(f'Unknown softmax implementation={implementation}')

    inv_in = exp_out
    inv_out = node.get_attr('inv_table_t').precision
    inv_name = 'INV_TABLE'

    def inv_func(x):
        if x == 0:
            return inv_out.max
        return 1.0 / x

    exp_table = build_table(
        name=exp_name,
        func=exp_func,
        table_size=exp_table_size,
        input_precision=exp_in,
        output_precision=exp_out,
        table_range=exp_table_range
    )
    inv_table = build_table(
        name=inv_name,
        func=inv_func,
        table_size=inv_table_size,
        input_precision=inv_in,
        output_precision=inv_out,
        # We're inverting sum of exponents, which is always non-negative.
        table_range=LookupTableRange.NON_NEGATIVE
    )
    return [exp_table, inv_table]


def build_activation_table(node: Layer) -> XLSLookupTable:
    activation = node.get_attr('activation').lower()
    table_name = f'{activation.upper()}_TABLE'
    match activation:
        case 'elu':
            table_range = LookupTableRange.NEGATIVE
            alpha = node.get_attr('activ_param')

            def func(x):
                assert x < 0, f'Building ELU table only for x < 0, got {x}'
                return alpha * (math.exp(x) - 1)
        case 'selu':
            table_range = LookupTableRange.NEGATIVE
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946

            def func(x):
                assert x < 0, f'Building ELU table only for x < 0, got {x}'
                return scale * alpha * (math.exp(x) - 1)
        case 'softplus':
            table_range = LookupTableRange.FULL

            def func(x):
                return math.log(1 + math.exp(x))
        case 'softsign':
            table_range = LookupTableRange.NON_NEGATIVE

            def func(x):
                return x / (1 + abs(x))
        case 'tanh':
            table_range = LookupTableRange.NON_NEGATIVE

            def func(x):
                return math.tanh(x)
        case 'sigmoid':
            table_range = LookupTableRange.FULL

            def func(x):
                return 1 / (1 + math.exp(-x))
        case _:
            raise ValueError(f'Unknown activation={activation}')

    match table_range:
        case LookupTableRange.FULL:
            pass
        case LookupTableRange.NON_NEGATIVE:
            table_name += '_NON_NEGATIVE'
        case LookupTableRange.NEGATIVE:
            table_name += '_NEGATIVE'

    return build_table(
        name=table_name,
        func=func,
        table_size=int(node.get_attr('table_size')),
        input_precision=node.get_input_variable().type.precision,
        output_precision=node.get_output_variable().type.precision,
        table_range=table_range
    )


class BuildTables(OptimizerPass):
    """Builds attributes that store the softmax and multiplication inverse for the approximation
    of the Softmax function.
    """

    def match(self, node: Layer) -> bool:
        match node.class_name:
            case 'Softmax':
                return node.get_attr('implementation', 'stable') != 'argmax'
            case 'Activation':
                return node.get_attr('activation').lower() in [
                    'selu', 'softplus', 'softsign', 'tanh', 'sigmoid'
                ]
            case 'ParametrizedActivation':
                return node.get_attr('activation').lower() in ['elu', 'prelu']
            case _:
                return False

    def transform(self, model: ModelGraph, node: Layer) -> Literal[False]:
        lookup_tables = node.get_attr('lookup_tables', [])
        match node.class_name:
            case 'Softmax':
                lookup_tables += build_softmax_tables(node)
            case 'Activation':
                lookup_tables.append(build_activation_table(node))
            case 'ParametrizedActivation':
                lookup_tables.append(build_activation_table(node))
            case _:
                raise ValueError(f'Unknown layer type: {node.class_name}')

        node.set_attr('lookup_tables', lookup_tables)
        return False
