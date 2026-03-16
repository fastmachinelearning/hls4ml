# Typing imports
from __future__ import annotations  # makes all annotations into strings

import warnings
from typing import Literal, Callable, TYPE_CHECKING, Any
from copy import copy

from hls4ml.backends.xls.xls_types import XLSArray, XLSArrayType, XLSFixedPointType, XLSFunctionCall, XLSConst, \
    XLSLookupTable, XLSFixedPoint
from hls4ml.model.types import FixedPrecisionType

if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer

from hls4ml.model.optimizer import OptimizerPass

import math
from fxpmath import Fxp


def build_raw_table(func: Callable[[float], float], table_size: int, x_start, step, input_precision: FixedPrecisionType,
                    output_precision: FixedPrecisionType) -> list[int]:
    raw_data = []
    for i in range(table_size):
        x = x_start + i * step
        fxp = Fxp(val=func(x), signed=True, n_word=output_precision.width, n_frac=output_precision.fractional,
                  rounding='around', overflow='saturate')
        # TODO store raw or Fxp?
        raw_data.append(fxp.raw())
    return raw_data


def build_table(name: str, func: Callable[[float], float], table_size: int, input_precision: FixedPrecisionType,
                output_precision: FixedPrecisionType,
                include_negative: bool) -> XLSLookupTable:
    if include_negative:
        in_width = input_precision.width
    else:
        in_width = input_precision.width - 1
    # Adjust table size.
    # TODO: this should be moved to FixSoftmaxTableSize, which currently does not account for include_negative=False case.
    max_table_size_in = 2 ** in_width
    max_table_size_out = 2 ** output_precision.width
    if table_size > max_table_size_in:
        warnings.warn(
            f'{name}: table size {table_size} is too large for input bitwidth and will be set to {max_table_size_in}.',
            stacklevel=1)
        table_size = max_table_size_in
    if table_size > max_table_size_out:
        warnings.warn(
            f'{name}: table size {table_size} is too large for output bitwidth and will be set to {max_table_size_out}.',
            stacklevel=1)
        table_size = max_table_size_out

    N = math.ceil(math.log2(table_size))
    log2_step = in_width - input_precision.fractional - N
    # x = -inf..+inf
    if include_negative:
        x_start = float(input_precision.min)
        xls_x_min = XLSFixedPoint.min_value(input_precision)
    # x = 0..+inf
    else:
        x_start = 0.0
        xls_x_min = XLSFixedPoint.zero(input_precision)

    # Sanity check for table_size adjustment above
    assert log2_step >= -input_precision.fractional, \
        f'Lookup table size {table_size} is too large for input precision {input_precision}, include_negative={include_negative}.'
    assert log2_step >= -output_precision.fractional, \
        f'Lookup table size {table_size} is too large for output precision {output_precision}, include_negative={include_negative}.'

    step = 2 ** log2_step

    raw_data = build_raw_table(func=func, table_size=table_size, x_start=x_start, step=step,
                               input_precision=input_precision,
                               output_precision=output_precision)

    return XLSLookupTable(
        name=name,
        input_precision=XLSFixedPointType.from_precision(input_precision),
        output_precision=XLSFixedPointType.from_precision(output_precision),
        x_min=xls_x_min,
        log2_step=log2_step,
        raw_table=raw_data)


class BuildTables(OptimizerPass):
    """Builds attributes that store the softmax and multiplication inverse for the approximation
    of the Softmax function.
    """

    def match(self, node: Layer) -> bool:
        """Matches to all softmax layers. The only optimization that does not include a table lookup is 'argmax'.
        """
        if node.class_name == 'Softmax' and dict(node.attributes).get('implementation', 'stable') != 'argmax':
            return True
        return False

    def transform(self, model: ModelGraph, node: Layer) -> Literal[False]:
        table_size = int(node.get_attr('table_size'))
        exp_table_size = int(node.get_attr('exp_table_size', table_size))
        inv_table_size = int(node.get_attr('inv_table_size', table_size))
        implementation = node.get_attr('implementation', 'stable')
        input_precision = node.get_input_variable().type.precision
        exp_in = copy(input_precision)
        exp_out = node.get_layer_precision()['softmax_exp_table_t'].precision
        match implementation:
            case 'stable':
                exp_in.width += 1
                exp_in.integer += 1
                exp_name = 'EXP_NEG_TABLE'
                exp_func = lambda x: math.exp(-x)
                # Arguments of exp_func are (x_max - x_i) > 0
                exp_include_negative = False
            case 'latency':
                exp_name = 'EXP_TABLE'
                exp_func = math.exp
                # Arguments of exp_func are x_i, which can be both positive and negative
                exp_include_negative = True
            case _:
                raise ValueError(f'Unknown softmax implementation={implementation}')

        inv_in = exp_out
        inv_out = node.get_layer_precision()['softmax_inv_table_t'].precision
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
            include_negative=exp_include_negative
        )
        inv_table = build_table(
            name=inv_name,
            func=inv_func,
            table_size=inv_table_size,
            input_precision=inv_in,
            output_precision=inv_out,
            # We're inverting sum of exponents, which is always non-negative.
            include_negative=False
        )

        lookup_tables = node.get_attr('lookup_tables', []) + [exp_table, inv_table]
        node.set_attr('lookup_tables', lookup_tables)

        return False
