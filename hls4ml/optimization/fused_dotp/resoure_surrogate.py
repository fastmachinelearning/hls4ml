from copy import copy

import numpy as np
import pandas as pd

from hls4ml.model import ModelGraph

from .precision import FixedPointPrecision
from .symbolic_variable import Variable
from .utils import precision_from_const


class DictWrap(dict):
    "Wrapper class for dict to support addition of values with same keys."

    def __add__(self, other: dict | int) -> 'DictWrap':
        r = copy(self)
        if not isinstance(other, dict):
            return r
        for k in other:
            if k in r:
                r[k] += other[k]
            else:
                r[k] = other[k]
        return r

    def __radd__(self, other: dict | int):
        return self.__add__(other)


def resource_bin_add(p1: FixedPointPrecision, p2: FixedPointPrecision):
    I1, I2 = p1.I, p2.I
    f1, f2 = p1.f, p2.f
    return DictWrap(add=max(I1, I2) + max(f1, f2), n_add=1)

    # H, h = max(I1, I2), min(I1, I2)
    # L = -min(f1, f2)
    # return DictWrap(add=(H - L) * (h > L))


def resource_bin_mul(p1: FixedPointPrecision, p2: FixedPointPrecision):
    return DictWrap(mul=p1.b * p2.b, n_mul=1)


def resource_bin_shift(p1: FixedPointPrecision, shift: int):
    return DictWrap(shift=p1.b * abs(shift), n_shift=1)


def resource_bin_max(p1: FixedPointPrecision, p2: FixedPointPrecision):
    return DictWrap(cmp=max(p1.b, p2.b), n_cmp=1)


def resource_bin_sub(p1: FixedPointPrecision, p2: FixedPointPrecision):
    return DictWrap(sub=resource_bin_add(p1, p2)['add'], n_sub=1)


def resource_bin_neg(p1: FixedPointPrecision):
    return DictWrap(neg=p1.b, n_neg=1)


class ResourceSurrogate:
    def __init__(self):
        self.layers: dict[str, DictWrap] = {}

    def add(self, var: Variable):
        const = var.const
        precisions = [a.precision for a in var.ancestors]
        if const != 0:
            precisions.append(precision_from_const(const))
        if len(precisions) == 2:
            return resource_bin_add(*precisions)
        if len(precisions) == 1:
            return 0
        precisions_accum = np.cumsum(precisions[:-1])  # type: ignore
        return sum((resource_bin_add(p1, p2) for p1, p2 in zip(precisions_accum, precisions[1:])))

    def sub(self, var: Variable):
        p1, p2 = var.ancestors[0].precision, var.ancestors[1].precision
        return resource_bin_sub(p1, p2)

    def mul(self, var: Variable):
        const = var.const
        precisions = [a.precision for a in var.ancestors]
        if const != 1:
            precisions.append(precision_from_const(const))
        if len(precisions) == 2:
            return resource_bin_mul(*precisions)
        if len(precisions) == 1:
            return 0
        precisions_accum = np.cumprod(precisions[:-1])  # type: ignore
        return sum((resource_bin_mul(p1, p2) for p1, p2 in zip(precisions_accum, precisions[1:])))

    def shift(self, var: Variable):
        return resource_bin_shift(var.ancestors[0].precision, int(var.const))

    def neg(self, var: Variable):
        return resource_bin_neg(var.ancestors[0].precision)

    def new(self, var: Variable):
        return 0

    def const(self, var: Variable):
        return 0

    def _trace(self, v: Variable | int | float, recorded: set) -> int | DictWrap:
        if not isinstance(v, Variable):
            return 0
        if v.operation == 'new' or v in recorded:
            return 0
        resource = 0
        for a in v.ancestors:
            resource += self._trace(a, recorded)
        dr: int | DictWrap = getattr(self, v.operation)(v)
        resource += dr
        recorded.add(v)
        return resource

    def trace(self, r: list | np.ndarray, name: str, pf: int = 1):
        s = set()
        arr = np.array(r).ravel()
        zero = DictWrap(add=0, sub=0, mul=0, shift=0, neg=0, cmp=0, depth=0)
        zero = zero + {f'n_{k}': 0 for k in zero.keys()}
        params: DictWrap = zero + sum(self._trace(v, s) for v in arr)  # type: ignore
        if params == 0:  # layer outputs const array, no operation performed. skip
            return
        if len(arr) > 0:
            depth = max(v.depth for v in arr if isinstance(v, Variable))
            n_depth = sum(v.n_depth for v in arr if isinstance(v, Variable))
            params['depth'] = depth
            params['n_depth'] = n_depth
        params['pf'] = pf
        self.layers[name] = params

    def scan(self, model: ModelGraph):
        zero = DictWrap(add=0, sub=0, mul=0, shift=0, neg=0, cmp=0, depth=0)
        zero = zero + {f'n_{k}': 0 for k in zero.keys()}
        zero['pf'] = 1
        for name, layer in model.graph.items():
            r_variables = layer.attributes.attributes.get('r_variables')
            if r_variables is not None:
                pf = layer.attributes.attributes.get('parallelization_factor', 1)
                self.trace(r_variables, name, pf)

            result_t = layer.attributes.attributes.get('result_t')
            if result_t is None:
                continue
            overflow_mode = str(result_t.precision._saturation_mode)
            # round_mode = str(result_t.precision._rounding_mode)
            if not hasattr(layer.attributes.attributes[name], 'shape'):
                # Some layer doesn't have output shape...
                continue
            size = np.prod(layer.attributes.attributes[name].shape)
            width = result_t.precision.width
            if layer.attributes.attributes.get('accum_t') is not None:
                width = layer.attributes.attributes['accum_t'].precision.width
            if 'SAT' in overflow_mode:
                params = {'cmp': width * size * 2}
                self.layers[name] = self.layers.get(name, zero) + DictWrap(params)
            # if 'RND' in round_mode:
            #     params = {'cmp': width * size}
            #     self.layers[name] = self.layers.get(name, zero) + DictWrap(params)

    def _summary(self):
        if not self.layers:
            raise ValueError(
                'No layer is registered. If you have run `scan` already, the model contains no unrolled layer that this surrogate can analyze.'  # noqa: E501
            )
        df = pd.DataFrame.from_dict(self.layers, orient='index')
        lut = np.round((df['add'] + df['neg'] + 2 * df['sub']) * 0.65 + df['cmp'] * 1.5).astype(int) * df['pf']
        dsp = np.round(df['n_mul']).astype(int) * df['pf']
        latency_ns = df['depth'] * 0.86
        summary = pd.DataFrame({'LUT': lut, 'DSP': dsp, 'Latency (ns)': latency_ns})

        total_df = df.sum()
        total_df['pf'] = -1
        total_summary = summary.sum()
        df.loc['Total'] = total_df
        summary.loc['Total'] = total_summary
        return df, summary

    def summary(self):
        df, summary = self._summary()
        return summary

    def full_summary(self):
        df, summary = self._summary()
        return pd.concat([df, summary], axis=1)


# def resource_addr(v: Variable):
#     if len(v.ancestors) == 1:
#         return 0
#     k0, i0, f0 = v.ancestors[0].precision.kif
#     k1, i1, f1 = v.ancestors[1].precision.kif

#     i0, i1 = i0 + k0, i1 + k1
#     H, h = max(i0, i1), min(i0, i1)
#     L = -min(f0, f1)
#     return (H - L) * (h > L)


# def _accum_resource(v: Variable, recorded=set()):
#     if v.operation == 'new' or v in recorded:
#         return 0
#     accum = 0
#     for a in v.ancestors:
#         accum += _accum_resource(a, recorded)
#     if v.operation == 'add':
#         accum += resource_addr(v)
#     recorded.add(v)

#     return accum


# def accum_resource(r: list | np.ndarray):
#     s = set()
#     arr = np.array(r).ravel()
#     accum = 0
#     for v in arr:
#         if isinstance(v, Variable):
#             accum += _accum_resource(v, s)
#     return accum


# class Surrogate:
#     def __init__(self):
#         self.accum: dict[str, int] = {}

#     def inference(self):
#         return sum(self.accum.values())

#     def trace(self, r: list | np.ndarray, name: str):
#         self.accum[name] = accum_resource(r)
