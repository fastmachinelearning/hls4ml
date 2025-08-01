# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import Literal, TYPE_CHECKING
from numpy.typing import NDArray
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer


from hls4ml.model.optimizer import OptimizerPass

import math
from fxpmath import Fxp

    
class BuildTables(OptimizerPass):
    """Builds attributes that store the softmax and multiplication inverse for the approximation
    of the Softmax function.
    """

    def match(self, node: Layer) -> bool:
        if node.class_name == 'Softmax':
            return True
        return False

    def transform(self, model: ModelGraph, node: Layer) -> Literal[False]:      

        # i * 2^{integer_part - clog2(table_size)}
        def get_real_val_from_idx(i, table_size, integer, negative):
            N = math.ceil(math.log2(table_size))
            exp = integer - N

            if negative:
                base = i
                return -(base * 2**(exp-1))
            
            else:
                if i < table_size / 2:
                    base = i
                else:
                    base = -(table_size - i)                
                return base * 2**exp
        
        table_size = dict(node.attributes)['table_size']
        exp_table = []
        inv_table = []

        # Types:
        #   softmax_accum_t:        18
        #   softmax_inv_table_t:    18
        #   softmax_exp_table_t:    18
        #   softamx_inv_inp_t:      18
        #   result_t:               16
        #   softmax_table_t:        18
        # TODO: manage the differnet types

        for name, var in node.get_layer_precision().items():
            print(name, ': ', var.precision.width)
        exp_width = node.get_layer_precision()['softmax_exp_table_t'].precision.width
        exp_frac = exp_width - node.get_layer_precision()['softmax_exp_table_t'].precision.integer
        inv_width = node.get_layer_precision()['softmax_inv_table_t'].precision.width
        inv_frac = inv_width - node.get_layer_precision()['softmax_inv_table_t'].precision.integer


        type_var = node.get_layer_precision()['softmax_table_t']
        width = type_var.precision.width 
        frac  = type_var.precision.width - type_var.precision.integer

        nb = int(node.get_attr('in_nb').split(':', 1)[1])
        bu = int(node.get_attr('in_bu').split(':', 1)[1])
        in_integer = nb - bu
        requires_negative_exp = dict(node.attributes).get('implementation', 'stable') == 'stable'

        # create exp table
        for i in range(table_size):
            real_val = get_real_val_from_idx(i, table_size, integer=in_integer, negative=requires_negative_exp)
            e = math.exp(real_val)
            # print("XLS: ", i, " x: ", real_val)
            fxp_e = Fxp(e, signed=True, n_word=exp_width, n_frac=exp_frac, rounding='around', overflow='saturate').raw()
            exp_table.append(fxp_e)

        # create div table
        for i in range(table_size):
            real_val = get_real_val_from_idx(i, table_size, integer=8, negative=False)
            inv = 1.0 / real_val if real_val != 0 else 2**(inv_width - 1)
            fxp_inv = Fxp(inv, signed=True, n_word=inv_width, n_frac=inv_frac, rounding='around', overflow='saturate').raw()
            inv_table.append(fxp_inv)

        node.set_attr('write_table', True)
        node.set_attr('exp_table_xls', exp_table)
        node.set_attr('inv_table_xls', inv_table)

        return False

