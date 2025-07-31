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
        def get_real_val_from_idx(i, type_var, table_size):  
            return i * (2 ** (type_var.precision.integer - math.ceil(math.log2(table_size))))
        
        table_size = dict(node.attributes)['table_size']
        exp_table = []
        inv_table = []

        _, type_var = list(node.get_layer_precision().items())[0]
        width = type_var.precision.width 
        frac  = type_var.precision.width - type_var.precision.integer

        # create exp table
        for i in range(table_size):
            real_val = get_real_val_from_idx(i, type_var, table_size)
            e = math.exp(real_val)
            fxp_e = Fxp(e, signed=True, n_word=width, n_frac=frac, rounding='around', overflow='saturate').raw()
            exp_table.append(fxp_e)

        # create div table
        for i in range(table_size):
            real_val = get_real_val_from_idx(i, type_var, table_size)
            inv = 1.0 / real_val if real_val != 0 else 2**(type_var.precision.width - 1)
            fxp_inv = Fxp(inv, signed=True, n_word=width, n_frac=frac, rounding='around', overflow='saturate').raw()
            inv_table.append(fxp_inv)

        node.set_attr('write_table', True)
        node.set_attr('exp_table_xls', exp_table)
        node.set_attr('inv_table_xls', inv_table)

        return False

