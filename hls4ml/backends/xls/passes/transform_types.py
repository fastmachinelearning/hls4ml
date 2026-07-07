# from hls4ml.backends.xls.xls_types import ()

from hls4ml.backends.xls.xls_types import (
    XLSInterfaceVarConverter,
    XLSPrecisionConverter,
    XLSTypeConverter,
    XLSVarConverter,
    XLSWeightVarConverter,
)
from hls4ml.model.optimizer import GlobalOptimizerPass


class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.precision_converter = XLSPrecisionConverter()
        self.type_converter = XLSTypeConverter(precision_converter=self.precision_converter)
        self.var_converter = XLSVarConverter(type_converter=self.type_converter)
        self.interface_var_converter = XLSInterfaceVarConverter(type_converter=self.type_converter)
        self.weight_var_converter = XLSWeightVarConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        for out_name, var in node.variables.items():
            if out_name in node.inputs or out_name in node.outputs:
                new_var = self.interface_var_converter.convert(var)
            else:
                new_var = self.var_converter.convert(var)
            node.set_attr(out_name, new_var)

        for w_name, weight in node.weights.items():
            new_weight = self.weight_var_converter.convert(weight_var=weight, node=node, weights_key=w_name)
            node.set_attr(w_name, new_weight)

        for t_name, type in node.types.items():
            new_type = self.type_converter.convert(type)
            node.set_attr(t_name, new_type)
