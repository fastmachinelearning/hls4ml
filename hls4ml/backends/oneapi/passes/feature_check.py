from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FloatPrecisionType


class ValidateAcTypes(OptimizerPass):
    def match(self, node):
        return True

    def transform(self, model, node):
        prec_types = [prec_type.precision for prec_type in node.get_layer_precision().values()]
        prec_types = [prec_type for prec_type in prec_types if isinstance(prec_type, FloatPrecisionType)]
        if len(prec_types) > 0:
            raise Exception(f'Layer "{node.name}" uses ac_float types that are not supported in oneAPI')
