from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.hls_types import FixedPrecisionType, IntegerPrecisionType
from hls4ml.backends.fpga.fpga_types import ACFixedPrecisionType, ACIntegerPrecisionType, ArrayVariable, StreamVariable


class TransformACTypes(GlobalOptimizerPass):
    def transform(self, model, node):
        for hls_type in node.types.values():
            precision_type = hls_type.precision
            if isinstance(precision_type, IntegerPrecisionType):
                hls_type.precision = ACIntegerPrecisionType.from_precision(precision_type)
            elif isinstance(precision_type, FixedPrecisionType):
                hls_type.precision = ACFixedPrecisionType.from_precision(precision_type)
            else:
                raise Exception('Unknown precision type {} in {} ({})'.format(precision_type.__class__.__name__, node.name, node.class_name))
