from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import StandardFloatPrecisionType


class ValidateConvImplementation(OptimizerPass):
    def match(self, node):
        return 'Conv' in node.class_name

    def transform(self, model, node):
        if node.get_attr('implementation', 'linebuffer') == 'encoded':
            print(
                f'WARNING: "Encoded" implementation in "{node.name}" ({node.class_name}) is not supported in Vitis backend. '
                'Switching to "LineBuffer" implementation.'
            )
            node.set_attr('implementation', 'linebuffer')


class ValidateResourceStrategy(OptimizerPass):
    _resource_layer_cls = ['Conv1D', 'Conv2D', 'Dense']

    def match(self, node):
        is_resource_layer = len([layer_cls for layer_cls in self._resource_layer_cls if layer_cls in node.class_name]) > 0
        is_resource_strategy = node.model.config.is_resource_strategy(node)

        return is_resource_layer and is_resource_strategy

    def transform(self, model, node):
        n_in, _ = model.config.backend.get_layer_mult_size(node)
        rf = node.get_attr('reuse_factor')
        if rf > n_in and rf % n_in > 0:
            print(
                f'WARNING: "Resource" strategy in "{node.name}" ({node.class_name}) may have suboptimal QoR in Vitis '
                'backend due to use of "urem" cores in Vitis HLS <= 2022.1.\n'
                'Consider using a different ReuseFactor or switching to "Latency" strategy if using older versions '
                'of Vitis HLS.'
            )


class ValidateResourceUnrolledStrategy(OptimizerPass):
    _unrolled_layer_cls = ['Conv1D', 'Conv2D', 'Dense', 'GRU', 'LSTM']

    def match(self, node):
        is_unrolled_layer = len([layer_cls for layer_cls in self._unrolled_layer_cls if layer_cls in node.class_name]) > 0
        is_unrolled_strategy = node.get_attr('strategy', 'latency').lower() == 'resource_unrolled'

        return is_unrolled_layer and is_unrolled_strategy

    def transform(self, model, node):
        print(
            f'WARNING: "ResourceUnrolled" strategy in "{node.name}" ({node.class_name}) may have unexpected II in'
            'Vitis backend.\nVerify that the final design satisfies the latency/II constraints.'
        )


class ValidateBidirectionalMergeMode(OptimizerPass):
    _unrolled_layer_cls = ['Bidirectional']

    def match(self, node):
        is_bidirectional_rnn_layer = (
            len([layer_cls for layer_cls in self._unrolled_layer_cls if layer_cls in node.class_name]) > 0
        )
        is_merge_mode_not_concat = node.get_attr('merge_mode', 'concat') != 'concat'

        return is_bidirectional_rnn_layer and is_merge_mode_not_concat

    def transform(self, model, node):
        merge_mode = node.get_attr('merge_mode', 'concat')
        print(
            f'WARNING: "{merge_mode}" merge mode in "{node.name}" ({node.class_name}) is not supported in Vitis backend. '
            'Switching to "concat" merge mode.'
        )
        node.set_attr('merge_mode', 'concat')


class ValidateBidirectionalIoType(OptimizerPass):
    _unrolled_layer_cls = ['Bidirectional']

    def match(self, node):
        is_bidirectional_rnn_layer = (
            len([layer_cls for layer_cls in self._unrolled_layer_cls if layer_cls in node.class_name]) > 0
        )
        is_layer_io_type_stream = node.model.config.config['IOType'] != 'io_parallel'

        return is_bidirectional_rnn_layer and is_layer_io_type_stream

    def transform(self, model, node):
        raise Exception(
            f'WARNING: "{node.model.config.config["IOType"]}" IO Type is not supported in Vitis backend '
            f'for "{node.name}" ({node.class_name}). Please use "io_parallel".'
        )


class ValidateStdCppTypes(OptimizerPass):
    def match(self, node):
        return True

    def transform(self, model, node):
        prec_types = [prec_type.precision for prec_type in node.get_layer_precision().values()]
        prec_types = [
            prec_type
            for prec_type in prec_types
            if isinstance(prec_type, StandardFloatPrecisionType)
            and prec_type.use_cpp_type
            and str(prec_type) not in ('float', 'double')
        ]
        if len(prec_types) > 0:
            print(
                f'WARNING: Layer "{node.name}" uses C++ types that are not synthesizable with Vitis backend. '
                'Use only for testing purposes.'
            )
