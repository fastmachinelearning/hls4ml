from hls4ml.model.layers import Dense
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


class ValidateFusedConfiguration(OptimizerPass):
    """Check what the fused strategy has been asked to do, once the fusion passes have run.

    Whether the backend provides the strategy at all is checked earlier, by the pass of the same family
    in the Vivado backend. What is left is whether this model can use it, and whether it was given
    something it cannot carry out:

    * the io type, which must be io_parallel;
    * the layer type, as only Dense is implemented;
    * the reuse factor, which the form of a layer limits.

    The first is an error and the other two are reported, because a model of mixed layer types with the
    strategy set for all of them should still build, with its Dense chains fused.
    """

    def match(self, node):
        if node.get_attr('strategy') is None:
            return False
        return str(node.model.config.get_strategy(node)).lower() == 'fused'

    def transform(self, model, node):
        self._check_io_type(model, node)
        if self._check_layer_type(node):
            return False
        if self._reads_interval(node):
            self._check_interval(node)
        else:
            self._check_reuse_factor(node)
        return False

    @staticmethod
    def _reads_interval(node):
        from hls4ml.backends.vitis.passes.fuse_dense import reads_interval

        return reads_interval(node)

    def _check_io_type(self, model, node):
        io_type = model.config.get_config_value('IOType')
        if io_type == 'io_parallel':
            return
        raise Exception(
            f'Layer "{node.name}" ({node.class_name}) has strategy = "fused", which needs '
            f'io_type = "io_parallel"; this model uses "{io_type}". One read of an io_stream connection '
            'carries a whole row, which the fused kernels cannot use.'
        )

    def _check_layer_type(self, node):
        """Report a layer type that does not implement the strategy. Returns True if it is one."""
        if isinstance(node, Dense):
            return False
        print(
            f'WARNING: Layer "{node.name}" ({node.class_name}) has strategy = "fused", which is '
            f'implemented for Dense layers only. The layer is built with strategy '
            f'"{node.get_attr("strategy")}".'
        )
        return True

    def _check_interval(self, node):
        """Check a layer whose reuse factor gives a requested interval, and report what was built."""

        target = max(1, int(node.get_attr('reuse_factor', 1) or 1))

        floor = node.get_attr('fused_interval_floor')
        if floor is not None:
            raise Exception(
                f'Layer "{node.name}" ({node.class_name}) cannot achieve an interval of {target}. With '
                f'all of its multipliers in use it still needs {floor} cycles. Request {floor} or more, '
                'or leave ReuseFactorAsInterval unset so the reuse factor keeps its usual meaning.'
            )

        producer = node.get_input_node()
        if producer is not None and producer.get_attr('fused_form') is not None and not self._reads_interval(producer):
            raise Exception(
                f'Layers "{producer.name}" and "{node.name}" are fused into one region, but only the '
                'second uses its reuse factor as an interval. A region has a single interval, so its '
                'layers must use the reuse factor the same way.'
            )

        consumers = node.get_output_nodes()
        neighbour = consumers[0] if consumers else None
        if neighbour is not None and neighbour.get_attr('fused_form') is not None:
            if not self._reads_interval(neighbour):
                raise Exception(
                    f'Layers "{node.name}" and "{neighbour.name}" are fused into one region, but only '
                    'the first uses its reuse factor as an interval. A region has a single interval, so '
                    'its layers must use the reuse factor the same way.'
                )
            neighbour_target = max(1, int(neighbour.get_attr('reuse_factor', 1) or 1))
            if neighbour_target != target:
                raise Exception(
                    f'Layers "{node.name}" and "{neighbour.name}" are fused into one region but request '
                    f'intervals of {target} and {neighbour_target}. A region has a single interval, so '
                    'its layers must request the same one.'
                )

        built = node.get_attr('fused_multipliers')
        pad = node.get_attr('fused_pad_cycles') or 0
        slack = node.get_attr('fused_interval_slack') or 0
        result = f'between {target - slack} and {target}' if slack else f'{target}'
        print(
            f'Layer "{node.name}": reuse factor {target} used as a requested interval. Built with '
            f'{built} multipliers and {pad} wait cycles. The interval will be {result} cycles.'
        )

    def _check_reuse_factor(self, node):
        """Check the reuse factor in two stages: reject 1, then report a value the layer cannot reach.

        Reuse factor 1 asks for a fully parallel layer, which these kernels cannot build.
        """

        asked = max(1, int(node.get_attr('reuse_factor', 1) or 1))
        built = node.get_attr('fused_multipliers')
        # The lowest reuse factor that uses every multiplier the form can use; below it every value
        # builds the same design. A layer the planner gave no form has neither number.
        lowest_usable = None
        if built is not None:
            built = int(built)
            lowest_usable = int(node.get_attr('n_in')) * int(node.get_attr('n_out')) // built

        if asked == 1:
            advice = f'For the most parallel fused design set ReuseFactor to {lowest_usable}. ' if lowest_usable else ''
            raise Exception(
                f'Layer "{node.name}" ({node.class_name}) has strategy "fused" with reuse factor 1. The '
                'fused strategy shares multipliers over several cycles and cannot build a fully parallel '
                f'layer. {advice}For a fully parallel layer use strategy "Latency", or '
                '"distributed_arithmetic".'
            )

        if lowest_usable is None or lowest_usable <= asked:
            return
        print(
            f'WARNING: Layer "{node.name}" ({node.class_name}) asks for reuse factor {asked} with '
            f'strategy "fused", which cannot be built: the {node.get_attr("fused_form")} form uses at '
            f'most {built} multipliers at a time. The layer is built with {built}, which is reuse '
            f'factor {lowest_usable}.'
        )
