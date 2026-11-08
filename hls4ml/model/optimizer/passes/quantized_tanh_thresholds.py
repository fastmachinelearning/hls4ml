from hls4ml.model.optimizer import OptimizerPass


class UpdateQuantizedTanhThresholds(OptimizerPass):
    """The thresholds of a batch normalization merged with a binary or ternary activation depend on the type of its
    input: binary data, which stores -1 as 0, gets thresholds for the stored values. Passes that run after the merge can
    change that type, such as xnor_pooling and binary_zero_padding, so the thresholds are computed again for the new
    type.
    """

    def match(self, node):
        return (
            node.class_name == 'BatchNormalizationQuantizedTanh'
            and node.get_attr('scale_data') is not None
            and node.get_attr('thresholds_input') != node.thresholds_input()
        )

    def transform(self, model, node):
        node.set_thresholds(node.get_attr('scale_data'), node.get_attr('bias_data'), node.get_attr('ternary_threshold'))
        return False
