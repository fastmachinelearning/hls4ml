from copy import deepcopy

import numpy as np

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType, UnspecifiedPrecisionType


class InferPrecisionTypes(OptimizerPass):
    def match(self, node):
        for layer_type in node.types.values():
            if isinstance(layer_type.precision, UnspecifiedPrecisionType):
                return True
        return False

    def transform(self, model, node):
        types_to_infer = []
        for type_name, type_obj in node.types.items():
            if isinstance(type_obj.precision, UnspecifiedPrecisionType):
                types_to_infer.append(type_name)

        inferred_types = self._infer_precision(node, types_to_infer)
        for type_name in types_to_infer:
            if type_name not in inferred_types:
                self._infer_default_type(node, type_name)

        return False  # No model graph changes made

    def _infer_precision(self, node, types_to_infer):
        node_class = node.class_name
        if node_class in ['Dense']:
            return self._infer_dense_precision(node, types_to_infer)

        if node_class in ['BatchNormalization']:
            return self._infer_bn_precision(node, types_to_infer)

        if node_class in ['Conv1D', 'Conv2D', 'PointwiseConv1D', 'PointwiseConv2D', 'Conv2DBatchnorm']:
            return self._infer_conv_precision(node, types_to_infer)

        if node_class in ['SeparableConv1D', 'SeparableConv2D', 'DepthwiseConv2D']:
            return self._infer_sepconv_precision(node, types_to_infer)

        if node_class in ['Pooling1D', 'Pooling2D']:
            return self._infer_pooling_precision(node, types_to_infer)

        if node_class in ['Clone', 'Reshape', 'Resize', 'Transpose', 'ZeroPadding1D', 'ZeroPadding2D']:
            return self._infer_output_matching_precision(node, types_to_infer)

        if node_class in ['Concatenate', 'Merge']:
            return self._infer_merge_precision(node, types_to_infer)

        # What about quantized activation layer? Setting it to 'auto' manually will break it here. We should prevent
        # this in config_from_* functions

        return []

    def _infer_default_type(self, node, type_name):
        model_config = node.model.config
        default_precision = model_config.backend.convert_precision_string(model_config.model_precision['default'])
        # No need to change the name of the NamedType since we use the default precision
        node.types[type_name].precision = default_precision

    def _infer_output_matching_precision(self, node, types_to_infer):
        assert 'result_t' in types_to_infer and len(types_to_infer) == 1

        in_var = node.get_input_variable()
        out_var = node.get_output_variable()
        in_out_type = in_var.type.precision
        out_var.type.precision = in_out_type

        return ['result_t']

    def _infer_common_precision(self, node, types_to_infer, n_ops):
        inferred_types = []

        input_precision = node.get_input_variable().type.precision
        input_width = input_precision.width
        input_integers = input_precision.integer

        if 'weight_t' in types_to_infer:
            weight_quantizer = node.get_attr('weight_quantizer', None)
            if weight_quantizer is not None:
                weight_width = weight_quantizer.bits
                weight_integers = weight_quantizer.hls_type.integer
                node.types['weight_t'].name = node.name + '_weight_t'
                node.types['weight_t'].precision = weight_quantizer.hls_type
            else:
                self._infer_default_type(node, 'weight_t')
                weight_width = node.types['weight_t'].precision.width
                weight_integers = node.types['weight_t'].precision.integer
            node.weights['weight'].update_precision(node.types['weight_t'].precision)

            inferred_types.append('weight_t')
        else:
            weight_width = node.types['weight_t'].precision.width
            weight_integers = node.types['weight_t'].precision.integer

        if 'bias_t' in types_to_infer:
            bias_quantizer = node.get_attr('bias_quantizer', None)
            if bias_quantizer is not None:
                bias_width = bias_quantizer.bits
                bias_integers = bias_quantizer.hls_type.integer
                node.types['bias_t'].name = node.name + '_bias_t'
                node.types['bias_t'].precision = bias_quantizer.hls_type
            else:
                self._infer_default_type(node, 'bias_t')
                bias_width = node.types['bias_t'].precision.width
                bias_integers = node.types['bias_t'].precision.integer
            node.weights['bias'].update_precision(node.types['bias_t'].precision)

            inferred_types.append('bias_t')
        else:
            bias_width = node.types['bias_t'].precision.width
            bias_integers = node.types['bias_t'].precision.integer

        new_type = FixedPrecisionType(
            width=int(max(np.ceil(input_width + weight_width + np.log2(n_ops)), bias_width) + 1),
            integer=int(max(np.ceil(input_integers + weight_integers + np.log2(n_ops)), bias_integers) + 1),
        )

        if 'accum_t' in types_to_infer:
            node.types['accum_t'].name = node.name + '_accum_t'
            node.types['accum_t'].precision = new_type

            inferred_types.append('accum_t')

        if 'result_t' in types_to_infer:
            node.types['result_t'].name = node.name + '_result_t'
            node.types['result_t'].precision = new_type

            inferred_types.append('result_t')

        return inferred_types

    def _infer_dense_precision(self, node, types_to_infer):
        n_ops = node.get_attr('n_in') * node.get_attr('n_out')
        return self._infer_common_precision(node, types_to_infer, n_ops)

    def _infer_conv_precision(self, node, types_to_infer):
        n_ops = node.get_attr('n_chan') * node.get_attr('filt_height', 1) * node.get_attr('filt_width')
        return self._infer_common_precision(node, types_to_infer, n_ops)

    def _infer_sepconv_precision(self, node, types_to_infer):
        inferred_types = []

        input_precision = node.get_input_variable().type.precision
        input_width = input_precision.width
        input_integers = input_precision.integer

        if 'depthwise_t' in types_to_infer:
            # TODO Current HLS implementations use data_T (input type) as the result hence this doesn't affect the output
            # precision ATM, but this will probably change in the future
            depthwise_quantizer = node.get_attr('depthwise_quantizer', None)
            if depthwise_quantizer is not None:
                node.types['depthwise_t'].name = node.name + '_depthwise_t'
                node.types['depthwise_t'].precision = depthwise_quantizer.hls_type
            else:
                self._infer_default_type(node, 'depthwise_t')
            node.weights['depthwise'].update_precision(node.types['depthwise_t'].precision)

            inferred_types.append('depthwise_t')

        if 'pointwise_t' in types_to_infer:
            pointwise_quantizer = node.get_attr('pointwise_quantizer', None)
            if pointwise_quantizer is not None:
                pointwise_width = pointwise_quantizer.bits
                pointwise_integers = pointwise_quantizer.hls_type.integer
                node.types['pointwise_t'].name = node.name + '_pointwise_t'
                node.types['pointwise_t'].precision = pointwise_quantizer.hls_type
            else:
                self._infer_default_type(node, 'pointwise_t')
                pointwise_width = node.types['pointwise_t'].precision.width
                pointwise_integers = node.types['pointwise_t'].precision.integer
            node.weights['pointwise'].update_precision(node.types['pointwise_t'].precision)

            inferred_types.append('pointwise_t')
        else:
            pointwise_width = node.types['pointwise_t'].precision.width
            pointwise_integers = node.types['pointwise_t'].precision.integer

        if 'bias_t' in types_to_infer:
            bias_quantizer = node.get_attr('bias_quantizer', None)
            if bias_quantizer is not None:
                bias_width = bias_quantizer.bits
                bias_integers = bias_quantizer.hls_type.integer
                node.types['bias_t'].name = node.name + '_bias_t'
                node.types['bias_t'].precision = bias_quantizer.hls_type
            else:
                self._infer_default_type(node, 'bias_t')
                bias_width = node.types['bias_t'].precision.width
                bias_integers = node.types['bias_t'].precision.integer
            node.weights['bias'].update_precision(node.types['bias_t'].precision)

            inferred_types.append('bias_t')
        else:
            bias_width = node.types['bias_t'].precision.width
            bias_integers = node.types['bias_t'].precision.integer

        n_ops = node.get_attr('n_chan')
        new_type = FixedPrecisionType(
            width=int(max(np.ceil(input_width + pointwise_width + np.log2(n_ops)), bias_width) + 1),
            integer=int(max(np.ceil(input_integers + pointwise_integers + np.log2(n_ops)), bias_integers) + 1),
        )

        if 'accum_t' in types_to_infer:
            node.types['accum_t'].name = node.name + '_accum_t'
            node.types['accum_t'].precision = new_type

            inferred_types.append('accum_t')

        if 'result_t' in types_to_infer:
            node.types['result_t'].name = node.name + '_result_t'
            node.types['result_t'].precision = new_type

            inferred_types.append('result_t')

        return inferred_types

    def _infer_bn_precision(self, node, types_to_infer):
        inferred_types = []

        if 'scale_t' in types_to_infer:
            self._infer_default_type(node, 'scale_t')
            node.weights['scale'].update_precision(node.types['scale_t'].precision)
            inferred_types.append('scale_t')

        if 'bias_t' in types_to_infer:
            self._infer_default_type(node, 'bias_t')
            node.weights['bias'].update_precision(node.types['bias_t'].precision)
            inferred_types.append('bias_t')

        if 'result_t' in types_to_infer:
            scale_precision = node.types['scale_t'].precision
            bias_precision = node.types['bias_t'].precision

            out_precision = deepcopy(node.get_input_node().get_output_variable().type.precision)
            out_precision.integer += scale_precision.integer
            out_precision.fractional = max(out_precision.fractional, scale_precision.fractional)

            out_precision.integer = max(out_precision.integer, bias_precision.integer) + 1
            out_precision.fractional = max(out_precision.fractional, bias_precision.fractional)
            out_precision.width = out_precision.fractional + out_precision.integer

            node.types['result_t'].name = node.name + '_result_t'
            node.types['result_t'].precision = out_precision

            inferred_types.append('result_t')

        return inferred_types

    def _infer_pooling_precision(self, node, types_to_infer):
        inferred_types = []

        if 'accum_t' in types_to_infer:
            input_precision = node.get_input_variable().type.precision
            input_width = input_precision.width
            input_integers = input_precision.integer

            n_ops = node.get_attr('n_filt') * node.get_attr('pool_height', 1) * node.get_attr('pool_width')

            accum_type = FixedPrecisionType(
                width=int(np.ceil(input_width + np.log2(n_ops)) + 1),
                integer=int(np.ceil(input_integers + np.log2(n_ops)) + 1),
            )

            node.types['accum_t'].name = node.name + '_accum_t'
            node.types['accum_t'].precision = accum_type

            inferred_types.append('accum_t')

        if 'result_t' in types_to_infer:
            self._infer_output_matching_precision(node, ['result_t'])
            inferred_types.append('result_t')

        return inferred_types

    def _infer_merge_precision(self, node, types_to_infer):
        assert 'result_t' in types_to_infer and len(types_to_infer) == 1

        input_1 = node.get_input_variable(node.inputs[0]).type.precision
        input_2 = node.get_input_variable(node.inputs[1]).type.precision

        new_width = max(input_1.fractional, input_2.fractional) + max(input_1.integer, input_2.integer)
        new_int = max(input_1.integer, input_2.integer)

        out_precision = FixedPrecisionType(new_width, new_int)
        node.types['result_t'].name = node.name + '_result_t'
        node.types['result_t'].precision = out_precision

        return ['result_t']
