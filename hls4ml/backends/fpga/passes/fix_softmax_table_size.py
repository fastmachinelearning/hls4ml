from hls4ml.model.layers import Layer, Softmax
from hls4ml.model.optimizer import OptimizerPass


class FixSoftmaxTableSize(OptimizerPass):
    def match(self, node):
        if not isinstance(node, Softmax):
            return False
        if node.get_attr('table_sizes_checked'):
            return False  # This optimizer has already run
        return True

    def transform(self, model, node: Layer):
        inp_layer = node.get_input_node()  # type: ignore
        if not isinstance(inp_layer, Layer):
            raise RuntimeError(f'Softmax layer {node.name} does not have an input layer')

        table_size = int(node.get_attr('table_size'))  # type: ignore
        exp_table_size = node.get_attr('exp_table_size', table_size)
        inv_table_size = node.get_attr('inv_table_size', table_size)

        implemenation = node.get_attr('implementation')

        if implemenation == 'stable':
            inp_norm_bw = node.get_attr('inp_norm_t').precision.width
            node.set_attr('exp_table_size', min(2**inp_norm_bw, exp_table_size))

            inv_inp_bw = node.get_attr('inv_inp_t').precision.width
            node.set_attr('inv_table_size', min(2**inv_inp_bw, inv_table_size))

        elif implemenation == 'latency':
            input_bw = inp_layer.get_attr('result_t').precision.width
            node.set_attr('exp_table_size', min(2**input_bw, exp_table_size))

            accum_bw = node.get_attr('accum_t').precision.width
            node.set_attr('inv_table_size', min(2**accum_bw, inv_table_size))

        # One could try shrinking the size of legacy, but ignore it for now.
        # Argmax doesn't use tables so the size is irrelevant in that case.

        node.set_attr('table_sizes_checked', True)

        return False


def register_softmax__table_size_fix(backend):
    backend.register_pass('fix_softmax_table_size', FixSoftmaxTableSize)
