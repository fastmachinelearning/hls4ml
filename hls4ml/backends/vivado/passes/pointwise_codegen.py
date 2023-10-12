from hls4ml.model.layers import Conv1D
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import Source


class GeneratePointwiseConv1D(OptimizerPass):
    '''Generates code for pointwise 1D convolution'''

    def match(self, node):
        return isinstance(node, Conv1D) and node.model.config.get_config_value('IOType') == 'io_parallel'

    def transform(self, model, node):
        node_class = node.__class__.__name__
        if '1D' in node_class:
            self._generate_pointwise_conv1d(node)
        else:
            raise Exception(f'Cannot generate instructions for node {node.name} ({node_class})')

    def _generate_pointwise_conv1d(self, node):
        code_str = node.model.config.backend.generate_pointwise_conv1d_fn(
            node.get_attr('index'),
            node.get_attr('reuse_factor'),
        )

        node.set_attr('pointwise_conv1d_codegen', Source(code_str))
