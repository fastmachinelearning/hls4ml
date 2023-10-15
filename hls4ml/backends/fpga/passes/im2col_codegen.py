from hls4ml.model.layers import Conv1D, Conv2D
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import Source


class GenerateConvIm2col(OptimizerPass):
    '''Generates tcode for im2col step of 1D/2d convolution'''

    def match(self, node):
        return isinstance(node, (Conv1D, Conv2D)) and node.model.config.get_config_value('IOType') == 'io_parallel'

    def transform(self, model, node):
        node_class = node.__class__.__name__
        if '1D' in node_class:
            self._generate_im2col_1d(node)
        elif '2D' in node_class:
            self._generate_im2col_2d(node)
        else:
            raise Exception(f'Cannot generate instructions for node {node.name} ({node_class})')

    def _generate_im2col_1d(self, node):
        code_str = node.model.config.backend.generate_conv1d_line_buffer_fn(
            node.get_attr('index'),
            node.get_attr('n_partitions'),
            node.get_input_variable().shape[0],
            node.get_input_variable().shape[1],
            kernel=node.get_attr('filt_width'),
            stride=node.get_attr('stride_width'),
            pad=(node.get_attr('pad_left'), node.get_attr('pad_right')),
        )

        node.set_attr('line_buffer_codegen', Source(code_str))

    def _generate_im2col_2d(self, node):
        code_str = node.model.config.backend.generate_conv2d_line_buffer_fn(
            node.get_attr('index'),
            node.get_attr('n_partitions'),
            node.get_input_variable().shape[0],
            node.get_input_variable().shape[1],
            node.get_input_variable().shape[2],
            kernel=(node.get_attr('filt_height'), node.get_attr('filt_width')),
            stride=(node.get_attr('stride_height'), node.get_attr('stride_width')),
            pad=(
                node.get_attr('pad_top'),
                node.get_attr('pad_bottom'),
                node.get_attr('pad_left'),
                node.get_attr('pad_right'),
            ),
        )

        node.set_attr('line_buffer_codegen', Source(code_str))
