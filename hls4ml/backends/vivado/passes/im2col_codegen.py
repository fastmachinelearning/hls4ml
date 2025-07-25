from hls4ml.model.layers import Conv1D, Conv2D, SeparableConv1D, SeparableConv2D
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import Source


class GenerateConvIm2col(OptimizerPass):
    '''Generates tcode for im2col step of 1D/2d convolution'''

    # Note, DepthwizeConv1D/2D also matches because it inherits from Conv1D/2D
    def match(self, node):
        return (
            isinstance(node, (Conv1D, Conv2D, SeparableConv1D, SeparableConv2D))
            and node.model.config.get_config_value('IOType') == 'io_parallel'
        )

    def transform(self, model, node):
        node_class = node.class_name
        if 'Separable' in node_class:
            if '1D' in node_class:
                self._generate_separable_im2col_1d(node)
            elif '2D' in node_class:
                self._generate_separable_im2col_2d(node)
            else:
                raise Exception(f'Cannot generate instructions for node {node.name} ({node_class})')
        else:
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

    def _generate_separable_im2col_1d(self, node):
        dw_code_str = node.model.config.backend.generate_conv1d_line_buffer_fn(
            str(node.get_attr('index')) + '_dw',
            node.get_attr('n_partitions'),
            node.get_input_variable().shape[0],
            node.get_input_variable().shape[1],
            kernel=node.get_attr('filt_width'),
            stride=node.get_attr('stride_width'),
            pad=(node.get_attr('pad_left'), node.get_attr('pad_right')),
        )

        node.set_attr('dw_line_buffer_codegen', Source(dw_code_str))

        pw_code_str = node.model.config.backend.generate_conv1d_line_buffer_fn(
            str(node.get_attr('index')) + '_pw',
            node.get_attr('n_partitions'),
            node.get_output_variable().shape[0],
            node.get_input_variable().shape[1],
            kernel=1,
        )

        node.set_attr('pw_line_buffer_codegen', Source(pw_code_str))

    def _generate_separable_im2col_2d(self, node):
        dw_code_str = node.model.config.backend.generate_conv2d_line_buffer_fn(
            str(node.get_attr('index')) + '_dw',
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

        node.set_attr('dw_line_buffer_codegen', Source(dw_code_str))

        pw_code_str = node.model.config.backend.generate_conv2d_line_buffer_fn(
            str(node.get_attr('index')) + '_pw',
            node.get_attr('n_partitions'),
            node.get_output_variable().shape[0],
            node.get_output_variable().shape[1],
            node.get_input_variable().shape[2],
            kernel=(1, 1),
        )

        node.set_attr('pw_line_buffer_codegen', Source(pw_code_str))
