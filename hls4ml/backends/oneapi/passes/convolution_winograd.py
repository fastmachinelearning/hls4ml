import math

import numpy as np

from hls4ml.model.layers import Conv1D, Conv2D
from hls4ml.model.optimizer import OptimizerPass


class ApplyWinogradKernelTransformation(OptimizerPass):
    '''
    Transforms the weights of a Conv2D kernel to a format suitable for Wingorad convolution
    For further information, refer to Lavin & Gray, 2015 - Fast Algorithms for Convolutional Neural Networks
    '''

    def match(self, node):
        node_matches = isinstance(node, (Conv1D, Conv2D))

        # This optimizer works only after the Resource Strategy Optimizer, since order of transposition matters
        weights_transformed = node.get_attr('_weights_transposed', False) is True

        # User opted for Winograd
        implementation_is_winograd = (
            node.get_attr('implementation', 'combination') == 'combination'
            or node.get_attr('implementation', 'combination') == 'winograd'
        )

        parallel_io_type = node.model.config.get_config_value('IOType') == 'io_parallel'

        # Winograd algorithm-specific conditions
        if isinstance(node, Conv1D):
            # Winograd only applies to specific kernel sizes
            # Current implementation only supports fs = 3; easily extendable to other filter sizes
            filter_size_matches = node.get_attr('filt_width', 3) == 3

            # Winograd's minimal filtering algorithm doesn't work with stride != 1
            stride_is_one = node.get_attr('stride_width', 1) == 1

            # HLS Compiler fails to pipeline the entire component if Winograd loop only executes once
            loop_itr_gt_one = node.get_attr('out_width') > 2

            winograd_conditions = filter_size_matches and stride_is_one and loop_itr_gt_one and parallel_io_type

        elif isinstance(node, (Conv2D)):
            # Winograd only applies to specific kernel sizes
            # Current implementation only supports fs = 3; easily extendable to other filter sizes
            filter_size_matches = node.get_attr('filt_height', 3) == 3 and node.get_attr('filt_width', 3) == 3

            # Winograd's minimal filtering algorithm doesn't work with striede != 1
            stride_is_one = node.get_attr('stride_height', 1) == 1 and node.get_attr('stride_width', 1) == 1

            # HLS Compiler fails to pipeline the entire component if Winograd loop only executes once
            loop_itr_gt_one = node.get_attr('out_height') > 2 and node.get_attr('out_width') > 2

            padding_is_equal = node.get_attr('pad_top', 0) == node.get_attr('pad_bottom', 0) and node.get_attr(
                'pad_left', 0
            ) == node.get_attr('pad_right', 0)

            winograd_conditions = (
                filter_size_matches and stride_is_one and padding_is_equal and loop_itr_gt_one and parallel_io_type
            )

        else:
            winograd_conditions = False

        # Check any previous transformations
        already_transformed = node.get_attr('_winograd_transformation_applied', False) is True

        if not winograd_conditions and node.get_attr('implementation', 'combination') == 'winograd':
            raise RuntimeError(
                'Not possible to use Winograd algorithm with current architecture. '
                'Please set implementation to im2col or combination'
            )

        return (
            node_matches
            and weights_transformed
            and winograd_conditions
            and not already_transformed
            and implementation_is_winograd
        )

    def transform(self, model, node):
        if isinstance(node, Conv1D):
            if node.get_attr('filt_width', 3) == 3:
                # First, transpose to a format suitable for the Winograd algorithm (F, C, W)
                # Note, this assumes a format post-resource strategy optimizer, that is (F, W, C)
                # Therefore, (F, W, C) => (F, C, W)
                node.weights['weight'].data = np.transpose(node.weights['weight'].data, axes=[0, 2, 1])

                # Temporary copy of data
                weights = node.weights['weight'].data

                # Expand weight dimensionality (3) => (4)
                node.weights['weight'].data = np.zeros((weights.shape[0], weights.shape[1], 4))

                # Transformation matrices for 3x1 kernels
                G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]])

                # Transformation GfG'
                for filter in range(0, weights.data.shape[0]):
                    for channel in range(0, weights.data.shape[1]):
                        node.weights['weight'].data[filter][channel] = np.matmul(G, weights[filter][channel])
                        node.weights['weight'].data_length = node.weights['weight'].data.size

                # Winograd's minimal filtering algorithm transforms the weight matrix
                # This transformation consists of addition and division (by 2&4) of the weight matrix
                # Therefore, increase precision (if needed), to accomodate for new weights
                # This error is only noticeable for low precisions, such as those used with QKeras

                # Integer precision is only updated if it exceeds the one defined in hls4ml config
                maximum_value_rounded = int(math.ceil(np.abs(node.weights['weight'].data).max()))
                if maximum_value_rounded.bit_length() + 1 > node.weights['weight'].type.precision.integer:
                    node.weights['weight'].type.precision.integer = maximum_value_rounded.bit_length() + 1
                    node.weights['weight'].type.precision.width += (
                        maximum_value_rounded.bit_length() + 1 - node.weights['weight'].type.precision.integer
                    )

                # Fractional precision is increased by 2 bits (division by 4),
                # for low-precision (less than 8) fractional weights
                if node.weights['weight'].type.precision.fractional < 8:
                    node.weights['weight'].type.precision.fractional += 2
                    node.weights['weight'].type.precision.width += 2

                # Modified kernel size
                node.set_attr('impl_filt_width', 4)

        elif isinstance(node, Conv2D):
            if node.get_attr('filt_height', 3) == 3 and node.get_attr('filt_width', 3) == 3:
                # First, transpose to a format suitable for the Winograd algorithm (F, C, H, W)
                # Note, this assumes a format post-resource strategy optimizer, that is (F, H, W, C)
                # Therefore, (F, H, W, C) => (F, C, H, W)
                node.weights['weight'].data = np.transpose(node.weights['weight'].data, axes=[0, 3, 1, 2])

                # Temporary copy of data
                weights = node.weights['weight'].data

                # Expand weight dimensionality (3x3) => (4x4)
                node.weights['weight'].data = np.zeros((weights.shape[0], weights.shape[1], 4, 4))

                # Transformation matrices for 3x3 kernels
                G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]])
                GT = np.array([[1, 0.5, 0.5, 0], [0, 0.5, -0.5, 0], [0, 0.5, 0.5, 1]])

                # Transformation GfG'
                for filter in range(0, weights.data.shape[0]):
                    for channel in range(0, weights.data.shape[1]):
                        node.weights['weight'].data[filter][channel] = np.matmul(np.matmul(G, weights[filter][channel]), GT)
                        node.weights['weight'].data_length = node.weights['weight'].data.size

                # Winograd's minimal filtering algorithm transforms the weight matrix
                # This transformation consists of addition and division (by 2&4) of the weight matrix
                # Therefore, increase precision (if needed), to accomodate for new weights
                # This error is only noticeable for low precisions, such as those used with QKeras

                # Integer precision is only updated if it exceeds the one defined in hls4ml config
                maximum_value_rounded = int(math.ceil(np.abs(node.weights['weight'].data).max()))
                if maximum_value_rounded.bit_length() + 1 > node.weights['weight'].type.precision.integer:
                    node.weights['weight'].type.precision.integer = maximum_value_rounded.bit_length() + 1
                    node.weights['weight'].type.precision.width += (
                        maximum_value_rounded.bit_length() + 1 - node.weights['weight'].type.precision.integer
                    )

                # Fractional precision is increased by 2 bits (division by 4),
                # for low-precision (less than 8) fractional weights
                if node.weights['weight'].type.precision.fractional < 8:
                    node.weights['weight'].type.precision.fractional += 2
                    node.weights['weight'].type.precision.width += 2

                # Modified kernel size
                node.set_attr('impl_filt_height', 4)
                node.set_attr('impl_filt_width', 4)
        else:
            raise Exception(f'Unexpected layer {node.class_name} with Winograd kernel optimizer')

        node.set_attr('_winograd_transformation_applied', True)

        return False
