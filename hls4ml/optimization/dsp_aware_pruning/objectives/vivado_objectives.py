import logging
import math

import numpy as np

from hls4ml.optimization.dsp_aware_pruning.attributes import OptimizationAttributes
from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.dsp_aware_pruning.objectives import ObjectiveEstimator


# Optimizes DSP utilisation for Vivado backend
class VivadoDSPEstimator(ObjectiveEstimator):
    @classmethod
    def is_layer_optimizable(self, layer_attributes):
        if not layer_attributes.weight_shape or layer_attributes.args['hls4ml_attributes'].weight_precision.width < 9:
            return False, None
        else:
            if layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource':
                return True, OptimizationAttributes(
                    SUPPORTED_STRUCTURES.PATTERN,
                    pruning=True,
                    weight_sharing=False,
                    pattern_offset=np.prod(layer_attributes.weight_shape)
                    // layer_attributes.args['hls4ml_attributes'].reuse_factor,
                    consecutive_patterns=1,
                )
            else:
                return True, OptimizationAttributes(SUPPORTED_STRUCTURES.UNSTRUCTURED, pruning=True, weight_sharing=False)

    @classmethod
    def layer_resources(self, layer_attributes):
        if not layer_attributes.weight_shape or layer_attributes.args['hls4ml_attributes'].weight_precision.width < 9:
            return [0]
        else:
            # TOOD - Extend for parallelization factor
            return [np.prod(layer_attributes.weight_shape) // layer_attributes.args['hls4ml_attributes'].reuse_factor]

    @classmethod
    def layer_savings(self, layer_attributes):
        if not layer_attributes.weight_shape or layer_attributes.args['hls4ml_attributes'].weight_precision.width < 9:
            return [0]

        # TODO - Once we know how to implement constant coefficient multiplication via LUT, enable for weight sharing
        pruning = layer_attributes.optimization_attributes.pruning
        if not pruning:
            logging.warn(
                'Pruning needs to be enabled to decrease the number of DSPs used. \
                It is recommened to use the default attributes, returned from is_layer_optimizable(...)'
            )
            return [0]

        structure_type = layer_attributes.optimization_attributes.structure_type
        if layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency':
            if layer_attributes.args['hls4ml_attributes'].reuse_factor == 1:
                return [1]
            else:
                return [0]
        else:
            if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
                if layer_attributes.args['hls4ml_attributes'].reuse_factor == 1:
                    return [1]
                else:
                    return [0]
            elif structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                if (
                    layer_attributes.args['hls4ml_attributes'].reuse_factor
                    == layer_attributes.args['hls4ml_attributes'].n_in
                ):
                    return [1]
                else:
                    return [0]
            elif structure_type == SUPPORTED_STRUCTURES.PATTERN:
                pattern_offset = layer_attributes.optimization_attributes.pattern_offset
                number_of_patterns = np.prod(layer_attributes.weight_shape) // pattern_offset

                if number_of_patterns == layer_attributes.args['hls4ml_attributes'].reuse_factor:
                    return [layer_attributes.optimization_attributes.consecutive_patterns]
                else:
                    return [0]
            elif structure_type == SUPPORTED_STRUCTURES.BLOCK:
                logging.warn('hls4ml does not support block sparsity patterns...setting layer savings to zero')
                return [0]


# Optimizes BRAM and DSP for Vivado backend
class VivadoMultiObjectiveEstimator(ObjectiveEstimator):
    @classmethod
    def is_layer_optimizable(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return False, None

        if (
            layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
            and layer_attributes.args['hls4ml_attributes'].reuse_factor > 1
        ):
            if 36 % layer_attributes.args['hls4ml_attributes'].weight_precision.width == 0:
                consecutive_patterns = int(36 // layer_attributes.args['hls4ml_attributes'].weight_precision.width)
            else:
                consecutive_patterns = int(
                    math.ceil(2 * 36 / layer_attributes.args['hls4ml_attributes'].weight_precision.width)
                )

            return True, OptimizationAttributes(
                SUPPORTED_STRUCTURES.PATTERN,
                pruning=True,
                weight_sharing=False,
                pattern_offset=int(
                    np.prod(layer_attributes.weight_shape) // layer_attributes.args['hls4ml_attributes'].reuse_factor
                ),
                consecutive_patterns=consecutive_patterns,
            )
        else:
            return True, OptimizationAttributes(SUPPORTED_STRUCTURES.UNSTRUCTURED, pruning=True, weight_sharing=False)

    @classmethod
    def layer_resources(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]

        # TOOD - Extend for parallelization factor
        if layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency':
            return [
                int(np.prod(layer_attributes.weight_shape) // layer_attributes.args['hls4ml_attributes'].reuse_factor),
                0,
            ]
        else:
            # Resource strategy, RF == 1 is similar to Latency strategy (but slower)
            if layer_attributes.args['hls4ml_attributes'].reuse_factor == 1:
                return [
                    int(np.prod(layer_attributes.weight_shape) // layer_attributes.args['hls4ml_attributes'].reuse_factor),
                    0,
                ]
            else:
                # For RF > 1, BRAM utilised by weights can be estimated by (bit_width * n_in * n_out) / (RF * 36)
                return [
                    int(np.prod(layer_attributes.weight_shape) // layer_attributes.args['hls4ml_attributes'].reuse_factor),
                    int(
                        math.ceil(
                            np.prod(layer_attributes.weight_shape)
                            * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                            / (layer_attributes.args['hls4ml_attributes'].reuse_factor * 36)
                        )
                    ),
                ]

    @classmethod
    def layer_savings(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]

        # TODO - Once we know how to implement constant coefficient multiplication via LUT, enable for weight sharing
        pruning = layer_attributes.optimization_attributes.pruning
        if not pruning:
            logging.warn(
                'Pruning needs to be enabled to decrease the number of DSPs used. \
                It is recommened to use the default attributes, returned from is_layer_optimizable(...)'
            )
            return [0]

        structure_type = layer_attributes.optimization_attributes.structure_type
        if layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency':
            if layer_attributes.args['hls4ml_attributes'].reuse_factor == 1:
                return [1, 0]
            else:
                return [0, 0]
        else:
            if (
                layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
                and layer_attributes.args['hls4ml_attributes'].reuse_factor == 1
            ):
                return [1, 0]
            else:
                if structure_type == SUPPORTED_STRUCTURES.PATTERN:
                    pattern_offset = layer_attributes.optimization_attributes.pattern_offset
                    consecutive_patterns = layer_attributes.optimization_attributes.consecutive_patterns
                    weight_precision = layer_attributes.args['hls4ml_attributes'].weight_precision.width

                    number_of_patterns = np.prod(layer_attributes.weight_shape) // pattern_offset
                    saved_one_bram_block = (
                        36 == consecutive_patterns * weight_precision and 36 % weight_precision == 0
                    ) or (72 == consecutive_patterns * weight_precision)

                    if (
                        number_of_patterns == layer_attributes.args['hls4ml_attributes'].reuse_factor
                        and saved_one_bram_block
                    ):
                        return [consecutive_patterns, 1]
                    else:
                        logging.warn('Support for multi-objective optimisation is not fully implemented yet....')
                        return [0, 0]
                else:
                    logging.warn('Support for multi-objective optimisation is not fully implemented yet....')
                    return [0, 0]


class VivadoFFEstimator(ObjectiveEstimator):
    @classmethod
    def is_layer_optimizable(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return False, None

        # Resource strategy and I/O type io_stream store both weights and activation tensors in BRAM; skipping
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            logging.warn('FFs are at minimum utilization with io_stream and Resource strategy')
            return False, None

        # With io_stream in Latency, weight are stored in FFs, so unstructured pruning will benefit the most
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            return True, OptimizationAttributes(SUPPORTED_STRUCTURES.UNSTRUCTURED, pruning=True, weight_sharing=False)

        # In io_parallel with Resource, weights are stored in BRAM but activation tensors in FFs,
        # So structured pruning is the most suitable, it reduces the size out output before compile-time
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            return True, OptimizationAttributes(SUPPORTED_STRUCTURES.STRUCTURED, pruning=True, weight_sharing=False)

        # In io_parallel with Latency, weights and activation tensors are all stored in FFs,
        # So it is equivalent to unstructured, high sparsity pruning
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            return True, OptimizationAttributes(SUPPORTED_STRUCTURES.UNSTRUCTURED, pruning=True, weight_sharing=False)

    # TODO - This method is inaccurate (accross all cases); in general, estimating FFs is hard,
    # But as long as it is consistent(ly wrong), it should not matter for the pruning
    @classmethod
    def layer_resources(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]

        # Resource strategy and I/O type io_stream store both weights and activation tensors in BRAM; minimal FF utilization
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            return [0]

        # With io_stream in Latency, weight are stored in FFs, so FF ~ number_of_weights x weight_precision
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            return [
                np.prod(layer_attributes.weight_shape) * layer_attributes.args['hls4ml_attributes'].weight_precision.width
            ]

        # In io_parallel with Resource, weights are stored in BRAM but activation tensors in FFs,
        # So FF ~ number_of_outputs x weight_precision
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            return [
                np.prod(layer_attributes.output_shape) * layer_attributes.args['hls4ml_attributes'].output_precision.width
            ]

        # In io_parallel with Latency, weights and latency are all stored in FFs,
        # So it is equivalent to the sum of the above two cases
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            return [
                np.prod(layer_attributes.weight_shape) * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                + np.prod(layer_attributes.output_shape) * layer_attributes.args['hls4ml_attributes'].output_precision.width
            ]

    @classmethod
    def layer_savings(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]

        structure_type = layer_attributes.optimization_attributes.structure_type
        pruning = layer_attributes.optimization_attributes.pruning
        weight_sharing = layer_attributes.optimization_attributes.weight_sharing

        if weight_sharing:
            logging.warn(
                'Weight sharing does not decrease the number of parameters. \
                         It is recommened to use the default attributes, returned from is_layer_optimizable(...)'
            )
            return [0]

        if not pruning:
            logging.warn(
                'Pruning needs to be enabled to decrease the number of parameters. \
                         It is recommened to use the default attributes, returned from is_layer_optimizable(...)'
            )
            return [0]

        # Resource strategy and I/O type io_stream store both weights and activation tensors in BRAM; minimal FF utilization
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            return [0]

        # With io_stream in Latency, weight are stored in FFs, so any type of pruning will help:
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
                return [layer_attributes.args['hls4ml_attributes'].weight_precision.width]
            elif structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                return [
                    layer_attributes.args['hls4ml_attributes'].n_in
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]
            elif structure_type == SUPPORTED_STRUCTURES.PATTERN:
                number_of_patterns = (
                    np.prod(layer_attributes.weight_shape) // layer_attributes.optimization_attributes.pattern_offset
                )
                return [
                    number_of_patterns
                    * layer_attributes.optimization_attributes.consecutive_patterns
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]
            elif structure_type == SUPPORTED_STRUCTURES.BLOCK:
                return [
                    np.prod(layer_attributes.optimization_attributes.block_shape)
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]

        # In io_parallel with Resource, weights are stored in BRAM but activation tensors in FFs,
        # So only structured pruning helps
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            if structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                return [layer_attributes.args['hls4ml_attributes'].output_precision.width]
            else:
                return [0]

        # In io_parallel with Latency, weights and latency are all stored in FFs, so any type of pruning helps
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            # This is a significant under-estimate, as some savings are incurred due to less intermediate results
            if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
                return [layer_attributes.args['hls4ml_attributes'].weight_precision.width]
            elif structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                return [
                    layer_attributes.args['hls4ml_attributes'].n_in
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                    + layer_attributes.args['hls4ml_attributes'].output_precision.width
                ]
            elif structure_type == SUPPORTED_STRUCTURES.PATTERN:
                number_of_patterns = (
                    np.prod(layer_attributes.weight_shape) // layer_attributes.optimization_attributes.pattern_offset
                )
                return [
                    number_of_patterns
                    * layer_attributes.optimization_attributes.consecutive_patterns
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]
            elif structure_type == SUPPORTED_STRUCTURES.BLOCK:
                return [
                    np.prod(layer_attributes.optimization_attributes.block_shape)
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]
