import logging

import numpy as np

from hls4ml.optimization.dsp_aware_pruning.attributes import OptimizationAttributes
from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.dsp_aware_pruning.objectives import ObjectiveEstimator


class GPUFLOPEstimator(ObjectiveEstimator):
    @classmethod
    def is_layer_optimizable(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return False, None
        else:
            return True, OptimizationAttributes(
                structure_type=SUPPORTED_STRUCTURES.STRUCTURED, pruning=True, weight_sharing=False
            )

    @classmethod
    def layer_resources(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]
        else:
            if 'Dense' in layer_attributes.layer_type.__name__:
                return [2 * np.prod(layer_attributes.weight_shape) + layer_attributes.weight_shape[1]]
            elif 'Conv2D' in layer_attributes.layer_type.__name__:
                return [
                    2
                    * np.prod(layer_attributes.weight_shape)
                    * layer_attributes.output_shape[0]
                    * layer_attributes.output_shape[1]
                    + layer_attributes.weight_shape[3]
                ]
            else:
                raise Exception('Unknown layer encountered when estimating FLOP utilization.')

    @classmethod
    def layer_savings(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]

        structure_type = layer_attributes.optimization_attributes.structure_type
        pruning = layer_attributes.optimization_attributes.pruning
        weight_sharing = layer_attributes.optimization_attributes.weight_sharing

        if weight_sharing:
            logging.warn(
                'Weight sharing does not decrease FLOPs. \
                         It is recommened to use the default attributes, returned from is_layer_optimizable(...)'
            )
            return [0]

        if not pruning:
            logging.warn(
                'Pruning needs to be enabled to decrease FLOPs. \
                         It is recommened to use the default attributes, returned from is_layer_optimizable(...)'
            )
            return [0]

        # TODO - The below formulas underestimate FLOP savings
        # Removing a filter in a layer removes channels / neurons in subsequent layers
        if structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
            if 'Dense' in layer_attributes.layer_type.__name__:
                return [2 * layer_attributes.weight_shape[0] + 1]
            elif 'Conv2D' in layer_attributes.layer_type.__name__:
                return [
                    2
                    * np.prod(layer_attributes.weight_shape[0:3])
                    * layer_attributes.output_shape[0]
                    * layer_attributes.output_shape[1]
                    + 1
                ]
            else:
                raise Exception('Unknown layer encountered when estimating FLOP savings.')
        else:
            logging.warn(
                'FLOP savings occur with structured pruning. \
                         It is recommened to use the default attributes, returned from is_layer_optimizable(...)'
            )
            return [0]
