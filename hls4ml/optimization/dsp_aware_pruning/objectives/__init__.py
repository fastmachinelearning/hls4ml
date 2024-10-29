import logging
from abc import ABC, abstractmethod

import numpy as np

from hls4ml.optimization.dsp_aware_pruning.attributes import OptimizationAttributes
from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES

'''
Pruning & weight sharing are formulated as an optimization problem, with the aim of minimizing some metric
Metrics can include: total number of weights, DSP utilization, latency, FLOPs etc.
'''


class ObjectiveEstimator(ABC):
    '''
    Abstract class with methods for estimating the utilization and savings of a certain layer, with respect to some objective
    For each objective, an inherited class is written with the correct implementation of the below methods
    The objectives can be multi-dimensional, e.g. DSPs and BRAM
    Care needs to be taken when optimizing several objectives, especially if conflicting
    '''

    @abstractmethod
    def is_layer_optimizable(self, layer_attributes):
        '''
        For a given layer, checks whether optimizations make sense, with respect to the given objective(s)
        Furthermore, it returns the type of optimization (structured, unstructured etc.)
        Most suitable for minimizing the objective(s).

        Args:
            layer_attributes (hls4ml.optimization.attributes.LayerAttributes): Layer attributes

        Returns:
            tuple containing

            - optimizable (boolean): can optimizations be applied to this layer
            - optimization_attributes (hls4ml.optimization.attributes.OptimizationAttributes):
                Most suitable approach for optimization

        Examples:
            - Metric = Total weights, Layer = Dense, shape = (4, 4) -> return True, unstructured
            - Metric = DSP, Layer = Dense, Precision = ap_fixed<8, 0> -> return False
                (Vivado doesn't use DSP when precision < 9)
            - Metric = DSP, Layer = Dense, Precision = ap_fixed<16, 6> ->
                return True, pattern structure, both pruning and weight sharing
        '''
        pass

    @abstractmethod
    def layer_resources(self, layer_attributes):
        '''
        For a given layer, how many units of the metric are used, given a generic weight matrix

        Args:
            layer_attributes (hls4ml.optimization.attributes.LayerAttributes): Layer attributes

        Returns:
            resources (list, int): total resources (w.r.t every dimension of the objective) used

        Example:
            Metric = Total weights, Layer = Dense, shape = (4, 4) -> return [16] [regardless of layer sparsity]
        '''
        pass

    @abstractmethod
    def layer_savings(self, layer_attributes):
        '''
        For a given layer, how many units of the metric are saved, when optimizing one structure
        The structure type, alongside its parameters (e.g. block shape) are stored in layer attributes
        For best results, OptimizationAttributes in layer_attribtues should be obtained from is_layer_optimizable

        Args:
            layer_attributes (hls4ml.optimization.attributes.LayerAttributes): Layer attributes

        Returns:
            savings (list, int): savings achieved (one for every dimension of objective)
                With OptimizationAttributes from layer_attributes

        Example:
            Metric = Total weights, Layer = Dense, shape = (4, 4):
            - structure_type == unstructured -> return [1]
            - structure_type == structured -> return [4]
        '''
        pass


class ParameterEstimator(ObjectiveEstimator):
    '''
    A class containing objective estimation with the goal of minimizing
    The number of non-zero weights in a layer [corresponds to unstructured pruning]
    '''

    @classmethod
    def is_layer_optimizable(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return False, None
        else:
            return True, OptimizationAttributes(SUPPORTED_STRUCTURES.UNSTRUCTURED, pruning=True, weight_sharing=False)

    @classmethod
    def layer_resources(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]
        else:
            return [np.prod(layer_attributes.weight_shape)]

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

        # In this case, pruning = True and weight_sharing = False,
        # So calculate savings incurred by removing a group of weights
        if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
            return [1]
        elif structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
            if 'Dense' in layer_attributes.layer_type.__name__ or 'Conv2D' in layer_attributes.layer_type.__name__:
                return [np.prod(layer_attributes.weight_shape[:-1])]
            else:
                raise Exception('Unknown layer encountered when estimating parameter savings.')
        elif structure_type == SUPPORTED_STRUCTURES.PATTERN:
            number_of_patterns = (
                np.prod(layer_attributes.weight_shape) // layer_attributes.optimization_attributes.pattern_offset
            )
            return [number_of_patterns * layer_attributes.optimization_attributes.consecutive_patterns]
        elif structure_type == SUPPORTED_STRUCTURES.BLOCK:
            return [np.prod(layer_attributes.optimization_attributes.block_shape)]
