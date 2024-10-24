import logging
import sys

import numpy as np
import tensorflow as tf
from qkeras import QConv2D, QDense
from tensorflow.keras.layers import Conv2D, Dense

from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.dsp_aware_pruning.keras.config import SUPPORTED_LAYERS, SUPPORTED_METRICS
from hls4ml.optimization.dsp_aware_pruning.knapsack import solve_knapsack


def get_model_masks(
    keras_model,
    model_attributes,
    sparsity,
    objective,
    metric='l1',
    local=False,
    gradients=None,
    hessians=None,
    knapsack_solver='CBC_MIP',
):
    '''
    Function calculating a binary mask for all optimizable layers
    Entries equal to one correspond to the weight being updated during the training
    Entries equal to zero correspond to the weight being frozen during the training

    Masking is such that:
        * resource_utilization <= (1 - sparsity) * baseline_utilization OR
        * resource_saving > sparsity * baseline_utilization [equivalent formulation]

    Offsets are used for weight sharing - in the case of weight sharing, the mask is set to zero
    Therefore, the weights will be frozen during training; however, they still need to be the mean of the group
    Offsets represent the mean of each weight-shared group - therefore, it is important to have offsets only for
    frozen weights; that is where the corresponding entry in the mask tensor is zero

    If a layer supports both weight sharing and pruning, both the norm and variance of the group are calculated
    And the smaller one is considered; so if the norm is smaller, the group will be considered for pruning
    Otherise, the group will be considered for weight sharing.
    Both the norm and variance are normalized, to avoid magnitude biases.

    Args:
        keras_model (keras.model): Model to be masked
        model_attributes (dict): A layer-wise dictionary of LayerAttributes classes
        sparsity (float): Desired sparsity, with respect to the objective
        objective (ObjectiveEstimator): Objective to be minimized (e.g. DSP, FLOPs etc.)
        metric (string): Weight ranking metric - l1, l2, Oracle, saliency
        local (boolean): Equal layer-wise sparsity
        gradients (dict): A layer-wise dictionary of weight gradients
            (needed for Oracle ranking)
        hessians (dict): A layer-wise dictionary of second gradients
            (needed for saliency ranking)
        knapsack_solver (str): Algorithm for solving Knapsack problem; recommended is to use default.
            Unless dealing with highly dimensional problems, in which case greedy is better.

    Returns:
        tuple containing

        - masks (dict): Layer-wise dictionary of binary tensors
        - offsets (dict): Layer-wise dictionary of offsets for every weight
    '''

    if metric not in SUPPORTED_METRICS:
        raise Exception('Unknown metric for ranking weights')

    if metric == 'oracle' and gradients is None:
        raise Exception('Oracle ranking requires the gradient of the loss with respect to model weights')

    if metric == 'saliency' and hessians is None:
        raise Exception('Saliency ranking requires second order derivatives')

    if local:
        return __get_masks_local(
            keras_model, model_attributes, sparsity, objective, metric, gradients, hessians, knapsack_solver
        )
    else:
        return __get_masks_global(
            keras_model, model_attributes, sparsity, objective, metric, gradients, hessians, knapsack_solver
        )


def __get_masks_local(keras_model, model_attributes, sparsity, objective, metric, gradients, hessians, knapsack_solver):
    '''
    Function calculating a layer-wise binary mask for all optimizable layers
    This function performs layer-wise masking, so all layers have the same sparsity (with respect to the objective)
    '''
    masks = {}
    offsets = {}

    for layer in keras_model.layers:
        # Care needs to be taken if layer_savings = 0
        # As long as the default attributes are used (from is_layer_optimizable(...)), this should never happen
        # However, if the optimization attributes are manually changed,
        # It would be possible to select the structure type such that savings = 0
        # In this case, the goal is to keep resource utilization under a certain threshold;
        # Savings are equivalent to resources per single group
        # Therefore, in the knapsack solver, zero-saving would remain unmasked;
        # As it has a "weight" of zero, it is always stored in the knapsack
        # So not masking a group without saving is as expected;
        # However, if solved through greedy knaspack an exception will be thrown (division by zero)
        if isinstance(layer, SUPPORTED_LAYERS) and model_attributes[layer.name].optimizable:
            layer_savings = objective.layer_savings(model_attributes[layer.name])
            layer_resources = objective.layer_resources(model_attributes[layer.name])
            target_resources = ((1 - sparsity) * np.array(layer_resources)).astype(int)
            structure_type = model_attributes[layer.name].optimization_attributes.structure_type

            # All the groups (structures, patterns, blocks) have the same resource utilisation in one layer
            # So we can greedily prune the groups with the lowest "loss" (magnitude, saliency etc.)
            # Greedily pruning the groups with the lowest loss is a special case of the Knapsack problem with equal weights
            value = layer.get_weights()[0]
            if metric == 'oracle':
                value = np.abs(np.multiply(value, gradients[layer.name]))
                norm = 1
            elif metric == 'saliency':
                value = np.multiply(np.square(value), hessians[layer.name])
                norm = 1
            elif metric == 'l1':
                norm = 1
            else:
                norm = 2

            if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
                if model_attributes[layer.name].optimization_attributes.weight_sharing:
                    logging.warn('Weight sharing not suitable for unstructured pruning. Ignoring....')

                if model_attributes[layer.name].optimization_attributes.pruning:
                    # Since no norm is taken, calculate absolute value [avoids removing large negative weights]
                    value = np.abs(value)

                    # Get all posible indices in the weight tensor
                    indices = np.indices(value.shape).reshape(value.ndim, -1).T

                    # Find weights with the lowest loss
                    groups = []
                    for i in indices:
                        groups.append(__WeightGroups__(value[tuple(i)], layer_savings, tuple(i)))
                    _, selected = solve_knapsack(
                        np.array([g.value for g in groups]),
                        np.array([g.resources for g in groups]).T,
                        target_resources,
                        implementation=knapsack_solver,
                    )

                    # Selected weights are not masked
                    mask = np.zeros(value.shape, dtype=value.dtype)
                    for i in selected:
                        mask[groups[i].layer_position] = 1

                    # Offsets are always zero (weight sharing not applicable to unstructured)
                    masks[layer.name] = mask
                    offsets[layer.name] = np.zeros(value.shape, dtype=value.dtype)

            if structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                # Dense -> Masking neurons (columns)
                if isinstance(layer, (Dense, QDense)):
                    # If pruning enabled, find cost associated with pruning each neuron
                    if model_attributes[layer.name].optimization_attributes.pruning:
                        vals_norm = np.linalg.norm(value, axis=0, ord=norm)
                    else:
                        vals_norm = np.full((value.shape[1]), sys.float_info.max, dtype=value.dtype)

                    # If weight sharing enabled, find cost asociated with quantizing nerons to their mean
                    if model_attributes[layer.name].optimization_attributes.weight_sharing:
                        vals_var = np.var(value, axis=0)
                    else:
                        vals_var = np.full((value.shape[1]), sys.float_info.max, dtype=value.dtype)

                    # Choose min(pruning, weight sharing)
                    groups = []
                    for i in range(vals_norm.shape[0]):
                        if vals_norm[i] <= vals_var[i]:
                            groups.append(__WeightGroups__(vals_norm[i], layer_savings, i, optimization_type='pruning'))
                        else:
                            groups.append(__WeightGroups__(vals_var[i], layer_savings, i, optimization_type='sharing'))

                    # Select neurons with the lowest loss
                    _, selected = solve_knapsack(
                        np.array([g.value for g in groups]),
                        np.array([g.resources for g in groups]).T,
                        target_resources,
                        implementation=knapsack_solver,
                    )

                    # Selected neurons are not masked
                    mask = np.zeros(value.shape, value.dtype)
                    for i in selected:
                        mask[:, groups[i].layer_position] = 1

                    # Masked neurons can either be pruned or quantized
                    # If quantized, add the corresponding offset
                    offset = np.zeros(value.shape, value.dtype)
                    zeros = np.where(~np.all(mask, axis=0))[0]
                    for i in zeros:
                        if groups[i].optimization_type == 'sharing':
                            offset[:, i] = np.mean(layer.get_weights()[0][:, i])
                    masks[layer.name] = mask
                    offsets[layer.name] = offset

                # Conv2D -> Masking filters (W x H x C)
                elif isinstance(layer, (Conv2D, QConv2D)):
                    if model_attributes[layer.name].optimization_attributes.pruning:
                        vals_norm = np.linalg.norm(np.linalg.norm(value, axis=(0, 1), ord='fro'), axis=0, ord=norm)
                    else:
                        vals_norm = np.full((value.shape[3]), sys.float_info.max, dtype=value.dtype)

                    if model_attributes[layer.name].optimization_attributes.weight_sharing:
                        vals_var = np.var(np.linalg.norm(value, axis=(0, 1), ord='fro'), axis=0)
                    else:
                        vals_var = np.full((value.shape[3]), sys.float_info.max, dtype=value.dtype)

                    groups = []
                    for i in range(vals_norm.shape[0]):
                        if vals_norm[i] <= vals_var[i]:
                            groups.append(__WeightGroups__(vals_norm[i], layer_savings, i, optimization_type='pruning'))
                        else:
                            groups.append(__WeightGroups__(vals_var[i], layer_savings, i, optimization_type='sharing'))

                    _, selected = solve_knapsack(
                        np.array([g.value for g in groups]),
                        np.array([g.resources for g in groups]).T,
                        target_resources,
                        implementation=knapsack_solver,
                    )

                    mask = np.zeros(value.shape, value.dtype)
                    for i in selected:
                        mask[:, :, :, groups[i].layer_position] = 1

                    offset = np.zeros(value.shape, value.dtype)
                    zeros = np.where(~np.all(mask, axis=(0, 1, 2)))[0]
                    for i in zeros:
                        if groups[i].optimization_type == 'sharing':
                            offset[:, :, :, i] = np.mean(layer.get_weights()[0][:, :, :, i])

                    masks[layer.name] = mask
                    offsets[layer.name] = offset

            if structure_type == SUPPORTED_STRUCTURES.PATTERN:
                pattern_offset = model_attributes[layer.name].optimization_attributes.pattern_offset
                consecutive_patterns = model_attributes[layer.name].optimization_attributes.consecutive_patterns

                if (np.prod(value.shape)) % pattern_offset != 0:
                    raise Exception('Pattern offset needs to be a factor of matrix size')

                if pattern_offset % consecutive_patterns != 0:
                    raise Exception('Consecutive patterns need to be a factor of matrix size')

                # Transpose, as done in hls4ml Resource strategy
                if isinstance(layer, (Dense, QDense)):
                    value = value.T
                    transposed_shape = value.shape
                elif isinstance(layer, (Conv2D, QConv2D)):
                    value = np.transpose(value, axes=[3, 0, 1, 2])
                    transposed_shape = value.shape

                # Reshape weight matrix into [number_of_patterns, pattern_offset]
                # Note, swapping the axis will mess up the weight order
                # In the case of hls4ml, number_of_patterns is equivalent to reuse factor
                # And, pattern_offset, is the number of multiplications done in parallel
                number_of_patterns = np.prod(transposed_shape) // pattern_offset
                target_shape = (pattern_offset, number_of_patterns)
                reshaped = np.reshape(value, target_shape)

                # Group consecutive patterns (rows) into blocks and reshape
                total_blocks = pattern_offset // consecutive_patterns
                blocks = np.reshape(
                    tf.image.extract_patches(
                        np.expand_dims(np.expand_dims(reshaped, 2), 0),
                        [1, consecutive_patterns, number_of_patterns, 1],
                        [1, consecutive_patterns, number_of_patterns, 1],
                        [1, 1, 1, 1],
                        'SAME',
                    ).numpy(),
                    (total_blocks, -1),
                )

                # If pruning enabled, find cost associated with pruning each neuron
                if model_attributes[layer.name].optimization_attributes.pruning:
                    vals_norm = np.linalg.norm(blocks, axis=1, ord=norm)
                else:
                    vals_norm = np.full((blocks.shape[0],), sys.float_info.max, dtype=value.dtype)

                # If weight sharing enabled, find cost asociated with quantizing nerons to their mean
                if model_attributes[layer.name].optimization_attributes.weight_sharing:
                    vals_var = np.var(blocks, axis=1)
                else:
                    vals_var = np.full((blocks.shape[0],), sys.float_info.max, dtype=value.dtype)

                # Choose min(pruning, weight sharing)
                groups = []
                for i in range(vals_norm.shape[0]):
                    if vals_norm[i] <= vals_var[i]:
                        groups.append(__WeightGroups__(vals_norm[i], layer_savings, i, optimization_type='pruning'))
                    else:
                        groups.append(__WeightGroups__(vals_var[i], layer_savings, i, optimization_type='sharing'))

                # Select groups with highest importance
                _, selected = solve_knapsack(
                    np.array([g.value for g in groups]),
                    np.array([g.resources for g in groups]).T,
                    target_resources,
                    implementation=knapsack_solver,
                )

                # Decode masked groups into transposed shape and set selected groups to one
                mask = np.zeros((np.prod(transposed_shape),), value.dtype)
                for i in selected:
                    pos = i * number_of_patterns * consecutive_patterns
                    for j in range(pos, pos + number_of_patterns * consecutive_patterns, number_of_patterns):
                        mask[range(j, j + number_of_patterns)] = 1

                        # Decode offset
                not_selected = [i for i in range(len(groups)) if i not in selected]
                offset = np.zeros((np.prod(transposed_shape),), value.dtype)
                for i in not_selected:
                    if groups[i].optimization_type == 'sharing':
                        mean = np.mean(blocks[i, :])
                        pos = i * number_of_patterns * consecutive_patterns
                        for j in range(pos, pos + number_of_patterns * consecutive_patterns, number_of_patterns):
                            offset[range(j, j + number_of_patterns)] = mean

                            # Reshape into original shape and store result
                if isinstance(layer, (Dense, QDense)):
                    mask = np.reshape(mask, transposed_shape).T
                    offset = np.reshape(offset, transposed_shape).T
                elif isinstance(layer, (Conv2D, QConv2D)):
                    mask = np.transpose(np.reshape(mask, transposed_shape), (1, 2, 3, 0))
                    offset = np.transpose(np.reshape(offset, transposed_shape), (1, 2, 3, 0))
                masks[layer.name] = mask
                offsets[layer.name] = offset

            if structure_type == SUPPORTED_STRUCTURES.BLOCK:
                if len(value.shape) != 2:
                    raise Exception('Block pruning is supported for 2-dimensional weight matrices')

                block_shape = model_attributes[layer.name].optimization_attributes.block_shape
                if (value.shape[0] % block_shape[0]) != 0 or (value.shape[1] % block_shape[1] != 0):
                    raise Exception('Block sizes need to be fators of weight matrix dimensions')

                # TensorFlow has a built-in method for exctracting sub-tensors of given shape and stride
                # This method is commonly used to perform im2col,
                # Docs: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
                total_blocks = (value.shape[0] * value.shape[1]) // (block_shape[0] * block_shape[1])
                blocks_in_row = value.shape[1] // block_shape[1]
                blocks = np.reshape(
                    tf.image.extract_patches(
                        np.expand_dims(np.expand_dims(value, 2), 0),
                        [1, block_shape[0], block_shape[1], 1],
                        [1, block_shape[0], block_shape[1], 1],
                        [1, 1, 1, 1],
                        'SAME',
                    ).numpy(),
                    (total_blocks, block_shape[0] * block_shape[1]),
                )

                # If pruning enabled, find cost associated with pruning each neuron
                if model_attributes[layer.name].optimization_attributes.pruning:
                    vals_norm = np.linalg.norm(blocks, axis=1, ord=norm)
                else:
                    vals_norm = np.full((blocks.shape[0],), sys.float_info.max, dtype=value.dtype)

                # If weight sharing enabled, find cost asociated with quantizing nerons to their mean
                if model_attributes[layer.name].optimization_attributes.weight_sharing:
                    vals_var = np.var(blocks, axis=1)
                else:
                    vals_var = np.full((blocks.shape[0],), sys.float_info.max, dtype=value.dtype)

                # Choose min(pruning, weight sharing)
                groups = []
                for i in range(vals_norm.shape[0]):
                    if vals_norm[i] <= vals_var[i]:
                        groups.append(__WeightGroups__(vals_norm[i], layer_savings, i, optimization_type='pruning'))
                    else:
                        groups.append(__WeightGroups__(vals_var[i], layer_savings, i, optimization_type='sharing'))

                # Select groups with highest importance
                _, selected = solve_knapsack(
                    np.array([g.value for g in groups]),
                    np.array([g.resources for g in groups]).T,
                    target_resources,
                    implementation=knapsack_solver,
                )

                # Decode position of masked weights and set selected weights to one
                mask = np.zeros(value.shape, value.dtype)
                for i in selected:
                    row = block_shape[0] * (i // blocks_in_row)
                    col = block_shape[1] * (i % blocks_in_row)
                    cols = np.linspace(col, col + block_shape[1], block_shape[1], endpoint=False, dtype=np.int32)
                    rows = np.linspace(row, row + block_shape[0], block_shape[0], endpoint=False, dtype=np.int32)
                    zeros = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)
                    mask[zeros[:, 0], zeros[:, 1]] = 1

                # Calculate offset
                not_selected = [i for i in range(len(groups)) if i not in selected]
                offset = np.zeros(value.shape, value.dtype)
                for i in not_selected:
                    if groups[i].optimization_type == 'sharing':
                        mean = np.mean(blocks[i, :])
                        row = block_shape[0] * (i // blocks_in_row)
                        col = block_shape[1] * (i % blocks_in_row)
                        cols = np.linspace(col, col + block_shape[1], block_shape[1], endpoint=False, dtype=np.int32)
                        rows = np.linspace(row, row + block_shape[0], block_shape[0], endpoint=False, dtype=np.int32)
                        pos = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)
                        offset[pos[:, 0], pos[:, 1]] = mean

                masks[layer.name] = mask
                offsets[layer.name] = offset

    return masks, offsets


def __get_masks_global(keras_model, model_attributes, sparsity, objective, metric, gradients, hessians, knapsack_solver):
    '''
    Function calculating a layer-wise binary mask for all optimizable layers
    Global masking, with layers of different sparsity; masks are calculated by solving a Knapsack problem
    Most of the logic remains similar to local masking; comments describing implementation are given in the function above
    '''
    groups = []
    total_resources = []

    # Iterate through all layers and create a list of all the optimizable groups (single weight, structure, pattern, block)
    # Each entry contains the value associated with the group,
    # Alongside the layer it belongs to and its location in the layer
    # The values is normalised w.r.t to the to largest element in the group, to avoid bias towards large layers
    # We also keep track of total model resources, with respect to the objective
    # A detailed comment in the local masking function is given for
    # Considerations on exception of layer savings and how to address them
    for layer in keras_model.layers:
        # Optimizable should be always enabled if either pruning or weight sharing are enabled
        # However, if the objectives are implemented incorrectly,
        # It is possible to have optimizatons enabled without any types of optimization (pruning, weight sharing) enabled
        layer_optimizable = model_attributes[layer.name].optimizable and (
            model_attributes[layer.name].optimization_attributes.weight_sharing
            or model_attributes[layer.name].optimization_attributes.pruning
        )
        if isinstance(layer, SUPPORTED_LAYERS) and layer_optimizable:
            value = layer.get_weights()[0]
            structure_type = model_attributes[layer.name].optimization_attributes.structure_type
            layer_savings = objective.layer_savings(model_attributes[layer.name])
            total_resources.append(objective.layer_resources(model_attributes[layer.name]))

            if metric == 'oracle':
                value = np.abs(np.multiply(value, gradients[layer.name]))
                norm = 1
            elif metric == 'saliency':
                value = np.multiply(np.square(value), hessians[layer.name])
                norm = 1
            elif metric == 'l1':
                norm = 1
            else:
                norm = 2

            if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
                if model_attributes[layer.name].optimization_attributes.weight_sharing:
                    logging.warn('Weight sharing not suitable for unstructured pruning. Ignoring....')

                if model_attributes[layer.name].optimization_attributes.pruning:
                    value = np.abs(value)
                    value = value / np.max(value)
                    indices = np.indices(value.shape).reshape(value.ndim, -1).T
                    for i in indices:
                        group = __WeightGroups__(
                            value[tuple(i)], layer_savings, tuple(i), structure_type, layer.name, 'pruning'
                        )
                        groups.append(group)

            if structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                if isinstance(layer, (Dense, QDense)):
                    if model_attributes[layer.name].optimization_attributes.pruning:
                        vals_norm = np.linalg.norm(value, axis=0, ord=norm)
                        vals_norm = vals_norm / np.max(vals_norm)
                    else:
                        vals_norm = np.full((value.shape[1]), sys.float_info.max, dtype=value.dtype)

                    if model_attributes[layer.name].optimization_attributes.weight_sharing:
                        vals_var = np.var(value, axis=0)
                    else:
                        vals_var = np.full((value.shape[1]), sys.float_info.max, dtype=value.dtype)

                    for i in range(vals_norm.shape[0]):
                        if vals_norm[i] <= vals_var[i]:
                            groups.append(
                                __WeightGroups__(
                                    vals_norm[i], layer_savings, i, structure_type, layer.name, optimization_type='pruning'
                                )
                            )
                        else:
                            groups.append(
                                __WeightGroups__(
                                    vals_var[i], layer_savings, i, structure_type, layer.name, optimization_type='sharing'
                                )
                            )

                elif isinstance(layer, (Conv2D, QConv2D)):
                    if model_attributes[layer.name].optimization_attributes.pruning:
                        vals_norm = np.linalg.norm(np.linalg.norm(value, axis=(0, 1), ord='fro'), axis=0, ord=norm)
                        vals_norm = vals_norm / np.max(vals_norm)
                    else:
                        vals_norm = np.full((value.shape[3]), sys.float_info.max, dtype=value.dtype)

                    if model_attributes[layer.name].optimization_attributes.weight_sharing:
                        vals_var = np.var(np.linalg.norm(value, axis=(0, 1), ord='fro'), axis=0)
                    else:
                        vals_var = np.full((value.shape[3]), sys.float_info.max, dtype=value.dtype)

                    for i in range(vals_norm.shape[0]):
                        if vals_norm[i] <= vals_var[i]:
                            groups.append(
                                __WeightGroups__(
                                    vals_norm[i], layer_savings, i, structure_type, layer.name, optimization_type='pruning'
                                )
                            )
                        else:
                            groups.append(
                                __WeightGroups__(
                                    vals_var[i], layer_savings, i, structure_type, layer.name, optimization_type='sharing'
                                )
                            )

            if structure_type == SUPPORTED_STRUCTURES.PATTERN:
                pattern_offset = model_attributes[layer.name].optimization_attributes.pattern_offset
                consecutive_patterns = model_attributes[layer.name].optimization_attributes.consecutive_patterns
                if (np.prod(value.shape)) % pattern_offset != 0:
                    raise Exception('Pattern offset needs to be a factor of matrix size')

                if pattern_offset % consecutive_patterns != 0:
                    raise Exception('Consecutive patterns need to be a factor of matrix size')

                # Transpose, as done in hls4ml Resource strategy
                if isinstance(layer, (Dense, QDense)):
                    value = value.T
                    transposed_shape = value.shape
                elif isinstance(layer, (Conv2D, QConv2D)):
                    value = np.transpose(value, axes=[3, 0, 1, 2])
                    transposed_shape = value.shape

                # Reshape weight matrix into [number_of_patterns, pattern_offset]
                # Note, swapping the axis will mess up the weight order
                # In the case of hls4ml, number_of_patterns is equivalent to reuse factor
                # And, pattern_offset, is the number of multiplications done in parallel
                number_of_patterns = np.prod(transposed_shape) // pattern_offset
                target_shape = (pattern_offset, number_of_patterns)
                reshaped = np.reshape(value, target_shape)

                # Group consecutive patterns (rows) into blocks and reshape
                total_blocks = pattern_offset // consecutive_patterns
                blocks = np.reshape(
                    tf.image.extract_patches(
                        np.expand_dims(np.expand_dims(reshaped, 2), 0),
                        [1, consecutive_patterns, number_of_patterns, 1],
                        [1, consecutive_patterns, number_of_patterns, 1],
                        [1, 1, 1, 1],
                        'SAME',
                    ).numpy(),
                    (total_blocks, -1),
                )

                # If pruning enabled, find cost associated with pruning each neuron
                if model_attributes[layer.name].optimization_attributes.pruning:
                    vals_norm = np.linalg.norm(blocks, axis=1, ord=norm)
                else:
                    vals_norm = np.full((blocks.shape[0],), sys.float_info.max, dtype=value.dtype)

                # If weight sharing enabled, find cost asociated with quantizing nerons to their mean
                if model_attributes[layer.name].optimization_attributes.weight_sharing:
                    vals_var = np.var(blocks, axis=1)
                else:
                    vals_var = np.full((blocks.shape[0],), sys.float_info.max, dtype=value.dtype)

                # Choose min(pruning, weight sharing)
                for i in range(vals_norm.shape[0]):
                    if vals_norm[i] <= vals_var[i]:
                        groups.append(
                            __WeightGroups__(
                                vals_norm[i], layer_savings, i, structure_type, layer.name, optimization_type='pruning'
                            )
                        )
                    else:
                        groups.append(
                            __WeightGroups__(
                                vals_var[i], layer_savings, i, structure_type, layer.name, optimization_type='sharing'
                            )
                        )

            if structure_type == SUPPORTED_STRUCTURES.BLOCK:
                if len(value.shape) != 2:
                    raise Exception('Block pruning is supported for 2-dimensional weight matrices')

                block_shape = model_attributes[layer.name].optimization_attributes.block_shape
                if (value.shape[0] % block_shape[0]) != 0 or (value.shape[1] % block_shape[1] != 0):
                    raise Exception('Block sizes need to be fators of weight matrix dimensions')

                total_blocks = (value.shape[0] * value.shape[1]) // (block_shape[0] * block_shape[1])
                blocks = np.reshape(
                    tf.image.extract_patches(
                        np.expand_dims(np.expand_dims(value, 2), 0),
                        [1, block_shape[0], block_shape[1], 1],
                        [1, block_shape[0], block_shape[1], 1],
                        [1, 1, 1, 1],
                        'SAME',
                    ).numpy(),
                    (total_blocks, block_shape[0] * block_shape[1]),
                )

                if model_attributes[layer.name].optimization_attributes.pruning:
                    vals_norm = np.linalg.norm(blocks, axis=1, ord=norm)
                    vals_norm = vals_norm / np.max(vals_norm)
                else:
                    vals_norm = np.full((blocks.shape[0]), sys.float_info.max, dtype=value.dtype)

                if model_attributes[layer.name].optimization_attributes.weight_sharing:
                    vals_var = np.var(blocks, axis=1)
                else:
                    vals_var = np.full((blocks.shape[0]), sys.float_info.max, dtype=value.dtype)

                for i in range(vals_norm.shape[0]):
                    if vals_norm[i] <= vals_var[i]:
                        groups.append(
                            __WeightGroups__(
                                vals_norm[i], layer_savings, i, structure_type, layer.name, optimization_type='pruning'
                            )
                        )
                    else:
                        groups.append(
                            __WeightGroups__(
                                vals_var[i], layer_savings, i, structure_type, layer.name, optimization_type='sharing'
                            )
                        )

    # The goal is to maximize network accuracy (values) subject to resorces (objective) staying under some threshold
    # This is a Knapsack problem; several implementations are provided in the helper functions
    # The selected values correspond to weight / groups being kept in the network; the rest are pruned / weight shared
    total_resources = np.sum(np.array(total_resources), axis=0)
    target_resources = ((1 - sparsity) * np.array(total_resources)).astype(int)
    _, selected = solve_knapsack(
        np.array([s.value for s in groups]),
        np.array([s.resources for s in groups]).T,
        target_resources,
        implementation=knapsack_solver,
    )
    # Update masks and offsets
    masks = {}
    offsets = {}

    for layer in keras_model.layers:
        if isinstance(layer, SUPPORTED_LAYERS) and model_attributes[layer.name].optimizable:
            structure_type = model_attributes[layer.name].optimization_attributes.structure_type
            selected_layer = [i for i in selected if groups[i].layer_name == layer.name]
            not_selected_layer = [
                i for i in range(len(groups)) if groups[i].layer_name == layer.name and i not in selected_layer
            ]

            if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
                mask = np.zeros(model_attributes[layer.name].weight_shape, layer.get_weights()[0].dtype)
                for i in selected_layer:
                    mask[groups[i].layer_position] = 1
                masks[layer.name] = mask
                offsets[layer.name] = np.zeros(model_attributes[layer.name].weight_shape, layer.get_weights()[0].dtype)

            if structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                mask = np.zeros(model_attributes[layer.name].weight_shape, layer.get_weights()[0].dtype)
                offset = np.zeros(model_attributes[layer.name].weight_shape, layer.get_weights()[0].dtype)
                if isinstance(layer, (Dense, QDense)):
                    for i in selected_layer:
                        mask[:, groups[i].layer_position] = 1
                    for i in not_selected_layer:
                        if groups[i].optimization_type == 'sharing':
                            offset[:, groups[i].layer_position] = np.mean(layer.get_weights()[0][:, i])
                if isinstance(layer, (Conv2D, QConv2D)):
                    for i in selected_layer:
                        mask[:, :, :, groups[i].layer_position] = 1
                    for i in not_selected_layer:
                        if groups[i].optimization_type == 'sharing':
                            offset[
                                :,
                                :,
                                :,
                                groups[i].layer_position,
                            ] = np.mean(layer.get_weights()[0][:, :, :, i])
                masks[layer.name] = mask
                offsets[layer.name] = offset

            if structure_type == SUPPORTED_STRUCTURES.PATTERN:
                pattern_offset = model_attributes[layer.name].optimization_attributes.pattern_offset
                consecutive_patterns = model_attributes[layer.name].optimization_attributes.consecutive_patterns
                number_of_patterns = np.prod(model_attributes[layer.name].weight_shape) // pattern_offset

                # Transpose shape, as done in hls4ml Resource strategy
                # We need the weights to recalculate the block means
                weight_shape = model_attributes[layer.name].weight_shape
                if isinstance(layer, (Dense, QDense)):
                    value = layer.get_weights()[0].T
                    transposed_shape = (weight_shape[1], weight_shape[0])
                elif isinstance(layer, (Conv2D, QConv2D)):
                    value = np.transpose(layer.get_weights()[0], (3, 0, 1, 2))
                    transposed_shape = (weight_shape[3], weight_shape[0], weight_shape[1], weight_shape[2])

                # Decode masks
                mask = np.zeros((np.prod(transposed_shape),), layer.get_weights()[0].dtype)
                for i in selected_layer:
                    pos = groups[i].layer_position * number_of_patterns * consecutive_patterns
                    for j in range(pos, pos + number_of_patterns * consecutive_patterns, number_of_patterns):
                        mask[range(j, j + number_of_patterns)] = 1

                        # Decode offsets
                offset = np.zeros((np.prod(transposed_shape),), layer.get_weights()[0].dtype)

                # Re-calculate the blocks, they are needed to calculate block means
                target_shape = (pattern_offset, number_of_patterns)
                reshaped = np.reshape(value, target_shape)

                total_blocks = pattern_offset // consecutive_patterns
                blocks = np.reshape(
                    tf.image.extract_patches(
                        np.expand_dims(np.expand_dims(reshaped, 2), 0),
                        [1, consecutive_patterns, number_of_patterns, 1],
                        [1, consecutive_patterns, number_of_patterns, 1],
                        [1, 1, 1, 1],
                        'SAME',
                    ).numpy(),
                    (total_blocks, -1),
                )

                for i in not_selected_layer:
                    if groups[i].optimization_type == 'sharing':
                        mean = np.mean(blocks[groups[i].layer_position, :])
                        pos = groups[i].layer_position * number_of_patterns * consecutive_patterns
                        for j in range(pos, pos + number_of_patterns * consecutive_patterns, number_of_patterns):
                            offset[range(j, j + number_of_patterns)] = mean

                            # Reshape into original shape and store result
                if isinstance(layer, (Dense, QDense)):
                    mask = np.reshape(mask, transposed_shape).T
                    offset = np.reshape(offset, transposed_shape).T
                elif isinstance(layer, (Conv2D, QConv2D)):
                    mask = np.transpose(np.reshape(mask, transposed_shape), (1, 2, 3, 0))
                    offset = np.transpose(np.reshape(offset, transposed_shape), (1, 2, 3, 0))
                masks[layer.name] = mask
                offsets[layer.name] = offset

            if structure_type == SUPPORTED_STRUCTURES.BLOCK:
                block_shape = model_attributes[layer.name].optimization_attributes.block_shape
                total_blocks = np.prod(model_attributes[layer.name].weight_shape) // np.prod(block_shape)
                blocks_in_row = model_attributes[layer.name].weight_shape[1] // block_shape[1]
                blocks = np.reshape(
                    tf.image.extract_patches(
                        np.expand_dims(np.expand_dims(layer.get_weights()[0], 2), 0),
                        [1, block_shape[0], block_shape[1], 1],
                        [1, block_shape[0], block_shape[1], 1],
                        [1, 1, 1, 1],
                        'SAME',
                    ).numpy(),
                    (total_blocks, block_shape[0] * block_shape[1]),
                )

                mask = np.zeros(model_attributes[layer.name].weight_shape, layer.get_weights()[0].dtype)
                for i in selected_layer:
                    row = block_shape[0] * (groups[i].layer_position // blocks_in_row)
                    col = block_shape[1] * (groups[i].layer_position % blocks_in_row)
                    cols = np.linspace(col, col + block_shape[1], block_shape[1], endpoint=False, dtype=np.int32)
                    rows = np.linspace(row, row + block_shape[0], block_shape[0], endpoint=False, dtype=np.int32)
                    zeros = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)
                    mask[zeros[:, 0], zeros[:, 1]] = 1

                offset = np.zeros(model_attributes[layer.name].weight_shape, layer.get_weights()[0].dtype)
                for i in not_selected_layer:
                    if groups[i].optimization_type == 'sharing':
                        mean = np.mean(blocks[groups[i].layer_position, :])
                        row = block_shape[0] * (groups[i].layer_position // blocks_in_row)
                        col = block_shape[1] * (groups[i].layer_position % blocks_in_row)
                        cols = np.linspace(col, col + block_shape[1], block_shape[1], endpoint=False, dtype=np.int32)
                        rows = np.linspace(row, row + block_shape[0], block_shape[0], endpoint=False, dtype=np.int32)
                        pos = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)
                        offset[pos[:, 0], pos[:, 1]] = mean

                masks[layer.name] = mask
                offsets[layer.name] = offset

    return masks, offsets


class __WeightGroups__:
    '''
    A helper class containing information about a group of weights
    '''

    def __init__(self, value, resources, layer_position, structure_type=None, layer_name=None, optimization_type=None):
        self.value = value
        self.resources = resources
        self.layer_position = layer_position
        self.structure_type = structure_type
        self.layer_name = layer_name
        self.optimization_type = optimization_type
