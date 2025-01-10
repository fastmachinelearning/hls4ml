import re

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from qkeras import QConv2D, QDense
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense

from hls4ml.optimization.dsp_aware_pruning.keras.config import SUPPORTED_LAYERS, TMP_DIRECTORY
from hls4ml.optimization.dsp_aware_pruning.keras.regularizers import Conv2DRegularizer, DenseRegularizer

co = {}
_add_supported_quantized_objects(co)


class HyperOptimizationModel(kt.HyperModel):
    '''
    Helper class for Keras Tuner
    '''

    def __init__(self, model, attributes, optimizer, loss_fn, validation_metric, regularization_range):
        """Create new instance of HyperOptimizationModel

        Args:
            model (keras.Model): Baseline model
            attributes (dict): Layer-wise dictionary of attributes
            optimizer (keras.optimizers.Optimizer or equivalent string description): Model optimizer
            loss_fn (keras.losses.Loss or equivalent string description): Model loss function
            validation_metric (keras.metrics.Metric or equivalent string description): Model validation metric
            regularization_range (list): List of suitable hyperparameters for weight decay
        """
        self.model = model
        self.attributes = attributes
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.validation_metric = validation_metric
        self.regularization_range = regularization_range

    def build(self, hp):
        model_to_prune = tf.keras.models.clone_model(self.model)
        default_regularizaton = self.regularization_range[len(self.regularization_range) // 2]

        # Make regularization loss a tunable hyperparameter
        for layer in model_to_prune.layers:
            if isinstance(layer, SUPPORTED_LAYERS) and self.attributes[layer.name].optimizable:
                structure_type = self.attributes[layer.name].optimization_attributes.structure_type
                block_shape = self.attributes[layer.name].optimization_attributes.block_shape
                pattern_offset = self.attributes[layer.name].optimization_attributes.pattern_offset
                consecutive_patterns = self.attributes[layer.name].optimization_attributes.consecutive_patterns

                pruning = self.attributes[layer.name].optimization_attributes.pruning
                weight_sharing = self.attributes[layer.name].optimization_attributes.weight_sharing

                alpha = (
                    hp.Choice(f'{layer.name}_alpha', values=self.regularization_range, default=default_regularizaton)
                    if pruning
                    else 0
                )
                beta = (
                    hp.Choice(f'{layer.name}_beta', values=self.regularization_range, default=default_regularizaton)
                    if weight_sharing
                    else 0
                )

                if isinstance(layer, (Dense, QDense)) and self.attributes[layer.name].optimizable:
                    layer.kernel_regularizer = DenseRegularizer(
                        alpha,
                        beta,
                        norm=1,
                        structure_type=structure_type,
                        block_shape=block_shape,
                        pattern_offset=pattern_offset,
                        consecutive_patterns=consecutive_patterns,
                    )
                elif isinstance(layer, (Conv2D, QConv2D)) and self.attributes[layer.name].optimizable:
                    layer.kernel_regularizer = Conv2DRegularizer(
                        alpha,
                        beta,
                        norm=1,
                        structure_type=structure_type,
                        pattern_offset=pattern_offset,
                        consecutive_patterns=consecutive_patterns,
                    )

        # Rebuild model graph
        model_to_prune = tf.keras.models.model_from_json(model_to_prune.to_json(), custom_objects=co)
        model_to_prune.set_weights(self.model.get_weights())
        model_to_prune.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.validation_metric])

        return model_to_prune


default_regularization_range = np.logspace(-6, -2, num=16).tolist()


def build_optimizable_model(
    model,
    attributes,
    optimizer,
    loss_fn,
    validation_metric,
    increasing,
    train_dataset,
    validation_dataset,
    batch_size,
    epochs,
    verbose=False,
    directory=TMP_DIRECTORY,
    tuner='Bayesian',
    regularization_range=default_regularization_range,
):
    '''
    Function identifying optimizable layers and adding a regularization loss

    Notes:
    - In general, the regularization and learning rate ranges do not need to be provided,
    as the implementation sets a generic enough range. if the user has an idea on the
    possible range on hyperparameter ranges, the tuning will complete faster.
    - The default tuner is Bayesian & when coupled with the correct ranges of hyperparameters,
    it performs quite well, fast. However, older version of Keras Tuner had a crashing bug with it.
    - In general, the directory does not need to be specified. However, if pruning several models simultaneously,
    to avoid conflicting intermediate results, it is useful to specify directory.

    Args:
        model (keras.Model): Model to be optimized
        attributes (dict): Layer-wise model attributes, obtained from hls4ml.optimization.get_attributes_from_keras_model()
        optimizer (keras.optimizers.Optimizer): Optimizer used during training
        loss_fn (keras.losses.Loss): Loss function used during training
        validation_metric (keras.metrics.Metric): Validation metric, used as a baseline
        train_dataset (tf.Dataset): Training inputs and labels, in the form of an iterable TF Dataset
        validation_dataset (tf.Dataset): Validation inputs and labels, in the form of an iterable TF Dataset
        batch_size (int): Batch size during training
        epochs (int): Maximum number of epochs to fine-tune model, in one iteration of pruning
        verbose (bool): Whether to log tuner outputs to the console
        directory (string): Directory to store tuning results
        tuner (str): Tuning algorithm, choose between Bayesian and Hyperband
        regularization_range (list): List of suitable hyperparameters for weight decay
        learning_rate_range (list): List of suitable hyperparameters for learning rate

    Returns:
        keras.Model: Model prepared for optimization
    '''
    # User provided manual hyper-parameters for regularisation loss
    # TODO - Maybe we could extend this to be hyper-parameters per layer? or layer-type?
    # Currently, the same (manually-set) hyper-parameter is set for every layer
    if tuner == 'Manual':
        model_to_prune = tf.keras.models.clone_model(model)
        for layer in model_to_prune.layers:
            if isinstance(layer, SUPPORTED_LAYERS) and attributes[layer.name].optimizable:
                structure_type = attributes[layer.name].optimization_attributes.structure_type
                block_shape = attributes[layer.name].optimization_attributes.block_shape
                pattern_offset = attributes[layer.name].optimization_attributes.pattern_offset
                consecutive_patterns = attributes[layer.name].optimization_attributes.consecutive_patterns

                pruning = attributes[layer.name].optimization_attributes.pruning
                weight_sharing = attributes[layer.name].optimization_attributes.weight_sharing

                alpha = regularization_range[0] if pruning else 0
                beta = regularization_range[0] if weight_sharing else 0

                if isinstance(layer, (Dense, QDense)) and attributes[layer.name].optimizable:
                    layer.kernel_regularizer = DenseRegularizer(
                        alpha,
                        beta,
                        norm=1,
                        structure_type=structure_type,
                        block_shape=block_shape,
                        pattern_offset=pattern_offset,
                        consecutive_patterns=consecutive_patterns,
                    )
                elif isinstance(layer, (Conv2D, QConv2D)) and attributes[layer.name].optimizable:
                    layer.kernel_regularizer = Conv2DRegularizer(
                        alpha,
                        beta,
                        norm=1,
                        structure_type=structure_type,
                        pattern_offset=pattern_offset,
                        consecutive_patterns=consecutive_patterns,
                    )

        # Rebuild model graph
        model_to_prune = tf.keras.models.model_from_json(model_to_prune.to_json(), custom_objects=co)
        model_to_prune.set_weights(model.get_weights())
        model_to_prune.compile(optimizer=optimizer, loss=loss_fn, metrics=[validation_metric])

        return model_to_prune

    # User opted for hyper-parameter tuning
    else:
        objective_direction = 'max' if increasing else 'min'
        if isinstance(validation_metric, str):
            objective_name = validation_metric
        else:
            objective_name = re.sub(r'(?<!^)(?=[A-Z])', '_', validation_metric.__class__.__name__).lower()
        if tuner == 'Bayesian':
            tuner = kt.BayesianOptimization(
                hypermodel=HyperOptimizationModel(
                    model, attributes, optimizer, loss_fn, validation_metric, regularization_range
                ),
                objective=kt.Objective(objective_name, objective_direction),
                max_trials=10,
                overwrite=True,
                directory=directory + '/tuning',
            )
        elif tuner == 'Hyperband':
            tuner = kt.Hyperband(
                hypermodel=HyperOptimizationModel(
                    model, attributes, optimizer, loss_fn, validation_metric, regularization_range
                ),
                objective=kt.Objective(objective_name, objective_direction),
                max_epochs=epochs,
                factor=3,
                hyperband_iterations=1,
                overwrite=True,
                directory=directory + '/tuning',
            )
        else:
            raise Exception('Unknown tuner; possible options are Bayesian and Hyperband')

        if verbose:
            tuner.search_space_summary()

        tuner.search(
            train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_dataset,
            callbacks=[EarlyStopping(monitor='val_loss', patience=1)],
        )

        if verbose:
            tuner.results_summary()

        return tuner.get_best_models(num_models=1)[0]


def remove_custom_regularizers(model):
    '''
    Helper function to remove custom regularizers (DenseRegularizer & Conv2DRegularizer)
    This makes it possible to load the model in a different environment without hls4ml installed

    Args:
        model (keras.Model): Baseline model

    Returns:
        keras.Model: Model without custom regularizers
    '''
    weights = model.get_weights()
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            if isinstance(layer.kernel_regularizer, (DenseRegularizer, Conv2DRegularizer)):
                layer.kernel_regularizer = None

    model = tf.keras.models.model_from_json(model.to_json(), custom_objects=co)
    model.set_weights(weights)
    return model
