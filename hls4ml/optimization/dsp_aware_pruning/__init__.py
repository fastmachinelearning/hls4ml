import numpy as np

from hls4ml.optimization.dsp_aware_pruning.attributes import get_attributes_from_keras_model_and_hls4ml_config
from hls4ml.optimization.dsp_aware_pruning.keras import optimize_model

default_regularization_range = np.logspace(-6, -2, num=16).tolist()


def optimize_keras_model_for_hls4ml(
    keras_model,
    hls_config,
    objective,
    scheduler,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    epochs,
    optimizer,
    loss_fn,
    validation_metric,
    increasing,
    rtol,
    callbacks=None,
    ranking_metric='l1',
    local=False,
    verbose=False,
    rewinding_epochs=1,
    cutoff_bad_trials=3,
    directory='hls4ml-optimization',
    tuner='Bayesian',
    knapsack_solver='CBC_MIP',
    regularization_range=default_regularization_range,
):
    '''
    Top-level function for optimizing a Keras model, given hls4ml config and a hardware objective(s)

    Args:
        keras_model (keras.Model): Model to be optimized
        hls_config (dict): hls4ml configuration, obtained from hls4ml.utils.config.config_from_keras_model(...)
        objective (hls4ml.optimization.objectives.ObjectiveEstimator):
        Parameter, hardware or user-defined objective of optimization
        scheduler (hls4ml.optimization.scheduler.OptimizationScheduler):
        Sparsity scheduler, choose between constant, polynomial and binary
        X_train (np.array): Training inputs
        y_train (np.array): Training labels
        X_val (np.array): Validation inputs
        y_val (np.array): Validation labels
        batch_size (int): Batch size during training
        epochs (int): Maximum number of epochs to fine-tune model, in one iteration of pruning
        optimizer (keras.optimizers.Optimizer or equivalent-string description): Optimizer used during training
        loss_fn (keras.losses.Loss or equivalent loss description): Loss function used during training
        validation_metric (keras.metrics.Metric or equivalent loss description): Validation metric, used as a baseline
        increasing (boolean): If the metric improves with increased values;
            e.g. accuracy -> increasing = True, MSE -> increasing = False
        rtol (float): Relative tolerance;
            pruning stops when pruned_validation_metric < (or >) rtol * baseline_validation_metric
        callbacks (list of keras.callbacks.Callback) Currently not supported, developed in future versions
        ranking_metric (string): Metric used for ranking weights and structures;
            currently supported l1, l2, saliency and Oracle
        local (boolean): Layer-wise or global pruning
        verbose (boolean): Display debug logs during model optimization
        rewinding_epochs (int): Number of epochs to retrain model without weight freezing,
            allows regrowth of previously pruned weights
        cutoff_bad_trials (int): After how many bad trials (performance below threshold),
            should model pruning / weight sharing stop
        directory (string): Directory to store temporary results
        tuner (str): Tuning algorithm, choose between Bayesian, Hyperband and None
        knapsack_solver (str): Algorithm to solve Knapsack problem when optimizing;
            default usually works well; for very large networks, greedy algorithm might be more suitable
        regularization_range (list): List of suitable hyperparameters for weight decay

    Returns:
        keras.Model: Optimized model
    '''

    # Extract model attributes
    model_attributes = get_attributes_from_keras_model_and_hls4ml_config(keras_model, hls_config)

    # Optimize model
    return optimize_model(
        keras_model,
        model_attributes,
        objective,
        scheduler,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size,
        epochs,
        optimizer,
        loss_fn,
        validation_metric,
        increasing,
        rtol,
        callbacks=callbacks,
        ranking_metric=ranking_metric,
        local=local,
        verbose=verbose,
        rewinding_epochs=rewinding_epochs,
        cutoff_bad_trials=cutoff_bad_trials,
        directory=directory,
        tuner=tuner,
        knapsack_solver=knapsack_solver,
        regularization_range=regularization_range,
    )
