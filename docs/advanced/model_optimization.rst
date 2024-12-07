=================================
Hardware-aware Optimization API
=================================

Pruning and weight sharing are effective techniques to reduce model footprint and computational requirements. The hls4ml Optimization API introduces hardware-aware pruning and weight sharing.
By defining custom objectives, the algorithm solves a Knapsack optimization problem aimed at maximizing model performance, while keeping the target resource(s) at a minimum. Out-of-the box objectives include network sparsity, GPU FLOPs, Vivado DSPs, memory utilization etc.

The code block below showcases three use cases of the hls4ml Optimization API - network sparsity (unstructured pruning), GPU FLOPs (structured pruning) and Vivado DSP utilization (pattern pruning). First, we start with unstructured pruning:

.. code-block:: Python

    from sklearn.metrics import accuracy_score
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import CategoricalAccuracy
    from tensorflow.keras.losses import CategoricalCrossentropy
    from hls4ml.optimization.dsp_aware_pruning.keras import optimize_model
    from hls4ml.optimization.dsp_aware_pruning.keras.utils import get_model_sparsity
    from hls4ml.optimization.dsp_aware_pruning.attributes import get_attributes_from_keras_model
    from hls4ml.optimization.dsp_aware_pruning.objectives import ParameterEstimator
    from hls4ml.optimization.dsp_aware_pruning.scheduler import PolynomialScheduler
    # Define baseline model and load data
    # X_train, y_train = ...
    # X_val, y_val = ...
    # X_test, y_test = ...
    # baseline_model = ...
    # Evaluate baseline model
    y_baseline = baseline_model.predict(X_test)
    acc_base = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_baseline, axis=1))
    sparsity, layers = get_model_sparsity(baseline_model)
    print(f'Baseline Keras accuracy: {acc_base}')
    print(f'Baseline Keras sparsity, overall: {sparsity}')
    print(f'Baseline Keras sparsity, per-layer: {layers}')
    # Defining training parameters
    # Epochs refers to the number of maximum epochs to train a model, after imposing some sparsity
    # If the model is pre-trained, a good rule of thumb is to use between a 1/3 and 1/2 of the number of epochs used to train baseline model
    epochs = 10
    batch_size = 128
    metric = 'accuracy'
    optimizer = Adam()
    loss_fn = CategoricalCrossentropy(from_logits=True)

    # Define the metric to monitor, as well as if its increasing or decreasing
    # This disctinction allows us to optimize both regression and classification models
    # In regression, e.g. minimize validation MSE & for classification e.g. maximize accuracy
    metric, increasing = CategoricalAccuracy(), True
    # Relative tolerance (rtol) is the the relative loss in metric the optimized model is allowed to incur
    rtol = 0.975

    # A scheduler defines how the sparsity is incremented at each step
    # In this case, the maximum sparsity is 50% and it will be applied at a polynomially decreasing rate, for 10 steps
    # If the final sparsity is unspecified, it is set to 100%
    # The optimization algorithm stops either when (i) the relative drop in performance is below threshold or (ii) final sparsity reached
    scheduler = PolynomialScheduler(5, final_sparsity=0.5)
    # Get model attributes
    model_attributes = get_attributes_from_keras_model(baseline_model)

    # Optimize model
    # ParameterEstimator is the objective and, in this case, the objective is to minimize the total number of parameters
    optimized_model = optimize_model(
        baseline_model, model_attributes, ParameterEstimator, scheduler,
        X_train, y_train, X_val, y_val, batch_size, epochs, optimizer, loss_fn, metric, increasing, rtol
    )
    # Evaluate optimized model
    y_optimized = optimized_model.predict(X_test)
    acc_optimized = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_optimized, axis=1))
    sparsity, layers = get_model_sparsity(optimized_model)
    print(f'Optimized Keras accuracy: {acc_optimized}')
    print(f'Optimized Keras sparsity, overall: {sparsity}')
    print(f'Opimized Keras sparsity, per-layer: {layers}')

In a similar manner, it is possible to target GPU FLOPs or Vivado DSPs. However, in that case, sparsity is not equivalent to model sparsity.
Instead, it is the sparsity of the target resource. As an example: Starting with a network utilizing 512 DSPs and a final sparsity of 50%; the optimized network will use 256 DSPs.

To optimize GPU FLOPs, the code is similar to above:

.. code-block:: Python

    from hls4ml.optimization.dsp_aware_pruning.objectives.gpu_objectives import GPUFLOPEstimator

    # Optimize model
    # Note the change from ParameterEstimator to GPUFLOPEstimator
    optimized_model = optimize_model(
        baseline_model, model_attributes, GPUFLOPEstimator, scheduler,
        X_train, y_train, X_val, y_val, batch_size, epochs, optimizer, loss_fn, metric, increasing, rtol
    )

    # Evaluate optimized model
    y_optimized = optimized_model.predict(X_test)
    acc_optimized = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_optimized, axis=1))
    print(f'Optimized Keras accuracy: {acc_optimized}')
    # Note the difference in total number of parameters
    # Optimizing GPU FLOPs is equivalent to removing entire structures (filters, neurons) from the network
    print(baseline_model.summary())
    print(optimized_model.summary())

Finally, optimizing Vivado DSPs is possible, given a hls4ml config:

.. code-block:: Python

    from hls4ml.utils.config import config_from_keras_model
    from hls4ml.optimization.dsp_aware_pruning.objectives.vivado_objectives import VivadoDSPEstimator

    # Note the change from optimize_model to optimize_keras_model_for_hls4ml
    # The function optimize_keras_model_for_hls4ml acts as a wrapper for the function, parsing hls4ml config to model attributes
    from hls4ml.optimization import optimize_keras_model_for_hls4ml

    # Create hls4ml config
    default_reuse_factor = 4
    default_precision = 'ac_fixed<16, 6>'
    hls_config = config_from_keras_model(baseline_model, granularity='name', default_precision=default_precision, default_reuse_factor=default_reuse_factor)
    hls_config['IOType'] = 'io_parallel'
     hls_config['Model']['Strategy'] = 'Resource'   # Strategy must be present for optimisation

    # Optimize model
    # Note the change from ParameterEstimator to VivadoDSPEstimator
    optimized_model = optimize_keras_model_for_hls4ml(
        baseline_model, model_attributes, VivadoDSPEstimator, scheduler,
        X_train, y_train, X_val, y_val, batch_size, epochs,
        optimizer, loss_fn, metric, increasing, rtol
    )

    # Evaluate optimized model
    y_optimized = optimized_model.predict(X_test)
    acc_optimized = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_optimized, axis=1))
    print(f'Optimized Keras accuracy: {acc_optimized}')

There are two more Vivado "optimizers" - VivadoFFEstimator, aimed at reducing register utilization and VivadoMultiObjectiveEstimator, aimed at optimizing BRAM and DSP utilization.
Note, to ensure DSPs are optimized, "unrolled" Dense multiplication must be used before synthesizing HLS, by modifying the config:

.. code-block:: Python

    hls_config = config_from_keras_model(optimized_model)
    hls_config['Model']['Strategy'] = 'Unrolled'
    # Any addition hls4ml config, reuse factor etc...
