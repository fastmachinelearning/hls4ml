==================================
Loading weights from external BRAM
==================================

.. note::
    This feature is being evaluated for re-implementation. We welcome feedback from users how to make the implementation more flexible.

``hls4ml`` can optionally store weights in BRAMs external to the design. This is supported in Vivado/Vitis and Catapult backends. It is the responsibility of the user to ensure the weights are properly loaded during the operation of the design.

The feature works as a threshold, exposed through a ``BramFactor`` config parameter. Layers with more weights above the threshold will be exposed as BRAM interface. Consider the following code:

.. code-block:: Python

    model = tf.keras.models.Sequential()
    model.add(Dense(10, activation="relu", input_shape=(12,), name="dense_1"))
    model.add(Dense(20, activation="relu", name="dense_2"))
    model.add(Dense(5, activation="softmax", name="dense_3"))
    model.compile(optimizer='adam', loss='mse')

    config = hls4ml.utils.config_from_keras_model(model)
    config["Model"]["Strategy"] = "Resource"
    config["Model"]["BramFactor"] = 100

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )

Having set ``BramFactor=100``, only layers with more than 100 weights will be exposed as external BRAM, in this case layers ``dense_1`` and ``dense_2``. ``BramFactor`` can currently be only set at the model level. The generated code will now have weights as part of the interface.

.. code-block:: C++

    void myproject(
        hls::stream<input_t> &dense_1_input,
        hls::stream<result_t> &layer7_out,
        model_default_t w2[120],
        model_default_t w4[200]
    ) {
        #pragma HLS INTERFACE axis port=dense_1_input,layer7_out
        #pragma HLS INTERFACE bram port=w2,w4
        ...

When integrating the design, users can use the exposed interface to implement weight reloading scheme.
