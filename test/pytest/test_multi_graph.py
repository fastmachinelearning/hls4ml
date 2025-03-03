from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense
import hls4ml

test_root_path = Path(__file__).parent

def create_test_model():
    """
    This architecture ensures testing of corner cases such as:
    double layer outputs and variety of layers to serve as spliting points.
    """
    inp = Input(shape=(4, 4, 3), name='input_layer')
    x = Conv2D(4, (3, 3), padding='same', name='conv1')(inp)
    x = Activation('relu', name='relu1')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(16, activation='relu', name='dense_common')(x)
    output1 = Dense(5, activation='relu', name='dense1')(x)
    output2 = Dense(5, activation='relu', name='dense2')(x)
    model = tf.keras.Model(inputs=inp, outputs=[output1, output2])
    
    return model

@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['model', 'name'])
@pytest.mark.parametrize('split_layers', [
    ('pool1', 'dense_common'),
    ('relu1', 'flatten')
])
def test_multimodelgraph_predict(split_layers, io_type, strategy, granularity):
    """
    Tests the multi-graph splitting and stitching process.
    - Verifies that predictions from the monolithic and multi-graph versions match with the CSimulation.
    - When granularity='name', an additional HLS build and stitched RTL simulation step is performed.
    - The RTL simulation outputs are compared against the predicted values from CSimulation.
    """
    backend = 'vitis'
    model = create_test_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    X_input = np.random.rand(5, 4, 4, 3).astype(np.float32)
    keras_pred = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity, default_precision='ap_fixed<32,16>')
    config['Model']['Strategy'] = strategy

    output_dir_mono = str(test_root_path / f"hls4mlprj_mono_{granularity}_{'_'.join(split_layers)}_{io_type}_{strategy}")
    output_dir_multi = str(test_root_path / f"hls4mlprj_multi_{granularity}_{'_'.join(split_layers)}_{io_type}_{strategy}")

    # --- Monolithic HLS conversion (no split) ---
    hls_model_mono = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir_mono,
        backend=backend,
        io_type=io_type
    )
    hls_model_mono.compile()
    pred_mono = hls_model_mono.predict(X_input)

    # --- Multi-model conversion with split ---
    hls_model_multi = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir_multi,
        backend=backend,
        io_type=io_type,
        split_layer_names=list(split_layers)
    )
    hls_model_multi.compile()
    pred_multi = hls_model_multi.predict(X_input)

    assert hasattr(hls_model_multi, 'graphs'), "Multi-model graph missing 'graphs' attribute."
    assert len(hls_model_multi.graphs) == 3, f"Expected 3 subgraphs, got {len(hls_model_multi.graphs)}"

    for mono_out, multi_out in zip(pred_mono, pred_multi):
        np.testing.assert_allclose(multi_out, mono_out, rtol=0, atol=1e-5)
    
    if granularity == 'name':
        if io_type == 'io_parallel' and split_layers == ('relu1', 'flatten'):
            pytest.skip("Skipping RTL simulation for io_parallel with split layer at flatten due to improper simulation behavior.")

        # --- Optional: Build the HLS project and run simulation ---
        hls_model_multi.build(csim=False, cosim=False, vsynth=False, export=True, 
                        stitch_design=True, sim_stitched_design=True, export_stitched_design=True)

        # test only the first sample, as batch prediction is not supported for stitched RTL simulations
        inp = np.expand_dims(X_input[0], axis=0)
        sim_results = hls_model_multi.predict(inp, sim = 'rtl')
        for sim_out, pred_out in zip(sim_results, list([pred_multi[0][0], pred_multi[1][0]])):
            np.testing.assert_allclose(sim_out, pred_out, rtol=0, atol=0.3)
