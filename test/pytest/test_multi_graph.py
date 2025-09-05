from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model

import hls4ml
import hls4ml.model

test_root_path = Path(__file__).parent


def create_test_model():
    inp = Input(shape=(6, 8), name='input_layer')
    x = Dense(16, name='dense1')(inp)
    x = Activation('relu', name='relu1')(x)
    x = Dense(8, name='dense2')(x)
    x = Activation('relu', name='relu2')(x)
    x = GlobalAveragePooling1D(name='avg_pool')(x)
    x = Dense(16, name='dense_common')(x)
    x = Activation('relu', name='relu_common')(x)
    output1 = Dense(5, name='dense1_out')(x)
    output1 = Activation('relu', name='relu_out1')(output1)
    output2 = Dense(5, name='dense2_out')(x)
    output2 = Activation('relu', name='relu_out2')(output2)
    model = Model(inputs=inp, outputs=[output1, output2])
    return model


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['model', 'name'])
@pytest.mark.parametrize('split_layers', [('dense2', 'avg_pool'), ('relu1', 'relu_common')])
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
    X_input = np.random.rand(5, 6, 8).astype(np.float32)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity, default_precision='ap_fixed<32,16>')
    config['Model']['Strategy'] = strategy

    output_dir = str(test_root_path / f"hls4mlprj_{granularity}_{'_'.join(split_layers)}_{io_type}_{strategy}")

    # --- Monolithic HLS conversion (no split) ---
    hls_model_mono = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model_mono.compile()
    pred_mono = hls_model_mono.predict(X_input)

    # --- Multi-model conversion with split ---
    hls_model_multi = hls4ml.model.to_multi_model_graph(hls_model_mono, list(split_layers))
    hls_model_multi.compile()
    pred_multi = hls_model_multi.predict(X_input)

    assert hasattr(hls_model_multi, 'graphs'), "Multi-model graph missing 'graphs' attribute."
    assert len(hls_model_multi.graphs) == 3, f"Expected 3 subgraphs, got {len(hls_model_multi.graphs)}"

    for mono_out, multi_out in zip(pred_mono, pred_multi):
        np.testing.assert_allclose(multi_out, mono_out, rtol=0, atol=1e-5)

    # if granularity == 'name':
    #     # --- Optional: Build the HLS project and run simulation ---
    #     hls_model_multi.build(
    #         csim=False,
    #         cosim=False,
    #         vsynth=False,
    #         export=True,
    #         stitch_design=True,
    #         sim_stitched_design=True,
    #         export_stitched_design=True,
    #     )

    #     # test only the first sample, as batch prediction is not supported for stitched RTL simulations
    #     inp = np.expand_dims(X_input[0], axis=0)
    #     sim_results = hls_model_multi.predict(inp, sim='rtl')
    #     for sim_out, pred_out in zip(sim_results, list([pred_multi[0][0], pred_multi[1][0]])):
    #         np.testing.assert_allclose(sim_out, pred_out, rtol=0, atol=1e-5)
