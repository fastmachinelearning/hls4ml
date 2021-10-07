import pytest
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
import torch
import onnx
import onnxruntime
import numpy as np
import hls4ml
import os

'''
Test models of single layers. 

Random input data is generated and the output of the hls4ml converted model is compared
against the expected output from the original model. Tests fail when the observed output
is too different from the expected.

The idea is to test that the conversion of layers is working correctly, only by their external
behaviour - i.e. does the output match the expectation. This is not strictly a test of the 
backend behaviour, so default values are used but with a wide precision by default.

Handling of layer attributes like filter size, stride, padding, use_bias is tested implicitly by
their impact on the output shape and values.
'''

def single_layer_model_factory(layer):
    '''Create a model (Keras, PyTorch or ONNX) and HLSModel according to the test spec.'''
    assert isinstance(layer, LayerTestWrapper)
    # Keras
    if isinstance(layer.layer, tf.keras.layers.Layer):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(layer.input_shape))
        model.add(layer.layer)
        config = hls4ml.utils.config_from_keras_model(model, default_precision=layer.default_precision)
        hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config,
                                                               output_dir=layer.output_dir)
        hls_model.compile()
        return model, hls_model
    # Pytorch
    elif isinstance(layer.layer, torch.nn.Module):
        model = torch.nn.Sequential(layer.layer).to(memory_format=torch.channels_last)
        config = hls4ml.utils.config_from_pytorch_model(model, default_precision='ap_fixed<32,16>')
        hls_model = hls4ml.converters.convert_from_pytorch_model(model, hls_config=config,
                                                                 output_dir=layer.output_dir, input_shape=layer.input_shape)
        hls_model.compile()
        return model, hls_model
    # ONNX
    elif isinstance(layer.layer, onnx.onnx_ml_pb2.NodeProto):
        model = onnx_model_makers[layer.layer.op_type](layer)
        meta = model.metadata_props.add()
        meta.key = 'output_dir'
        meta.value = layer.output_dir
        onnx.checker.check_model(model)
        os.makedirs(layer.output_dir, exist_ok=True)
        onnx.save(model, f'{layer.output_dir}/model.onnx')
        config = hls4ml.utils.config_from_onnx_model(model, default_precision=layer.default_precision)
        hls_model = hls4ml.converters.convert_from_onnx_model(model, hls_config=config,
                                                              output_dir=layer.output_dir)
        hls_model.compile()
        return model, hls_model
    else:
        return None, None

def onnx_act_model(layer):
    inp = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.DOUBLE, [*layer.data_shape])
    out = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.DOUBLE, [None for i in range(len(layer.data_shape))])
    model = onnx.helper.make_model(onnx.helper.make_graph([layer.layer], layer.output_dir, [inp], [out]))
    return model

def onnx_gemm_model(layer):
    assert isinstance(layer, LayerTestWrapper)
    wshape = (*layer.input_shape, *layer.output_shape)
    w = onnx.helper.make_tensor('b', onnx.TensorProto.DOUBLE, wshape, rand_neg1topos1(*wshape).flatten())
    b = onnx.helper.make_tensor('c', onnx.TensorProto.DOUBLE, layer.output_shape, rand_neg1topos1(*layer.output_shape).flatten()) 
    #node = onnx.helper.make_node('Gemm', inputs=['x', 'b', 'c'], outputs=['y'], name='gemm')
    inp = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.DOUBLE, [None,*layer.input_shape])
    out = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.DOUBLE, [None,None])
    graph = onnx.helper.make_graph([layer.layer], 'gemm', [inp], [out])
    graph.initializer.append(w)
    graph.initializer.append(b)
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model

def onnx_matmul_model(layer):
    assert isinstance(layer, LayerTestWrapper)
    wshape = (*layer.input_shape, *layer.output_shape)
    w = onnx.helper.make_tensor('b', onnx.TensorProto.DOUBLE, wshape, rand_neg1topos1(*wshape).flatten())
    #node = onnx.helper.make_node('MatMul', inputs=['x', 'b'], outputs=['y'], name='matmul')
    inp = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.DOUBLE, [None,*layer.input_shape])
    out = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.DOUBLE, [None,None])
    graph = onnx.helper.make_graph([layer.layer], 'matmul', [inp], [out])
    graph.initializer.append(w)
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model

onnx_model_makers = {'Relu' : onnx_act_model,
                     'Gemm' : onnx_gemm_model,
                     'MatMul' : onnx_matmul_model}

def validate_model_predictions(model, hls_model, shape, fdata=np.random.rand, test=np.testing.assert_allclose, test_kwargs={'atol':1e-2, 'rtol':1e-2}):
    '''Generate random data with shape, execute inference on model and hls_model, and test for correctness'''
    X = fdata(*shape)
    y_ref = None
    if isinstance(model, tf.keras.models.Model):
        y_ref = model.predict(X)
    elif isinstance(model, torch.nn.Module):
        y_ref = model(torch.Tensor(X)).detach().numpy()
    elif isinstance(model, onnx.onnx_ml_pb2.ModelProto):
        session = onnxruntime.InferenceSession(f'{model.metadata_props[0].value}/model.onnx')
        y_ref = session.run(['y'], {'x' : X})[0]
    y_hls = hls_model.predict(X)
    # Reshape output for Conv models. Use the hls_model's shape for extra validation
    if len(y_ref.shape) > 2:
        y_hls = y_hls.reshape(shape[0], *hls_model.get_output_variables()[0].shape)
    test(y_hls, y_ref, **test_kwargs)

def rand_neg1topos1(*shape):
    return 2 * np.random.rand(*shape) - 1

class LayerTestWrapper:
    '''A representation of the test layer with other information necessary to create and
       validate the behaviour'''
    output_dirs = []
    N = 0

    def __init__(self, layer, output_dir, input_shape, output_shape, n_test, default_precision='ap_fixed<32,16>',
                 fdata=rand_neg1topos1, test=np.testing.assert_allclose, test_kwargs={'atol':1e-2, 'rtol':1e-2}):
        assert isinstance(layer, tf.keras.layers.Layer) or \
          isinstance(layer, torch.nn.Module) or \
            isinstance(layer, onnx.onnx_ml_pb2.NodeProto), \
        """ The layer argument should be an instance of tf.keras.layers.Layer (eg tf.keras.layers.Dense(...)),
            torch.nn.Module (e.g. torch.nn.Linear(...)) or  onnx.onnx_ml_pb2.NodeProto (e.g. onnx.helper.make_node(...) """
        self.layer = layer
        # Make sure every test has a unique output directory
        # Append "_" with incrementing N for each repeat occurence
        if output_dir in LayerTestWrapper.output_dirs:
            output_dir += f'_{LayerTestWrapper.N}'
        self.output_dir = output_dir
        LayerTestWrapper.output_dirs.append(output_dir)
        if isinstance(layer, tf.keras.layers.Layer) or isinstance(layer, onnx.onnx_ml_pb2.NodeProto):  
            self.input_shape = input_shape
        elif isinstance(layer, torch.nn.Module):
            self.input_shape = (1, *input_shape)
        self.output_shape = output_shape
        self.n_test = n_test
        self.data_shape = (n_test, *input_shape)
        self.default_precision = default_precision
        self.fdata = fdata
        self.test = test
        self.test_kwargs = test_kwargs
        LayerTestWrapper.N += 1

    def __repr__(self):
        return f'LayerTestWrapper({self.layer}, {self.output_dir}, {self.input_shape}, {self.n_test})'

odb = 'hls4mlprj_conversion_' # output directory base
# layers & settings to test
# TODO: add ThresholdedReLU test when it can be made to pass
# https://github.com/fastmachinelearning/hls4ml/issues/376     
# Dense layers                                  
keras_layers = [(tf.keras.layers.Dense(16), f'{odb}keras_dense_1', (16,), None, 100),
                (tf.keras.layers.Dense(16, use_bias=False), f'{odb}keras_dense_2', (16,), None, 100),
                # BatchNormalization
                (tf.keras.layers.BatchNormalization(), f'{odb}keras_batchnorm_1', (16,), None, 100),
                # Activation layers
                (tf.keras.layers.Activation(activation='relu'), f'{odb}keras_activation_relu', (1,), None, 1000),
                (tf.keras.layers.LeakyReLU(alpha=1.0), f'{odb}keras_activation_leaky_relu_1', (1,), None, 1000),
                (tf.keras.layers.LeakyReLU(alpha=0.5), f'{odb}keras_activation_leaky_relu_2', (1,), None, 1000),
                (tf.keras.layers.ELU(alpha=1.0), f'{odb}keras_activation_elu_1', (1,), None, 1000),
                (tf.keras.layers.ELU(alpha=0.5), f'{odb}keras_activation_elu_2', (1,), None, 1000),
                (tf.keras.layers.PReLU(alpha_initializer="zeros"), f'{odb}keras_activation_prelu', (1,), None, 1000),
                #LayerTestWrapper(tf.keras.layers.TresholdedReLU(theta=1.0), f'{odb}keras_trelu_1', (1,), None, 1000),
                # Conv1D
                (tf.keras.layers.Conv1D(32, 3, strides=1, padding='valid', use_bias=False),
                                        f'{odb}keras_conv1d_1', (128,4,), None, 100),
                (tf.keras.layers.Conv1D(32, 3, strides=1, padding='same', use_bias=False),
                                        f'{odb}keras_conv1d_2', (128,4,), None, 100),
                (tf.keras.layers.Conv1D(32, 4, strides=2, padding='same', use_bias=False),
                                        f'{odb}keras_conv1d_3', (128,4,), None, 100),                
                # MaxPooling1D
                (tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid'), f'{odb}keras_maxpooling1d_1', (128,4,), None, 100),
                (tf.keras.layers.MaxPooling1D(pool_size=2, padding='same'), f'{odb}keras_maxpooling1d_2', (128,4,), None, 100),
                (tf.keras.layers.MaxPooling1D(pool_size=3, padding='valid'), f'{odb}keras_maxpooling1d_3', (128,4,), None, 100),
                (tf.keras.layers.MaxPooling1D(pool_size=4, padding='valid'), f'{odb}keras_maxpooling1d_4', (128,4,), None, 100),
                (tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'), f'{odb}keras_maxpooling1d_5', (128,4,), None, 100),
                (tf.keras.layers.MaxPooling1D(pool_size=3, strides=3, padding='valid'), f'{odb}keras_maxpooling1d_6', (128,4,), None, 100),
                #(tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='valid'), f'{odb}keras_maxpooling1d_7', (128,4,), None, 100),
                # AveragePooling1D
                (tf.keras.layers.AveragePooling1D(pool_size=2, padding='valid'), f'{odb}keras_averagepooling1d_1', (128,4,), None, 100),
                (tf.keras.layers.AveragePooling1D(pool_size=2, padding='same'), f'{odb}keras_averagepooling1d_2', (128,4,), None, 100),
                (tf.keras.layers.AveragePooling1D(pool_size=3, padding='valid'), f'{odb}keras_averagepooling1d_3', (128,4,), None, 100),
                (tf.keras.layers.AveragePooling1D(pool_size=4, padding='valid'), f'{odb}keras_averagepooling1d_4', (128,4,), None, 100),
                (tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same'), f'{odb}keras_averagepooling1d_5', (128,4,), None, 100),
                (tf.keras.layers.AveragePooling1D(pool_size=3, strides=3, padding='valid'), f'{odb}keras_averagepooling1d_6', (128,4,), None, 100),
                #(tf.keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='valid'), f'{odb}keras_averagepooling1d_7', (128,4,), None, 100),
                # Conv2D
                (tf.keras.layers.Conv2D(32,(4,4),strides=(4,4),padding='same',use_bias=False),
                                        f'{odb}keras_conv2d_1', (28,28,3), None, 100),
                (tf.keras.layers.Conv2D(32,(4,4),strides=(4,4),padding='valid',use_bias=False),
                                        f'{odb}keras_conv2d_2', (28,28,3), None, 100),
                # MaxPooling2D
                (tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='valid'), f'{odb}keras_maxpooling2d_1', (32,32,4,), None, 100),
                (tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'), f'{odb}keras_maxpooling2d_2', (32,32,4,), None, 100),
                (tf.keras.layers.MaxPooling2D(pool_size=(3,3), padding='valid'), f'{odb}keras_maxpooling2d_3', (32,32,4,), None, 100),
                (tf.keras.layers.MaxPooling2D(pool_size=(4,4), padding='valid'), f'{odb}keras_maxpooling2d_4', (32,32,4,), None, 100),
                (tf.keras.layers.MaxPooling2D(pool_size=(2,4), padding='valid'), f'{odb}keras_maxpooling2d_5', (32,32,4,), None, 100),
                # AveragePooling2D
                (tf.keras.layers.AveragePooling2D(pool_size=(2,2), padding='valid'), f'{odb}keras_averagepooling2d_1', (32,32,4,), None, 100),
                (tf.keras.layers.AveragePooling2D(pool_size=(2,2), padding='same'), f'{odb}keras_averagepooling2d_2', (32,32,4,), None, 100),
                (tf.keras.layers.AveragePooling2D(pool_size=(3,3), padding='valid'), f'{odb}keras_averagepooling2d_3', (32,32,4,), None, 100),
                (tf.keras.layers.AveragePooling2D(pool_size=(4,4), padding='valid'), f'{odb}keras_averagepooling2d_4', (32,32,4,), None, 100),
                (tf.keras.layers.AveragePooling2D(pool_size=(2,4), padding='valid'), f'{odb}keras_averagepooling2d_5', (32,32,4,), None, 100),                                                                       
                # Transpose
                (tf.keras.layers.Permute((2,1,3)), f'{odb}keras_transpose', (32,32,4), None, 100)
] # close keras_layers

# TODO: add Linear with bias=False test when it can be made to pass
# TODO: add LeakyReLU, ELU tests when they can pass
# https://github.com/fastmachinelearning/hls4ml/issues/409
pytorch_layers = [(torch.nn.Linear(16, 16, bias=True), f'{odb}pytorch_linear', (16,), None, 100),
                  #(torch.nn.Linear(16, 16, bias=False), f'{odb}pytorch_linear', (16,), None, 100),                 
                  # Activations
                  (torch.nn.ReLU(), f'{odb}pytorch_relu', (16,), None, 100),
                  (torch.nn.LeakyReLU(negative_slope=1.0), f'{odb}pytorch_activation_leakyrelu_1', (16,), None, 100),
                  #(torch.nn.LeakyReLU(negative_slope=0.5), f'{odb}pytorch_activation_leakyrelu_1', (16,), None, 100),
                  (torch.nn.ELU(alpha=1.0), f'{odb}pytorch_activation_elu_1', (16,), None, 100),
                  #(torch.nn.ELU(alpha=0.5), f'{odb}pytorch_activation_elu_1', (16,), None, 100),                 
] # close pytorch_layers

# TODO: find out why Gemm tests don't pass
onnx_layers = [(onnx.helper.make_node('MatMul', inputs=['x', 'b'], outputs=['y'], name='matmul'), f'{odb}onnx_matmul_1', (16,), (16,), 100),
               (onnx.helper.make_node('MatMul', inputs=['x', 'b'], outputs=['y'], name='matmul'), f'{odb}onnx_matmul_2', (16,), (8,), 100),
               (onnx.helper.make_node('MatMul', inputs=['x', 'b'], outputs=['y'], name='matmul'), f'{odb}onnx_matmul_2', (8,), (16,), 100),              
               #(onnx.helper.make_node('Gemm', inputs=['x', 'b', 'c'], outputs=['y'], name='gemm'), f'{odb}onnx_gemm_1', (16,), (16,), 100),
               #(onnx.helper.make_node('Gemm', inputs=['x', 'b', 'c'], outputs=['y'], name='gemm'), f'{odb}onnx_gemm_2', (16,), (8,), 100),
               #(onnx.helper.make_node('Gemm', inputs=['x', 'b', 'c'], outputs=['y'], name='gemm'), f'{odb}onnx_gemm_3', (8,), (16,), 100),
               (onnx.helper.make_node('Relu', inputs=['x'], outputs=['y'], name='relu'), f'{odb}onnx_relu', (1,), None, 100),
] #close onnx_layers

layers = [*keras_layers, *pytorch_layers, *onnx_layers]
names = [layer[1].replace(f'{odb}','') for layer in layers]

@pytest.mark.parametrize('layer', layers, ids=names)
def test_layers(layer):
    layer = LayerTestWrapper(*layer)
    model, hls_model = single_layer_model_factory(layer)
    validate_model_predictions(model, hls_model, layer.data_shape, layer.fdata, layer.test, test_kwargs=layer.test_kwargs)
