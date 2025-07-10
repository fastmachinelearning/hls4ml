import importlib
from pathlib import Path

import numpy as np
import pytest

import hls4ml

CONV2D_WIDTH_HEIGHT = 12
CONV2D_IN_CHANNELS = 4
CONV2D_OUT_CHANNELS = 8
CONV1D_OUT_CHANNELS = 4
CONV_KERNEL_SIZE = 3
CONV1D_KERNEL_SIZE = 3
LINEAR_INPUT_UNITS = 48
BATCH_SIZE = 10


# PQuant Functions


def set_layer_quantization_attributes(layer, config, CompressedLayerBase, new_layer=None):
    if isinstance(layer, CompressedLayerBase):
        if config["quantization_parameters"]["use_high_granularity_quantization"]:
            i_weight = layer.hgq_weight.quantizer.i.cpu().numpy()
            f_weight = layer.hgq_weight.quantizer.f.cpu().numpy()
            quantization_parameters = {
                "i_weight": i_weight,
                "f_weight": f_weight,
                "k_weight": 1.0,
                "overflow": layer.overflow,
            }
            if layer.use_bias:
                i_bias = layer.hgq_bias.quantizer.i.cpu().numpy()
                f_bias = layer.hgq_bias.quantizer.f.cpu().numpy()

                quantization_parameters["i_bias"] = i_bias.cpu().numpy()
                quantization_parameters["f_bias"] = f_bias.cpu().numpy()
                quantization_parameters["k_bias"] = 1.0
        else:
            quantization_parameters = {
                "i_weight": layer.i_weight.cpu().numpy(),
                "f_weight": layer.f_weight.cpu().numpy(),
                "k_weight": 1.0,
                "i_bias": layer.i_bias.cpu().numpy(),
                "f_bias": layer.f_bias.cpu().numpy(),
                "k_bias": 1.0,
                "overflow": layer.overflow,
            }
        quantization_parameters["use_high_granularity_quantization"] = config["quantization_parameters"][
            "use_high_granularity_quantization"
        ]
    if new_layer is not None:
        new_layer.quantization_parameters = quantization_parameters
    return quantization_parameters


def set_activation_quantization_parameters(layer, config):
    if config["quantization_parameters"]["use_high_granularity_quantization"]:
        i = layer.hgq.quantizer.i.cpu().numpy()
        f = layer.hgq.quantizer.f.cpu().numpy()
        k = layer.hgq.quantizer.k.cpu().numpy()
        quantization_parameters = {
            "i_act": i,
            "f_act": f,
            "k_act": k,
            "overflow": layer.overflow,
        }
    else:
        quantization_parameters = {
            "i_act": layer.i.cpu().numpy(),
            "f_act": layer.f.cpu().numpy(),
            "k_act": layer.k.cpu().numpy(),
            "overflow": layer.overflow,
        }

    quantization_parameters["use_high_granularity_quantization"] = config["quantization_parameters"][
        "use_high_granularity_quantization"
    ]
    layer.quantization_parameters = quantization_parameters
    return quantization_parameters


def set_pooling_quantization_parameters(layer, config):
    if config["quantization_parameters"]["use_high_granularity_quantization"]:
        i = layer.hgq.quantizer.i.cpu().numpy()
        f = layer.hgq.quantizer.f.cpu().numpy()
        k = layer.hgq.quantizer.k.cpu().numpy()
        quantization_parameters = {
            "i_pool": i,
            "f_pool": f,
            "k_pool": k,
            "overflow": layer.overflow,
        }
    else:
        quantization_parameters = {
            "i_pool": layer.i.cpu().numpy(),
            "f_pool": layer.f.cpu().numpy(),
            "k_pool": 1.0,
            "overflow": layer.overflow,
        }

    quantization_parameters["use_high_granularity_quantization"] = config["quantization_parameters"][
        "use_high_granularity_quantization"
    ]
    layer.quantization_parameters = quantization_parameters
    return quantization_parameters


def remove_pruning_from_model_torch(module, config):
    import torch.nn as nn
    from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
    from pquant.core.torch_impl.compressed_layers_torch import (
        CompressedLayerBase,
        CompressedLayerConv1d,
        CompressedLayerConv2d,
        CompressedLayerLinear,
        QuantizedPooling,
    )

    for name, layer in module.named_children():
        if isinstance(layer, CompressedLayerLinear):
            if config["pruning_parameters"]["pruning_method"] == "pdp":  # Find better solution later
                if config["training_parameters"]["pruning_first"]:
                    weight = layer.pruning_layer.get_hard_mask(layer.weight) * layer.weight
                    weight, bias = layer.quantize(weight, layer.bias)
                else:
                    weight, bias = layer.quantize(layer.weight, layer.bias)
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight
            else:
                weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
            out_features = layer.out_features
            bias_values = bias
            in_features = layer.in_features
            bias = True if bias_values is not None else False
            new_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
            set_layer_quantization_attributes(layer, config, CompressedLayerBase, new_layer)
            setattr(module, name, new_layer)
            getattr(module, name).weight.data.copy_(weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(bias_values.data)
        elif isinstance(layer, (CompressedLayerConv2d, CompressedLayerConv1d)):
            if config["pruning_parameters"]["pruning_method"] == "pdp":  # Find better solution later
                if config["training_parameters"]["pruning_first"]:
                    weight = layer.pruning_layer.get_hard_mask(layer.weight) * layer.weight
                    weight, bias = layer.quantize(weight, layer.bias)
                else:
                    weight, bias = layer.quantize(layer.weight, layer.bias)
                    weight = layer.pruning_layer.get_hard_mask(weight) * weight
            else:
                weight, bias = layer.prune_and_quantize(layer.weight, layer.bias)
            bias_values = bias
            bias = True if bias_values is not None else False
            conv = nn.Conv2d if isinstance(layer, CompressedLayerConv2d) else nn.Conv1d
            set_layer_quantization_attributes(layer, CompressedLayerBase, config, conv)

            setattr(
                module,
                name,
                conv(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                    layer.groups,
                    bias,
                    layer.padding_mode,
                ),
            )
            getattr(module, name).weight.data.copy_(weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(bias_values.data)
        elif isinstance(layer, (QuantizedTanh, QuantizedReLU)):
            set_activation_quantization_parameters(layer, config)
        elif isinstance(layer, QuantizedPooling):
            set_pooling_quantization_parameters(layer, config)
        else:
            remove_pruning_from_model_torch(layer, config)
    return module


def remove_pruning_from_model_tf(model, config):
    import keras
    from keras.layers import Activation, Conv1D, Conv2D, Dense, DepthwiseConv2D, SeparableConv2D
    from pquant.core.activations_quantizer import QuantizedReLU, QuantizedTanh
    from pquant.core.tf_impl.compressed_layers_tf import (
        CompressedLayerBase,
        CompressedLayerConv1dKeras,
        CompressedLayerConv2dKeras,
        CompressedLayerDenseKeras,
        CompressedLayerDepthwiseConv2dKeras,
        CompressedLayerSeparableConv2dKeras,
        QuantizedPooling,
        _prune_and_quantize_layer,
    )

    x = model.layers[0].output
    for layer in model.layers[1:]:
        if isinstance(layer, CompressedLayerDepthwiseConv2dKeras):
            new_layer = DepthwiseConv2D(
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                dilation_rate=layer.dilation_rate,
                use_bias=layer.use_bias,
                depthwise_regularizer=layer.depthwise_regularizer,
                activity_regularizer=layer.activity_regularizer,
            )
            set_layer_quantization_attributes(layer, config, CompressedLayerBase, new_layer)
            x = new_layer(x)
            use_bias = layer.use_bias
            weight, bias = _prune_and_quantize_layer(layer, use_bias)
            new_layer.set_weights([weight, bias] if use_bias else [weight])
        elif isinstance(layer, CompressedLayerConv2dKeras):
            new_layer = Conv2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                dilation_rate=layer.dilation_rate,
                use_bias=layer.use_bias,
                kernel_regularizer=layer.kernel_regularizer,
                activity_regularizer=layer.activity_regularizer,
            )
            set_layer_quantization_attributes(layer, config, CompressedLayerBase, new_layer)
            x = new_layer(x)
            use_bias = layer.use_bias
            weight, bias = _prune_and_quantize_layer(layer, use_bias)
            new_layer.set_weights([weight, bias] if use_bias else [weight])
        elif isinstance(layer, CompressedLayerSeparableConv2dKeras):
            if not layer.enable_quantization:
                new_layer = SeparableConv2D(
                    filters=layer.pointwise_conv.filters,
                    kernel_size=layer.depthwise_conv.kernel_size,
                    strides=layer.depthwise_conv.strides,
                    padding=layer.depthwise_conv.padding,
                    dilation_rate=layer.depthwise_conv.dilation_rate,
                    use_bias=layer.pointwise_conv.use_bias,
                    depthwise_regularizer=layer.depthwise_conv.depthwise_regularizer,
                    pointwise_regularizer=layer.pointwise_conv.kernel_regularizer,
                    activity_regularizer=layer.activity_regularizer,
                )
                set_layer_quantization_attributes(layer, config, CompressedLayerBase, new_layer)
                x = new_layer(x)
                use_bias = layer.pointwise_conv.use_bias
                depthwise_weight, _ = _prune_and_quantize_layer(layer.depthwise_conv, False)
                pointwise_weight, bias = _prune_and_quantize_layer(layer.pointwise_conv, layer.pointwise_conv.use_bias)
                new_layer.set_weights(
                    [depthwise_weight, pointwise_weight, bias] if use_bias else [depthwise_weight, pointwise_weight]
                )

            else:
                new_layer_depthwise = DepthwiseConv2D(
                    kernel_size=layer.depthwise_conv.kernel_size,
                    strides=layer.depthwise_conv.strides,
                    padding=layer.depthwise_conv.padding,
                    dilation_rate=layer.depthwise_conv.dilation_rate,
                    data_format=layer.data_format,
                    use_bias=False,
                )
                new_layer_pointwise = Conv2D(
                    kernel_size=1,
                    filters=layer.pointwise_conv.filters,
                    use_bias=layer.pointwise_conv.use_bias,
                    padding=layer.pointwise_conv.padding,
                    dilation_rate=layer.pointwise_conv.dilation_rate,
                    data_format=layer.data_format,
                    activity_regularizer=layer.pointwise_conv.activity_regularizer,
                )
                set_layer_quantization_attributes(layer.depthwise_conv, config, CompressedLayerBase, new_layer_depthwise)
                set_layer_quantization_attributes(layer.pointwise_conv, config, CompressedLayerBase, new_layer_pointwise)
                x = new_layer_depthwise(x)
                depthwise_weight, _ = _prune_and_quantize_layer(layer.depthwise_conv, False)
                new_layer_depthwise.set_weights([depthwise_weight])

                if layer.enable_quantization:
                    if layer.use_high_granularity_quantization:
                        x = layer.hgq(x)
                    else:
                        quantizer = Activation(lambda x, q=layer.quantizer, k=1.0, i=layer.i, f=layer.f: q(x, k, i, f))
                        x = quantizer(x)

                x = new_layer_pointwise(x)
                use_bias = layer.pointwise_conv.use_bias
                pointwise_weight, bias = _prune_and_quantize_layer(layer.pointwise_conv, layer.pointwise_conv.use_bias)
                new_layer_pointwise.set_weights([pointwise_weight, bias] if use_bias else [pointwise_weight])
        elif isinstance(layer, CompressedLayerConv1dKeras):
            new_layer = Conv1D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                dilation_rate=layer.dilation_rate,
                use_bias=layer.use_bias,
                kernel_regularizer=layer.kernel_regularizer,
                activity_regularizer=layer.activity_regularizer,
            )
            set_layer_quantization_attributes(layer, config, CompressedLayerBase, new_layer)
            x = new_layer(x)
            use_bias = layer.use_bias
            weight, bias = _prune_and_quantize_layer(layer, use_bias)
            new_layer.set_weights([weight, bias] if use_bias else [weight])
        elif isinstance(layer, CompressedLayerDenseKeras):
            new_layer = Dense(units=layer.units, use_bias=layer.use_bias, kernel_regularizer=layer.kernel_regularizer)
            set_layer_quantization_attributes(layer, config, CompressedLayerBase, new_layer)
            x = new_layer(x)
            use_bias = new_layer.use_bias
            weight, bias = _prune_and_quantize_layer(layer, use_bias)
            new_layer.set_weights([weight, bias] if use_bias else [weight])
        elif isinstance(layer, (QuantizedTanh, QuantizedReLU)):
            set_activation_quantization_parameters(layer, config)
        elif isinstance(layer, QuantizedPooling):
            set_activation_quantization_parameters(layer, config)
            x = layer(x)
        else:
            x = layer(x)
    replaced_model = keras.Model(inputs=model.inputs, outputs=x)
    return replaced_model


# PyTorch Functions


def get_pytorch_model(INPUT_SHAPE):
    import torch.nn as nn

    class TestModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv2d = nn.Conv2d(CONV2D_IN_CHANNELS, CONV2D_OUT_CHANNELS, kernel_size=3, stride=2, padding=1, bias=True)
            self.relu = nn.ReLU()
            self.conv1d = nn.Conv1d(CONV2D_OUT_CHANNELS, CONV1D_OUT_CHANNELS, kernel_size=3, stride=1, padding=1, bias=True)
            self.tanh = nn.Tanh()
            self.avg = nn.AvgPool1d(kernel_size=3, stride=3)
            self.linear1 = nn.Linear(LINEAR_INPUT_UNITS, 10, bias=True)

        def forward(self, x):
            x = self.conv2d(x)
            x = self.relu(x)
            x = x.view(8, 36)
            x = self.conv1d(x)
            x = self.tanh(x)
            x = self.avg(x)
            x = x.view(-1)
            x = self.linear1(x)
            return x

    return TestModel()


def pass_through_hls4ml_pytorch(model, INPUT_SHAPE):
    backend = 'Vitis'
    default_precision = 'ap_fixed<32, 16>' if backend in ['Vivado', 'Vitis'] else 'ac_fixed<32, 16, true>'
    hls_config = hls4ml.utils.config_from_pytorch_model(
        model, input_shape=INPUT_SHAPE, granularity='name', default_precision=default_precision, backend=backend
    )

    output_dir = str(Path(__file__).parent / 'pytorch_test')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type='io_parallel', part='xc7a15tcpg236-3'
    )
    hls_model.compile()

    return hls_model


def create_data_pytorch(INPUT_SHAPE):
    import torch

    return torch.rand(INPUT_SHAPE)


# Keras Functions


def get_keras_model(INPUT_SHAPE):
    from keras.layers import (
        Activation,
        AveragePooling1D,
        Conv1D,
        Conv2D,
        Dense,
        DepthwiseConv2D,
        Flatten,
        Input,
        ReLU,
        Reshape,
    )
    from keras.models import Model

    inputs = Input(shape=INPUT_SHAPE)
    print(inputs.shape)
    x = DepthwiseConv2D(CONV_KERNEL_SIZE, strides=(2, 2))(inputs)
    print(x.shape)
    x = Conv2D(CONV2D_OUT_CHANNELS, CONV_KERNEL_SIZE, strides=(2, 2))(x)
    print(x.shape)
    x = ReLU()(x)
    x = Reshape((4, 8))(x)
    print(x.shape)
    x = Conv1D(filters=CONV1D_OUT_CHANNELS, kernel_size=CONV1D_KERNEL_SIZE, strides=1, padding="same", use_bias=True)(x)
    print(x.shape)
    x = Activation("tanh")(x)
    x = AveragePooling1D(2)(x)
    print(x.shape)
    x = Flatten()(x)
    print(x.shape)
    x = Dense(10)(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def pass_through_hls4ml_keras(model, INPUT_SHAPE):
    backend = 'Vitis'
    default_precision = 'ap_fixed<32, 16>' if backend in ['Vivado', 'Vitis'] else 'ac_fixed<32, 16, true>'
    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision=default_precision, backend=backend
    )

    output_dir = str(Path(__file__).parent / 'pytorch_test')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type='io_parallel', part='xc7a15tcpg236-3'
    )
    hls_model.compile()

    return hls_model


def create_data_keras(INPUT_SHAPE):
    import keras

    return keras.random.uniform(INPUT_SHAPE)


# Configuration Dictionary


def framework_config(framework):
    config = {
        'pytorch': {
            'get_model': get_pytorch_model,
            'remove_pruning': remove_pruning_from_model_torch,
            'pass_through_hls4ml': pass_through_hls4ml_pytorch,
            'create_data': create_data_pytorch,
            'INPUT_SHAPE': (CONV2D_IN_CHANNELS, CONV2D_WIDTH_HEIGHT, CONV2D_WIDTH_HEIGHT),
        },
        'keras': {
            'get_model': get_keras_model,
            'remove_pruning': remove_pruning_from_model_tf,
            'pass_through_hls4ml': pass_through_hls4ml_keras,
            'create_data': create_data_keras,
            'INPUT_SHAPE': (CONV2D_WIDTH_HEIGHT, CONV2D_WIDTH_HEIGHT, CONV2D_IN_CHANNELS),
        },
    }
    return config[framework]


# Act


def get_model(framework, config):
    if framework == 'pytorch':
        import os

        os.environ["KERAS_BACKEND"] = "torch"  # Needs to be set, some pruning layers as well as the quantizers are Keras
        pretrain_module = importlib.import_module('pquant.core.torch_impl.compressed_layers_torch')
        # layer_module = importlib.import_module('pquant.core.torch_impl.compressed_layers_torch')
        # compression = 'add_compression_layers_torch'
    elif framework == 'keras':
        import os

        os.environ["KERAS_BACKEND"] = "tensorflow"
        pretrain_module = importlib.import_module('pquant.core.tf_impl.compressed_layers_tf')
        # layer_module = importlib.import_module('pquant')
        # compression = 'add_compression_layers'
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    from pquant import add_compression_layers, get_default_config

    model = config['get_model'](config['INPUT_SHAPE'])

    # prune and quantize the model
    pquant_config = get_default_config("pdp")
    pquant_config["pruning_parameters"]["epsilon"] = 1.0
    PQUANT_SHAPE = (BATCH_SIZE, *(config['INPUT_SHAPE']))
    # model = getattr(layer_module, compression)(model, pquant_config, PQUANT_SHAPE)
    model = add_compression_layers(model, pquant_config, PQUANT_SHAPE)
    pretrain_module.post_pretrain_functions(model, pquant_config)
    model = config['remove_pruning'](model, pquant_config)

    return model


# Assert


@pytest.mark.parametrize('framework', ['pytorch', 'keras'])
def test_pquant(framework):

    # setup
    config = framework_config(framework)
    model = get_model(framework, config)

    # pass it through hls4ml
    hls_model = config['pass_through_hls4ml'](model, config['INPUT_SHAPE'])

    # predict
    data = config['create_data']((100 * BATCH_SIZE, *(config['INPUT_SHAPE'])))
    prediction = model(data)
    if framework == 'pytorch':
        prediction = prediction.detach()
    prediction = prediction.numpy().flatten()
    hls_prediction = hls_model.predict(data.numpy()).flatten()

    np.testing.assert_allclose(hls_prediction, prediction, rtol=0.0, atol=5e-3)
