import importlib
import os
import warnings

import yaml

from hls4ml.converters.keras_to_hls import KerasFileReader  # noqa: F401
from hls4ml.converters.keras_to_hls import KerasModelReader  # noqa: F401
from hls4ml.converters.keras_to_hls import KerasReader  # noqa: F401
from hls4ml.converters.keras_to_hls import get_supported_keras_layers  # noqa: F401
from hls4ml.converters.keras_to_hls import parse_keras_model  # noqa: F401
from hls4ml.converters.keras_to_hls import keras_to_hls, register_keras_layer_handler
from hls4ml.utils.config import create_config

# ----------Make converters available if the libraries can be imported----------#
try:
    from hls4ml.converters.pytorch_to_hls import (  # noqa: F401
        get_supported_pytorch_layers,
        pytorch_to_hls,
        register_pytorch_layer_handler,
    )

    __pytorch_enabled__ = True
except ImportError:
    warnings.warn("WARNING: Pytorch converter is not enabled!", stacklevel=1)
    __pytorch_enabled__ = False

try:
    from hls4ml.converters.onnx_to_hls import get_supported_onnx_layers  # noqa: F401
    from hls4ml.converters.onnx_to_hls import onnx_to_hls, register_onnx_layer_handler

    __onnx_enabled__ = True
except ImportError:
    warnings.warn("WARNING: ONNX converter is not enabled!", stacklevel=1)
    __onnx_enabled__ = False

# ----------Layer handling register----------#
model_types = ['keras', 'pytorch', 'onnx']

for model_type in model_types:
    for module in os.listdir(os.path.dirname(__file__) + f'/{model_type}'):
        if module == '__init__.py' or module[-3:] != '.py':
            continue
        try:
            lib = importlib.import_module(__name__ + f'.{model_type}.' + module[:-3])
            for _, func in list(lib.__dict__.items()):
                # if 'func' is callable (i.e., function, class...)
                # and has 'handles' attribute
                # and is defined in this module (i.e., not imported)
                if callable(func) and hasattr(func, 'handles') and func.__module__ == lib.__name__:
                    for layer in func.handles:
                        if model_type == 'keras':
                            register_keras_layer_handler(layer, func)
                        elif model_type == 'pytorch':
                            register_pytorch_layer_handler(layer, func)
                        elif model_type == 'onnx':
                            register_onnx_layer_handler(layer, func)

        except ImportError as err:
            print(f'WARNING: Failed to import handlers from {module}: {err.msg}.')
            continue


def parse_yaml_config(config_file):
    """Parse conversion configuration from the provided YAML file.

    This function parses the conversion configuration contained in the YAML
    file provided as an argument. It ensures proper serialization of hls4ml
    objects and should be called on YAML files created by hls4ml. A minimal
    valid YAML file may look like this::

        KerasH5: my_keras_model.h5
        OutputDir: my-hls-test
        ProjectName: myproject
        Part: xcku115-flvb2104-2-i
        ClockPeriod: 5
        IOType: io_stream
        HLSConfig:
            Model:
            Precision: ap_fixed<16,6>
            ReuseFactor: 10

    Please refer to the docs for more examples of valid YAML configurations.

    Arguments:
        config_file (str): Location of the file on the filesystem.

    Returns:
        dict: Parsed configuration.
    """

    def construct_keras_model(loader, node):
        from tensorflow.keras.models import load_model

        model_str = loader.construct_scalar(node)
        return load_model(model_str)

    yaml.add_constructor('!keras_model', construct_keras_model, Loader=yaml.SafeLoader)

    print('Loading configuration from', config_file)
    with open(config_file) as file:
        parsed_config = yaml.safe_load(file)
    return parsed_config


def convert_from_config(config):
    """Convert to hls4ml model based on the provided configuration.

    Arguments:
        config: A string containing the path to the YAML configuration file on
            the filesystem or a dict containing the parsed configuration.

    Returns:
        ModelGraph: hls4ml model.
    """

    if isinstance(config, str):
        yamlConfig = parse_yaml_config(config)
    else:
        yamlConfig = config

    model = None
    if 'OnnxModel' in yamlConfig:
        if __onnx_enabled__:
            model = onnx_to_hls(yamlConfig)
        else:
            raise Exception("ONNX not found. Please install ONNX.")
    elif 'PytorchModel' in yamlConfig:
        if __pytorch_enabled__:
            model = pytorch_to_hls(yamlConfig)
        else:
            raise Exception("PyTorch not found. Please install PyTorch.")
    else:
        model = keras_to_hls(yamlConfig)

    return model


def _check_hls_config(config, hls_config):
    """
    Check hls_config for to set appropriate parameters for config.
    """

    if 'LayerName' in hls_config:
        config['HLSConfig']['LayerName'] = hls_config['LayerName']

    if 'LayerType' in hls_config:
        config['HLSConfig']['LayerType'] = hls_config['LayerType']

    if 'Flows' in hls_config:
        config['HLSConfig']['Flows'] = hls_config['Flows']

    if 'Optimizers' in hls_config:
        config['HLSConfig']['Optimizers'] = hls_config['Optimizers']

    if 'SkipOptimizers' in hls_config:
        config['HLSConfig']['SkipOptimizers'] = hls_config['SkipOptimizers']

    return


def _check_model_config(model_config):
    if model_config is not None:
        if not all(k in model_config for k in ('Precision', 'ReuseFactor')):
            raise Exception('Precision and ReuseFactor must be provided in the hls_config')
    else:
        model_config = {}
        model_config['Precision'] = 'ap_fixed<16,6>'
        model_config['ReuseFactor'] = 1

    return model_config


def convert_from_keras_model(
    model,
    output_dir='my-hls-test',
    project_name='myproject',
    input_data_tb=None,
    output_data_tb=None,
    backend='Vivado',
    hls_config=None,
    **kwargs,
):
    """Convert Keras model to hls4ml model based on the provided configuration.

    Args:
        model: Keras model to convert
        output_dir (str, optional): Output directory of the generated HLS
            project. Defaults to 'my-hls-test'.
        project_name (str, optional): Name of the HLS project.
            Defaults to 'myproject'.
        input_data_tb (str, optional): String representing the path of input data in .npy or .dat format that will be
            used during csim and cosim.
        output_data_tb (str, optional): String representing the path of output data in .npy or .dat format that will be
            used during csim and cosim.
        backend (str, optional): Name of the backend to use, e.g., 'Vivado'
            or 'Quartus'.
        board (str, optional): One of target boards specified in `supported_board.json` file. If set to `None` a default
            device of a backend will be used. See documentation of the backend used.
        part (str, optional): The FPGA part. If set to `None` a default part of a backend will be used.
            See documentation of the backend used. Note that if `board` is specified, the part associated to that board
            will overwrite any part passed as a parameter.
        clock_period (int, optional): Clock period of the design.
            Defaults to 5.
        io_type (str, optional): Type of implementation used. One of
            'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
        hls_config (dict, optional): The HLS config.
        kwargs** (dict, optional): Additional parameters that will be used to create the config of the specified backend

    Raises:
        Exception: If precision and reuse factor are not present in 'hls_config'.

    Returns:
        ModelGraph: hls4ml model.
    """

    config = create_config(output_dir=output_dir, project_name=project_name, backend=backend, **kwargs)

    config['KerasModel'] = model
    config['InputData'] = input_data_tb
    config['OutputPredictions'] = output_data_tb
    config['HLSConfig'] = {}

    if hls_config is None:
        hls_config = {}

    model_config = hls_config.get('Model', None)
    config['HLSConfig']['Model'] = _check_model_config(model_config)

    _check_hls_config(config, hls_config)

    return keras_to_hls(config)


def convert_from_pytorch_model(
    model,
    input_shape,
    output_dir='my-hls-test',
    project_name='myproject',
    input_data_tb=None,
    output_data_tb=None,
    backend='Vivado',
    hls_config=None,
    **kwargs,
):
    """Convert PyTorch model to hls4ml model based on the provided configuration.

    Args:
        model: PyTorch model to convert.
        input_shape (list): The shape of the input tensor. First element is the batch size, needs to be None
        output_dir (str, optional): Output directory of the generated HLS project. Defaults to 'my-hls-test'.
        project_name (str, optional): Name of the HLS project. Defaults to 'myproject'.
        input_data_tb (str, optional): String representing the path of input data in .npy or .dat format that will be
            used during csim and cosim. Defaults to None.
        output_data_tb (str, optional): String representing the path of output data in .npy or .dat format that will be
            used during csim and cosim. Defaults to None.
        backend (str, optional): Name of the backend to use, e.g., 'Vivado' or 'Quartus'. Defaults to 'Vivado'.
        board (str, optional): One of target boards specified in `supported_board.json` file. If set to `None` a default
            device of a backend will be used. See documentation of the backend used.
        part (str, optional): The FPGA part. If set to `None` a default part of a backend will be used.
            See documentation of the backend used. Note that if `board` is specified, the part associated to that board
            will overwrite any part passed as a parameter.
        clock_period (int, optional): Clock period of the design.
            Defaults to 5.
        io_type (str, optional): Type of implementation used. One of
            'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
        hls_config (dict, optional): The HLS config.
        kwargs** (dict, optional): Additional parameters that will be used to create the config of the specified backend.

    Raises:
        Exception: If precision and reuse factor are not present in 'hls_config'.

    Notes:
        Pytorch uses the "channels_first" data format for its tensors, while hls4ml expects the "channels_last" format
        used by keras. By default, hls4ml will automatically add layers to the model which transpose the inputs to the
        "channels_last"format. Not that this is not supported for the "io_stream" io_type, for which the user will have
        to transpose the input by hand before passing it to hls4ml. In that case the "inputs_channel_last" argument of
        the "config_from_pytorch_model" function needs to be set to True. By default, the output of the model remains
        in the "channels_last" data format. The "transpose_outputs" argument of the "config_from_pytorch_model" can be
        used to add a layer to the model that transposes back to "channels_first". As before, this will not work for
        io_stream.

    Returns:
        ModelGraph: hls4ml model.
    """

    config = create_config(output_dir=output_dir, project_name=project_name, backend=backend, **kwargs)

    config['PytorchModel'] = model
    config['InputShape'] = input_shape
    config['InputData'] = input_data_tb
    config['OutputPredictions'] = output_data_tb
    config['HLSConfig'] = {}

    if hls_config is None:
        hls_config = {}

    model_config = hls_config.get('Model', None)
    config['HLSConfig']['Model'] = _check_model_config(model_config)

    _check_hls_config(config, hls_config)

    return pytorch_to_hls(config)


def convert_from_onnx_model(
    model,
    output_dir='my-hls-test',
    project_name='myproject',
    input_data_tb=None,
    output_data_tb=None,
    backend='Vivado',
    hls_config=None,
    **kwargs,
):
    """Convert Keras model to hls4ml model based on the provided configuration.

    Args:
        model: ONNX model to convert.
        output_dir (str, optional): Output directory of the generated HLS
            project. Defaults to 'my-hls-test'.
        project_name (str, optional): Name of the HLS project.
            Defaults to 'myproject'.
        input_data_tb (str, optional): String representing the path of input data in .npy or .dat format that will be
            used during csim and cosim.
        output_data_tb (str, optional): String representing the path of output data in .npy or .dat format that will be
            used during csim and cosim.
        backend (str, optional): Name of the backend to use, e.g., 'Vivado'
            or 'Quartus'.
        board (str, optional): One of target boards specified in `supported_board.json` file. If set to `None` a default
            device of a backend will be used. See documentation of the backend used.
        part (str, optional): The FPGA part. If set to `None` a default part of a backend will be used.
            See documentation of the backend used. Note that if `board` is specified, the part associated to that board
            will overwrite any part passed as a parameter.
        clock_period (int, optional): Clock period of the design.
            Defaults to 5.
        io_type (str, optional): Type of implementation used. One of
            'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
        hls_config (dict, optional): The HLS config.
        kwargs** (dict, optional): Additional parameters that will be used to create the config of the specified backend

    Raises:
        Exception: If precision and reuse factor are not present in 'hls_config'.

    Returns:
        ModelGraph: hls4ml model.
    """

    config = create_config(output_dir=output_dir, project_name=project_name, backend=backend, **kwargs)

    config['OnnxModel'] = model
    config['InputData'] = input_data_tb
    config['OutputPredictions'] = output_data_tb
    config['HLSConfig'] = {}

    if hls_config is None:
        hls_config = {}

    model_config = hls_config.get('Model', None)
    config['HLSConfig']['Model'] = _check_model_config(model_config)

    _check_hls_config(config, hls_config)

    return onnx_to_hls(config)
