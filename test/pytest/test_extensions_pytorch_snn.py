from pathlib import Path

import pytest
import torch

import hls4ml
import hls4ml.utils.torch

test_root_path = Path(__file__).parent


class TSNNWindowReadout(hls4ml.utils.torch.HLS4MLModule):
    """Example custom PyTorch module mapped to builtin SNNReadout."""

    def __init__(self, n_classes=4, window_size=8, decision_rule='argmax_spike_count', class_threshold=2):
        super().__init__()
        self.n_classes = n_classes
        self.window_size = window_size
        self.decision_rule = decision_rule
        self.class_threshold = class_threshold

    def forward(self, x):
        return x


def parse_custom_snn_readout(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'TSNNWindowReadout'

    layer = {}
    layer['class_name'] = 'SNNReadout'
    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['n_classes'] = int(class_object.n_classes)
    layer['window_size'] = int(class_object.window_size)
    layer['class_threshold'] = int(class_object.class_threshold)
    layer['decision_rule'] = str(class_object.decision_rule)

    output_shape = input_shapes[0][:]
    output_shape[-1] = 1
    return layer, output_shape


@pytest.fixture(scope='session', autouse=True)
def register_custom_snn_extension():
    hls4ml.converters.register_pytorch_layer_handler('TSNNWindowReadout', parse_custom_snn_readout)


def test_extensions_pytorch_snn_readout_parser(test_case_id):
    class PyTorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
            self.readout = TSNNWindowReadout(
                n_classes=4, window_size=10, decision_rule='threshold_then_argmax', class_threshold=3
            )

        def forward(self, x):
            x = self.fc(x)
            return self.readout(x)

    pmodel = PyTorchModel()
    config = hls4ml.utils.config_from_pytorch_model(
        pmodel, (4,), default_precision='ap_fixed<16,6>', granularity='name', backend='Vivado'
    )
    hmodel = hls4ml.converters.convert_from_pytorch_model(
        pmodel,
        output_dir=str(test_root_path / test_case_id),
        backend='Vivado',
        io_type='io_parallel',
        hls_config=config,
    )

    readouts = [layer for layer in hmodel.get_layers() if layer.class_name == 'SNNReadout']
    assert len(readouts) == 1
    assert readouts[0].get_attr('decision_rule') == 'threshold_then_argmax'
    assert readouts[0].get_attr('window_size') == 10
