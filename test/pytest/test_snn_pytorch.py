from pathlib import Path

import pytest
import torch

import hls4ml
import hls4ml.utils.torch

test_root_path = Path(__file__).parent


class Leaky(hls4ml.utils.torch.HLS4MLModule):
    def __init__(self, beta=0.9, threshold=1.0, reset_mechanism='subtract'):
        super().__init__()
        self.beta = torch.tensor(beta)
        self.threshold = torch.tensor(threshold)
        self.reset_mechanism = reset_mechanism

    def forward(self, x):
        return x


class LIFNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.neuron = Leaky(beta=0.95, threshold=1.2, reset_mechanism='subtract')

    def forward(self, x):
        x = self.fc(x)
        return self.neuron(x)


class SNNReadout(hls4ml.utils.torch.HLS4MLModule):
    def __init__(self, n_classes=4, window_size=16, decision_rule='argmax_spike_count', class_threshold=2):
        super().__init__()
        self.n_classes = n_classes
        self.window_size = window_size
        self.decision_rule = decision_rule
        self.class_threshold = class_threshold

    def forward(self, x):
        return x


class IFNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.neuron = Leaky(beta=1.0, threshold=0.8, reset_mechanism='zero')

    def forward(self, x):
        x = self.fc(x)
        return self.neuron(x)


class SNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.neuron = Leaky(beta=0.95, threshold=1.2, reset_mechanism='subtract')
        self.readout = SNNReadout(
            n_classes=4, window_size=12, decision_rule='threshold_then_argmax', class_threshold=3
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.neuron(x)
        return self.readout(x)


class IFNetTol(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.neuron = Leaky(beta=1.0 - 5e-7, threshold=0.8, reset_mechanism='subtract')

    def forward(self, x):
        x = self.fc(x)
        return self.neuron(x)


@pytest.mark.parametrize(
    'model_class,expected_layer',
    [
        (LIFNet, 'LIFNeuron'),
        (IFNet, 'IFNeuron'),
        (IFNetTol, 'IFNeuron'),
        (SNNClassifier, 'SNNReadout'),
    ],
)
def test_pytorch_snn_layers_are_parsed(test_case_id, model_class, expected_layer):
    model = model_class()
    config = hls4ml.utils.config_from_pytorch_model(
        model, (4,), default_precision='ap_fixed<16,6>', granularity='name', backend='Vivado'
    )

    hmodel = hls4ml.converters.convert_from_pytorch_model(
        model,
        output_dir=str(test_root_path / test_case_id),
        backend='Vivado',
        io_type='io_parallel',
        hls_config=config,
    )

    layer_names = [layer.class_name for layer in hmodel.get_layers()]
    assert expected_layer in layer_names

    if expected_layer == 'SNNReadout':
        readout = [layer for layer in hmodel.get_layers() if layer.class_name == 'SNNReadout'][0]
        assert readout.get_attr('decision_rule') == 'threshold_then_argmax'
        assert readout.get_attr('window_size') == 12
        assert readout.get_attr('class_threshold') == 3


@pytest.mark.parametrize(
    'beta,expected_layer',
    [
        (1.0, 'IFNeuron'),
        (1.0 - 5e-7, 'IFNeuron'),
        (0.999, 'LIFNeuron'),
        (0.95, 'LIFNeuron'),
    ],
)
def test_leaky_beta_maps_to_if_or_lif(test_case_id, beta, expected_layer):
    class BetaNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
            self.neuron = Leaky(beta=beta, threshold=1.0, reset_mechanism='subtract')

        def forward(self, x):
            x = self.fc(x)
            return self.neuron(x)

    model = BetaNet()
    config = hls4ml.utils.config_from_pytorch_model(
        model, (4,), default_precision='ap_fixed<16,6>', granularity='name', backend='Vivado'
    )

    hmodel = hls4ml.converters.convert_from_pytorch_model(
        model,
        output_dir=str(test_root_path / test_case_id),
        backend='Vivado',
        io_type='io_parallel',
        hls_config=config,
    )

    layer_names = [layer.class_name for layer in hmodel.get_layers()]
    assert expected_layer in layer_names
