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


class SNNReadoutWithResetPolicy(hls4ml.utils.torch.HLS4MLModule):
    def __init__(self, n_classes=4, stream_length=7, decision_rule='argmax_spike_count', class_threshold=2, reset_policy='tlast'):
        super().__init__()
        self.n_classes = n_classes
        self.stream_length = stream_length
        self.decision_rule = decision_rule
        self.class_threshold = class_threshold
        self.reset_policy = reset_policy

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


class SNNClassifierWithResetPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.neuron = Leaky(beta=0.95, threshold=1.2, reset_mechanism='subtract')
        self.readout = SNNReadoutWithResetPolicy(
            n_classes=4, stream_length=7, decision_rule='first_to_threshold', class_threshold=2, reset_policy='host_pulse'
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


class LIFVectorNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.neuron = Leaky(beta=[0.8, 0.9, 0.85, 0.95], threshold=[1.1, 1.0, 0.9, 1.2], reset_mechanism='subtract')

    def forward(self, x):
        x = self.fc(x)
        return self.neuron(x)


class IFVectorThresholdNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.neuron = Leaky(beta=1.0, threshold=[0.8, 0.9, 1.0, 1.1], reset_mechanism='zero')

    def forward(self, x):
        x = self.fc(x)
        return self.neuron(x)


class LearnedVectorParamsNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.neuron = Leaky(beta=[0.2, 0.3, 0.4, 0.5], threshold=[0.2, 0.3, 0.4, 0.5], reset_mechanism='subtract')
        # Simulate learned values being different from initialization at conversion time.
        self.neuron.beta = torch.nn.Parameter(torch.tensor([0.72, 0.81, 0.63, 0.94]))
        self.neuron.threshold = torch.nn.Parameter(torch.tensor([1.25, 0.95, 1.05, 0.85]))

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


@pytest.mark.parametrize(
    'model_class,expected_layer,beta_mode,threshold_mode',
    [
        (LIFVectorNet, 'LIFNeuron', 'vector', 'vector'),
        (IFVectorThresholdNet, 'IFNeuron', None, 'vector'),
        (LIFNet, 'LIFNeuron', 'scalar', 'scalar'),
        (IFNet, 'IFNeuron', None, 'scalar'),
    ],
)
def test_snn_scalar_vs_vector_modes(test_case_id, model_class, expected_layer, beta_mode, threshold_mode):
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

    layer = [layer for layer in hmodel.get_layers() if layer.class_name == expected_layer][0]
    assert layer.get_attr('threshold_mode') == threshold_mode
    if threshold_mode == 'vector':
        assert layer.get_weights('threshold_vec').data.shape[0] == layer.get_attr('n_out')
    if expected_layer == 'LIFNeuron':
        assert layer.get_attr('beta_mode') == beta_mode
        if beta_mode == 'vector':
            assert layer.get_weights('beta_vec').data.shape[0] == layer.get_attr('n_out')


def test_snn_uses_current_parameter_values_for_vector_params(test_case_id):
    model = LearnedVectorParamsNet()
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

    layer = [layer for layer in hmodel.get_layers() if layer.class_name == 'LIFNeuron'][0]
    assert layer.get_attr('beta_mode') == 'vector'
    assert layer.get_attr('threshold_mode') == 'vector'
    assert list(layer.get_weights('beta_vec').data) == pytest.approx([0.72, 0.81, 0.63, 0.94])
    assert list(layer.get_weights('threshold_vec').data) == pytest.approx([1.25, 0.95, 1.05, 0.85])


def test_snn_readout_stream_length_alias_and_reset_policy(test_case_id):
    model = SNNClassifierWithResetPolicy()
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

    readout = [layer for layer in hmodel.get_layers() if layer.class_name == 'SNNReadout'][0]
    assert readout.get_attr('window_size') == 7
    assert readout.get_attr('decision_rule') == 'first_to_threshold'
    assert readout.get_attr('state_reset_policy') == 'host_pulse'


def test_snn_layer_type_config_is_exposed_for_quantization(test_case_id):
    model = LIFVectorNet()
    config = hls4ml.utils.config_from_pytorch_model(
        model, (4,), default_precision='ap_fixed<16,6>', granularity='name', backend='Vivado'
    )
    config['LayerName']['neuron']['beta_t'] = 'ap_fixed<12,2>'
    config['LayerName']['neuron']['threshold_t'] = 'ap_fixed<10,3>'
    config['LayerName']['neuron']['membrane_t'] = 'ap_fixed<14,4>'

    hmodel = hls4ml.converters.convert_from_pytorch_model(
        model,
        output_dir=str(test_root_path / test_case_id),
        backend='Vivado',
        io_type='io_parallel',
        hls_config=config,
    )

    layer = [layer for layer in hmodel.get_layers() if layer.class_name == 'LIFNeuron'][0]
    assert layer.get_attr('beta_t').precision.definition_cpp() == 'ap_fixed<12,2>'
    assert layer.get_attr('threshold_t').precision.definition_cpp() == 'ap_fixed<10,3>'
    assert layer.get_attr('membrane_t').precision.definition_cpp() == 'ap_fixed<14,4>'
