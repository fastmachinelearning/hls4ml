import torch

from hls4ml.utils.torch import HLS4MLModule


class SNNReadout(HLS4MLModule):
    """PyTorch marker module for the hls4ml SNNReadout layer.

    In PyTorch this module is an identity. During conversion, hls4ml lowers it
    to the built-in SNNReadout IR layer and backend implementation.
    """

    VALID_OUTPUT_MODES = ('spike', 'membrane')
    VALID_DECISION_RULES = (
        'argmax_spike_count',
        'first_to_threshold',
        'threshold_then_argmax',
        'binary_logit',
        'argmax_membrane',
    )

    def __init__(
        self,
        n_classes=None,
        window_size=1,
        stream_length=None,
        decision_rule=None,
        class_threshold=1,
        output_mode='spike',
        beta=1.0,
        reset_policy='fixed_window',
    ):
        super().__init__()

        output_mode = str(output_mode).lower()
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(f'Unsupported SNNReadout output_mode "{output_mode}". Supported: spike, membrane.')

        if decision_rule is None:
            decision_rule = 'argmax_membrane' if output_mode == 'membrane' else 'argmax_spike_count'
        decision_rule = str(decision_rule)
        if decision_rule not in self.VALID_DECISION_RULES:
            raise ValueError(
                f'Unsupported SNNReadout decision_rule "{decision_rule}". Supported: {", ".join(self.VALID_DECISION_RULES)}.'
            )
        if output_mode == 'membrane' and decision_rule not in ('argmax_membrane', 'binary_logit'):
            raise ValueError('SNNReadout membrane mode supports decision_rule "argmax_membrane" or "binary_logit".')
        if output_mode == 'spike' and decision_rule == 'argmax_membrane':
            raise ValueError('SNNReadout decision_rule "argmax_membrane" requires output_mode "membrane".')

        self.n_classes = n_classes
        if stream_length is None:
            self.window_size = int(window_size)
        else:
            self.stream_length = int(stream_length)
        self.decision_rule = decision_rule
        self.class_threshold = int(class_threshold)
        self.output_mode = output_mode
        self.beta = torch.tensor(float(beta))
        self.reset_policy = str(reset_policy)

    def forward(self, x):
        return x


class StreamingSNN(HLS4MLModule):
    """Spiking network that hls4ml converts into one streaming block (arXiv:2611.00104).

    Maps (batch, n_steps) inputs, one sample per time step, to the number of output spikes, (batch,). The layers are
    ``Linear`` or ``Conv1d`` stages (all of one kind before the output stage), each followed by an snnTorch ``Leaky``; a
    ``Conv1d`` with kernel size K > 1 follows ``ConstantPad1d((K - 1, 0), 0.0)``. The last stage has one neuron. Pass it
    to the converter as a submodule, e.g. ``torch.nn.Sequential(OrderedDict(snn=model))``.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        if isinstance(x, torch.fx.Proxy):
            raise TypeError(
                'StreamingSNN cannot be the model passed to hls4ml; pass it as a submodule instead, '
                'for example torch.nn.Sequential(OrderedDict(snn=model)).'
            )
        s = x.unsqueeze(1)
        for layer in self.layers:
            if type(layer).__name__ == 'Leaky':
                mem, spikes = torch.zeros_like(s[..., 0]), []
                for t in range(s.shape[-1]):
                    spk, mem = layer(s[..., t], mem)
                    spikes.append(spk)
                s = torch.stack(spikes, dim=-1)
            elif isinstance(layer, torch.nn.Linear):
                s = layer(s.transpose(1, 2)).transpose(1, 2)
            else:
                s = layer(s)
        return s.squeeze(1).sum(-1)
