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
