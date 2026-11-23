import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import hls4ml
from hls4ml.contrib.snntorch import StreamingSNN

test_root_path = Path(__file__).parent


class Leaky(nn.Module):
    """Stand-in for snntorch.Leaky, which CI does not install: subtractive reset, one time step after the spike."""

    def __init__(self, beta, threshold):
        super().__init__()
        self.beta = torch.tensor(beta)
        self.threshold = torch.tensor(threshold)

    def forward(self, input_, mem):
        reset = (mem > self.threshold).to(mem.dtype)
        mem = self.beta.clamp(0, 1) * mem + input_ - reset * self.threshold
        return (mem > self.threshold).to(mem.dtype), mem


@pytest.mark.parametrize('family', ['dense', 'conv'])
def test_streaming_snn(test_case_id, family):
    if family == 'dense':
        layers = [nn.Linear(1, 6), Leaky(0.5, 1.0), nn.Linear(6, 4), Leaky(1.0, 0.75)]
    else:
        layers = [nn.ConstantPad1d((2, 0), 0.0), nn.Conv1d(1, 6, 3), Leaky(0.5, 1.0)]
        layers += [nn.ConstantPad1d((1, 0), 0.0), nn.Conv1d(6, 4, 2), Leaky(1.0, 0.75)]
    model = StreamingSNN(*layers, nn.Linear(4, 1), Leaky(0.5, 0.5))

    # Values on coarse grids keep the block exact and the membranes off the thresholds, where the block (>=) and
    # snnTorch (>) would differ
    rng = np.random.default_rng(5)
    for i, module in enumerate(m for m in model.layers if isinstance(m, (nn.Linear, nn.Conv1d))):
        module.weight.data = torch.tensor(
            rng.integers(-6 if i == 0 else -2, 7, module.weight.shape) / 4, dtype=torch.float32
        )
        module.bias.data = torch.tensor(rng.choice([-9, -5, -1, 3, 7], module.bias.shape) / 64, dtype=torch.float32)
    x = (rng.integers(-16, 16, (32, 12)) / 4).astype(np.float32)

    top = nn.Sequential(OrderedDict(snn=model))
    config = hls4ml.utils.config_from_pytorch_model(top, (12,), granularity='name', backend='Vitis')
    config['LayerName']['snn']['Precision']['membrane'] = 'ap_fixed<32,8>'
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        top, hls_config=config, output_dir=output_dir, backend='Vitis', io_type='io_parallel'
    )
    hls_model.compile()

    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    counts = np.asarray(hls_model.predict(x)).reshape(-1)
    np.testing.assert_array_equal(counts, expected)
    assert np.any(counts > 0)

    # Only the interface pragma at the top level (no reshaped ports, no top-level pipeline), and an integer count
    top_cpp = Path(output_dir, 'firmware', 'myproject.cpp').read_text()
    assert [line.split()[2] for line in top_cpp.splitlines() if '#pragma HLS' in line] == ['INTERFACE']
    assert 'typedef ap_uint<16> result_t;' in Path(output_dir, 'firmware', 'defines.h').read_text()

    # Saving leaves the model intact, so a second save also loads
    for _ in range(2):
        hls_model.save(Path(output_dir, 'saved.fml'))
    hls4ml.converters.load_saved_model(Path(output_dir, 'saved.fml'))

    # Rejected: a result type that cannot count the spikes, and an unsigned membrane
    for key, value, error in [('result', 'ap_uint<3>', 'cannot count'), ('membrane', 'ap_ufixed<32,8>', 'must be signed')]:
        bad_config = copy.deepcopy(config)
        bad_config['LayerName']['snn']['Precision'][key] = value
        with pytest.raises(Exception, match=error):
            hls4ml.converters.convert_from_pytorch_model(
                top, hls_config=bad_config, output_dir=output_dir, backend='Vitis', io_type='io_parallel'
            )
