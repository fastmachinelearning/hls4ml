from __future__ import absolute_import

from .optimizer import OptimizerPass, register_pass, get_optimizer, optimize_model


from .passes.nop import EliminateLinearActivation
from .passes.bn_quant import MergeBatchNormAndQuantizedTanh, QuantizeDenseOutput

register_pass('eliminate_linear_activation', EliminateLinearActivation)
register_pass('merge_batch_norm_quantized_tanh', MergeBatchNormAndQuantizedTanh)
register_pass('quantize_dense_output', QuantizeDenseOutput)