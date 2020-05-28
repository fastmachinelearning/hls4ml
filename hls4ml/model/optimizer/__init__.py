from __future__ import absolute_import

from .optimizer import OptimizerPass, register_pass, get_optimizer, optimize_model


from .passes.nop import EliminateLinearActivation
from .passes.bn_quant import MergeBatchNormAndQuantizedTanh, QuantizeDenseOutput
from .passes.dense_bn_fuse import FuseDenseAndBatchNormalization
from .passes.fuse_biasadd import FuseBiasAdd
try:
    from .passes.qkeras import OutputRoundingSaturationMode
    from .passes.qkeras import QKerasFactorizeAlpha
    from .passes.qkeras import FuseConsecutiveBatchNormalization
    __qkeras_optimizers__ = True
except ImportError:
    __qkeras_optimizers__ = False

register_pass('eliminate_linear_activation', EliminateLinearActivation)
register_pass('merge_batch_norm_quantized_tanh', MergeBatchNormAndQuantizedTanh)
register_pass('quantize_dense_output', QuantizeDenseOutput)
register_pass('fuse_dense_batch_norm', FuseDenseAndBatchNormalization)
register_pass('fuse_biasadd', FuseBiasAdd)
if __qkeras_optimizers__:
    register_pass('output_rounding_saturation_mode', OutputRoundingSaturationMode)
    register_pass('qkeras_factorize_alpha', QKerasFactorizeAlpha)
    register_pass('fuse_consecutive_batch_normalization', FuseConsecutiveBatchNormalization) 

