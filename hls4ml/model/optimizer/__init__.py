from __future__ import absolute_import

from hls4ml.model.optimizer.optimizer import OptimizerPass, GlobalOptimizerPass, register_pass, get_optimizer, optimize_model, get_available_passes, optimizer_pass, extract_optimizers_from_object


from hls4ml.model.optimizer.passes.nop import EliminateLinearActivation
from hls4ml.model.optimizer.passes.bn_fuse import FuseBatchNormalization
from hls4ml.model.optimizer.passes.fuse_biasadd import FuseBiasAdd

try:
    from hls4ml.model.optimizer.passes.qkeras import OutputRoundingSaturationMode
    from hls4ml.model.optimizer.passes.qkeras import QKerasFactorizeAlpha
    from hls4ml.model.optimizer.passes.qkeras import FuseConsecutiveBatchNormalization
    __qkeras_optimizers__ = True
except ImportError:
    __qkeras_optimizers__ = False

register_pass('eliminate_linear_activation', EliminateLinearActivation)
register_pass('fuse_batch_norm', FuseBatchNormalization)
register_pass('fuse_biasadd', FuseBiasAdd)

if __qkeras_optimizers__:
    register_pass('output_rounding_saturation_mode', OutputRoundingSaturationMode)
    register_pass('qkeras_factorize_alpha', QKerasFactorizeAlpha)
    register_pass('fuse_consecutive_batch_normalization', FuseConsecutiveBatchNormalization) 

