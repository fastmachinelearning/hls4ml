from __future__ import absolute_import

from hls4ml.model.optimizer.optimizer import OptimizerPass, register_pass, get_optimizer, optimize_model, get_available_passes


from hls4ml.model.optimizer.passes.nop import EliminateLinearActivation, EliminateLinearActivationQuant
from hls4ml.model.optimizer.passes.bn_quant import MergeBatchNormAndQuantizedTanh
from hls4ml.model.optimizer.passes.bn_quant import QuantizeDenseOutput
from hls4ml.model.optimizer.passes.bn_fuse import FuseBatchNormalization
from hls4ml.model.optimizer.passes.fuse_biasadd import FuseBiasAdd
from hls4ml.model.optimizer.passes.conv_same_pad import InsertZeroPaddingBeforeConv1D
from hls4ml.model.optimizer.passes.conv_same_pad import InsertZeroPaddingBeforeConv2D
from hls4ml.model.optimizer.passes.pointwise import OptimizePointwiseConv
from hls4ml.model.optimizer.passes.clone import CloneOutput
from hls4ml.model.optimizer.passes.repack_stream import ReshapeStream, BroadcastStream, RemoveFinalReshape
from hls4ml.model.optimizer.passes.transpose_opt import RemoveUselessTranspose
from hls4ml.model.optimizer.passes.multi_dense import ReplaceMultidimensionalDenseWithConv
from hls4ml.model.optimizer.passes.reshape_const import ReshapeConstant
from hls4ml.model.optimizer.passes.quant_opt import (
    QuantConstantParameters, QuantToActivation, FuseQuantWithConstant, QuantToAlphaActivationAlpha, ConstQuantToConstAlpha)
from hls4ml.model.optimizer.passes.batchnorm_opt import BatchNormConstantParameters, ConstantBatchNormMerging, FuseConsecutiveBatchNormalization
from hls4ml.model.optimizer.passes.merge_const import MergeTwoConstant, MergeToBatchNormalization, MergeToBatchNormalizationDiv
from hls4ml.model.optimizer.passes.matmul_const_to_dense import MatmulConstToDense
from hls4ml.model.optimizer.passes.conv_to_convxd import ConvToConvXD
from hls4ml.model.optimizer.passes.reshape_nop import ReshapeToNop

try:
    from hls4ml.model.optimizer.passes.qkeras import OutputRoundingSaturationMode
    from hls4ml.model.optimizer.passes.qkeras import QKerasFactorizeAlpha
    from hls4ml.model.optimizer.passes.qkeras import FuseConsecutiveBatchNormalization
    from hls4ml.model.optimizer.passes.qkeras import ExtractTernaryThreshold
    __qkeras_optimizers__ = True
except ImportError:
    __qkeras_optimizers__ = False

if __qkeras_optimizers__:
    register_pass('output_rounding_saturation_mode', OutputRoundingSaturationMode)
    register_pass('qkeras_factorize_alpha', QKerasFactorizeAlpha)
    register_pass('extract_ternary_threshold', ExtractTernaryThreshold)
    # register_pass('fuse_consecutive_batch_normalization', FuseConsecutiveBatchNormalization)

register_pass('reshape_constant', ReshapeConstant)
register_pass('quant_constant_params', QuantConstantParameters)
register_pass('quant_to_activation', QuantToActivation)
register_pass('fuse_quant_with_constant', FuseQuantWithConstant)
register_pass('quant_to_alph_activation_alpha', QuantToAlphaActivationAlpha)
register_pass('const_quant_to_const_alpha', ConstQuantToConstAlpha)
register_pass('batch_norm_constant_parameters', BatchNormConstantParameters)
register_pass('fuse_consecutive_base_batch_normalizations', FuseConsecutiveBatchNormalization)
register_pass('constant_batch_norm_fusion', ConstantBatchNormMerging)
register_pass('merge_two_constants', MergeTwoConstant)
register_pass('merge_to_batchnorm', MergeToBatchNormalization)
register_pass('merge_to_batchnorm_div', MergeToBatchNormalizationDiv)
register_pass('matmul_const_to_dense', MatmulConstToDense)
register_pass('conv_to_convxd', ConvToConvXD)
register_pass('eliminate_linear_activation', EliminateLinearActivation)
register_pass('merge_batch_norm_quantized_tanh', MergeBatchNormAndQuantizedTanh)
register_pass('quantize_dense_output', QuantizeDenseOutput)
register_pass('fuse_batch_norm', FuseBatchNormalization)
register_pass('fuse_biasadd', FuseBiasAdd)
register_pass('conv1d_same_pad', InsertZeroPaddingBeforeConv1D)
register_pass('conv2d_same_pad', InsertZeroPaddingBeforeConv2D)
register_pass('optimize_pointwise_conv', OptimizePointwiseConv)
register_pass('clone_output', CloneOutput)
register_pass('remove_final_reshape', RemoveFinalReshape)
register_pass('reshape_to_nop', ReshapeToNop)
register_pass('reshape_stream', ReshapeStream)
register_pass('remove_useless_transpose', RemoveUselessTranspose)
register_pass('replace_multidense_conv', ReplaceMultidimensionalDenseWithConv)
register_pass('broadcast_stream', BroadcastStream)
register_pass('eliminate_linear_activation_quant', EliminateLinearActivationQuant)
