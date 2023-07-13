import os

from hls4ml.model.flow.flow import register_flow
from hls4ml.model.optimizer.optimizer import (  # noqa: F401
    ConfigurableOptimizerPass,
    GlobalOptimizerPass,
    LayerOptimizerPass,
    ModelOptimizerPass,
    OptimizerPass,
    extract_optimizers_from_object,
    extract_optimizers_from_path,
    get_available_passes,
    get_backend_passes,
    get_optimizer,
    layer_optimizer,
    model_optimizer,
    optimize_model,
    optimizer_pass,
    register_pass,
)

opt_path = os.path.dirname(__file__) + '/passes'
module_path = __name__ + '.passes'

optimizers = extract_optimizers_from_path(opt_path, module_path)
for opt_name, opt in optimizers.items():
    register_pass(opt_name, opt)

del opt_path
del module_path
del optimizers

register_flow(
    'convert',
    [
        'channels_last_converter',
        'fuse_bias_add',
        'remove_useless_transpose',
        'expand_layer_group',
        'reshape_constant',
        'quant_constant_parameters',
        'quant_to_activation',
        'fuse_quant_with_constant',
        'quant_to_alpha_activation_alpha',
        'const_quant_to_const_alpha',
        'batch_norm_onnx_constant_parameters',
        'constant_batch_norm_fusion',
        'merge_two_constants',
        'scale_down_add',
        'scale_down_mat_mul',
        'scale_down_weight_conv',
        'scale_down_bias_conv',
        'scale_down_conv',
        'merge_to_apply_alpha',
        'merge_to_apply_alpha_div',
        'matmul_const_to_dense',
        'conv_to_conv_x_d',
        'output_rounding_saturation_mode',
        'qkeras_factorize_alpha',
        'extract_ternary_threshold',
        'fuse_consecutive_batch_normalization',
    ],
)  # TODO Maybe not all QKeras optmizers belong here?

register_flow(
    'optimize',
    [
        'eliminate_linear_activation',
        'fuse_consecutive_batch_normalization',
        'fuse_batch_normalization',
        'replace_multidimensional_dense_with_conv',
        'propagate_dense_precision',
        'propagate_conv_precision',
        'set_precision_concat',
        'merge_linear_activation',
    ],
    requires=['convert'],
)
