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
        'set_precision_concat',
    ],
    requires=['convert'],
)
