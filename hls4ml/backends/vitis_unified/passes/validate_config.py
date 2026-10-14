from hls4ml.backends.vitis_unified.vitis_unified_validation import validate_config
from hls4ml.model.optimizer import ModelOptimizerPass


class ValidateConfig(ModelOptimizerPass):
    def __init__(self):
        pass

    def transform(self, model):
        cfg = model.config.get_config_value('VitisUnifiedConfig')
        validate_config(
            cfg['Board'],
            cfg['axi_mode'],
            cfg['Driver'],
            cfg['InputDtype'],
            cfg['OutputDtype'],
            platform=cfg.get('Platform'),
            part=model.config.get_config_value('Part'),
        )

        n_inputs = len(model.get_input_variables())
        n_outputs = len(model.get_output_variables())
        if n_inputs < 1 or n_outputs < 1:
            raise Exception(
                'VitisUnified requires a model with at least one input and one output tensor '
                f'(got {n_inputs} inputs, {n_outputs} outputs).'
            )
        if cfg['axi_mode'] == 'axi_stream' and (n_inputs > 1 or n_outputs > 1):
            raise Exception(
                'axi_mode "axi_stream" only supports models with exactly one input and one output tensor '
                f'(got {n_inputs} inputs, {n_outputs} outputs). '
                'Use axi_mode "axi_master" for multi-input/multi-output models.'
            )
        return False
