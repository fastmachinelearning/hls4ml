from hls4ml.model.layers import Conv1D, Conv2D
from hls4ml.model.optimizer import ModelOptimizerPass


class SetPipelineStyle(ModelOptimizerPass):
    def __init__(self):
        pass

    def transform(self, model):
        if model.config.pipeline_style not in ['auto', 'pipeline', 'dataflow']:
            print(
                f'WARNING: Pipeline style set to {model.config.pipeline_style}, valid values: auto, pipeline, dataflow. '
                'Using "auto".'
            )
            self._set_pipeline_style(model, 'auto')

        if model.config.pipeline_style is None or model.config.pipeline_style == 'auto':

            if self._maybe_set_dataflow_io_stream(model):
                return True

            if self._maybe_set_dataflow_conv_layers(model):
                return True

            if self._maybe_set_dataflow_resource_strategy(model):
                return True

            if self._maybe_set_pipeline_resource_unrolled_strategy(model):
                return True

            if self._maybe_set_pipeline_io_parallel(model):
                return True

            self._set_safe_default_dataflow(model)
            return True
        else:
            self._validate_hls_config(model)

        return False  # No model changes made

    def _set_pipeline_style(self, model, pipeline_style):
        # Could add logging here
        model.config.pipeline_style = pipeline_style

    def _maybe_set_dataflow_io_stream(self, model):
        if model.config.get_config_value('IOType') == 'io_stream':
            self._set_pipeline_style(model, 'dataflow')
            return True

        return False

    def _maybe_set_dataflow_conv_layers(self, model):
        for layer in model.get_layers():
            if isinstance(layer, (Conv1D, Conv2D)):
                self._set_pipeline_style(model, 'dataflow')
                return True

        return False

    def _maybe_set_dataflow_resource_strategy(self, model):
        for layer in model.get_layers():
            if model.config.is_resource_strategy(layer):
                self._set_pipeline_style(model, 'dataflow')
                return True

        return False

    def _maybe_set_pipeline_resource_unrolled_strategy(self, model):
        have_unrolled = False
        for layer in model.get_layers():
            if model.config.get_strategy(layer).lower() == 'resource_unrolled':
                self._set_pipeline_style(model, 'pipeline')
                have_unrolled = True
                break

        if have_unrolled:
            model.config.pipeline_ii = max([int(layer.get_attr('reuse_factor')) for layer in model.get_layers()])

        return have_unrolled

    def _maybe_set_pipeline_io_parallel(self, model):
        if model.config.get_config_value('IOType') == 'io_parallel':
            self._set_pipeline_style(model, 'pipeline')
            return True

        return False

    def _set_safe_default_dataflow(self, model):
        print(
            'WARNING: Couldn\'t determine best pipeline style, defaulting to "DATAFLOW". '
            'Use "PipelineStyle" property to override.'
        )
        self._set_pipeline_style(model, 'dataflow')

    def _validate_hls_config(self, model):
        if model.config.pipeline_style.lower() == 'pipeline':
            if model.config.model_compression:
                print('WARNING: Compression enabled while pipeline style set to "pipeline".')
            if model.config.model_strategy.lower() == 'resource':
                print(
                    'WARNING: Model strategy "Resource" will lead to bad QoR in combination '
                    'with pipeline style set to "pipeline".'
                )
            if any(isinstance(layer, (Conv1D, Conv2D)) for layer in model.get_layers()):
                print('WARNING: Convolution layers require "dataflow" pipeline style.')
        for layer_type, strategy in model.config.layer_type_strategy.items():
            if strategy.lower() == 'resource' and model.config.pipeline_style.lower() == 'pipeline':
                print(
                    f'WARNING: Strategy for layer type {layer_type} set to "Resource", while pipeline style set to '
                    '"pipeline". This will lead to bad QoR.'
                )

        for layer_name, strategy in model.config.layer_name_strategy.items():
            if strategy.lower() == 'resource' and model.config.pipeline_style.lower() == 'pipeline':
                print(
                    'WARNING: Strategy for layer {} set to "Resource", while pipeline style set to "pipeline".'.format(
                        layer_name
                    )
                )

        for layer_type, compression in model.config.layer_type_compression.items():
            if compression and model.config.pipeline_style.lower() == 'pipeline':
                print(
                    'WARNING: Compression enabled for layer type {}, while pipeline style set to "pipeline".'.format(
                        layer_type
                    )
                )

        for layer_name, compression in model.config.layer_name_compression.items():
            if compression and model.config.pipeline_style.lower() == 'pipeline':
                print(f'WARNING: Compression enabled for layer {layer_name}, while pipeline style set to "pipeline".')
