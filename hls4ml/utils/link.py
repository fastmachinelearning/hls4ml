import os
from pathlib import Path

import yaml

from hls4ml.model.graph import HLSConfig, ModelGraph


class FilesystemModelGraph(ModelGraph):
    """
    A subclass of ModelGraph that can link with an existing project in the filesystem.

    This allows the user to call `compile()`, `predict()` and `build()` functions.
    All other methods are disabled and will raise an exception if accessed.
    """

    def __init__(self, project_dir: str | Path):
        """Create a FilesystemModelGraph that links to an existing project previously written with `ModelGraph.write()`.

        Args:
            project_dir (str ): _description_

        Raises:
            Exception: _description_
        """
        if isinstance(project_dir, str):
            project_dir = Path(project_dir)
        if not os.path.exists(project_dir / 'hls4ml_config.yml'):
            raise Exception(f'Cannot find hls4ml_config.yml in the directory {project_dir}.')

        self._allowed_methods = {'compile', 'predict', 'build', 'get_input_variables', 'get_output_variables'}

        yaml.add_multi_constructor('!keras_model', lambda loader, suffix, node: None, Loader=yaml.SafeLoader)
        config = yaml.safe_load(open(project_dir / 'hls4ml_config.yml'))

        self.in_vars = []
        self.out_vars = []
        for var_name, var_shape in config['InputShapes'].items():
            var = self.VariableWrapper(var_name, var_shape)
            self.in_vars.append(var)
        for var_name, var_shape in config['OutputShapes'].items():
            var = self.VariableWrapper(var_name, var_shape)
            self.out_vars.append(var)

        self.config = HLSConfig(config)
        self._top_function_lib = None

    def __getattribute__(self, name):
        # Allow access to private attributes and explicitly allowed methods
        if name.startswith('_') or name in object.__getattribute__(self, '_allowed_methods'):
            return object.__getattribute__(self, name)
        # Raise an exception for all other methods
        if name in dir(ModelGraph):
            raise Exception(f'The method "{name}" should not be invoked on FilesystemModelGraph.')
        return object.__getattribute__(self, name)

    def get_input_variables(self):
        return self.in_vars

    def get_output_variables(self):
        return self.out_vars

    def compile(self):
        return super()._compile()

    def predict(self, x):
        return super().predict(x)

    def build(self, **kwargs):
        return self.config.backend.build(self, **kwargs)

    class VariableWrapper:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

        def size(self):
            nelem = 1
            for dim in self.shape:
                nelem *= dim
            return nelem
