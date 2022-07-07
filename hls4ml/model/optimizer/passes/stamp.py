from hls4ml.model.optimizer import ModelOptimizerPass


class MakeStamp(ModelOptimizerPass):
    def __init__(self):
        self.name = 'make_stamp'

    def transform(self, model):
        def _make_stamp():
            """ Create a unique identifier for the generated code. This identifier is used to
            compile a unique library and link it with python. """
            from string import hexdigits
            from random import choice
            length = 8
            return ''.join(choice(hexdigits) for m in range(length))

        model.config.config['Stamp'] = _make_stamp()

        return False  # No model graph changes made
