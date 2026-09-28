from hls4ml.model.optimizer import ModelOptimizerPass


class ValidateBramWeights(ModelOptimizerPass):
    def __init__(self):
        pass

    def transform(self, model):
        bram_weights = [var.name for var in model.get_weight_variables() if var.storage.lower() == 'bram']
        if bram_weights:
            raise Exception(
                f'BramFactor weights ({", ".join(bram_weights)}) are not supported by the VitisUnified backend. '
                'Raise BramFactor above the largest weight size, or use the Vitis backend.'
            )
        return False
