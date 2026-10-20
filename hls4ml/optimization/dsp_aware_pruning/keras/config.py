import warnings

from tensorflow.keras.layers import Conv2D, Dense

"""
Optimizable layers in Keras / QKeras
QKeras layers are supported if QKeras (or QKeras v3 with Keras 3) is installed
Any new layers need to be registered here first
Additional logic in the source files may need to be written (e.g. recurrent layers should also optimize recurrent kernels)
"""
try:
    from qkeras import QConv2D, QDense

    QKERAS_AVAILABLE = True
    DENSE_LAYERS = (Dense, QDense)
    CONV2D_LAYERS = (Conv2D, QConv2D)
except ImportError as e:
    # The name is 'qkeras' if QKeras is not installed, or if a module of that name has no QDense and QConv2D (such as
    # the QKeras compatibility module of HGQ2). Any other name means QKeras is installed but failed to import.
    if e.name != 'qkeras':
        warnings.warn(
            f'QKeras is installed but could not be imported ({e}); QDense and QConv2D will not be optimized', stacklevel=2
        )
    QKERAS_AVAILABLE = False
    DENSE_LAYERS = (Dense,)
    CONV2D_LAYERS = (Conv2D,)

SUPPORTED_LAYERS = DENSE_LAYERS + CONV2D_LAYERS


"""
Supported ranking metrics, for classifying redundant (groups of) weights

1. l1 - groups of weights are ranked by their l1 norm
2. l2 - groups of weights are ranked by their l2 norm
3. oracle - abs(dL / dw * w), introduced by Molchanov et al. (2016)
    Pruning Convolutional Neural Networks for Resource Efficient Inference
4. saliency - (d^2L / dw^2 * w)^2, introduced by Lecun et al. (1989) Optimal Brain Damage
"""
SUPPORTED_METRICS = ('l1', 'l2', 'oracle', 'saliency')

"""
Temporary directory for storing best models, tuning results etc.
"""
TMP_DIRECTORY = 'hls4ml-optimization-keras'
