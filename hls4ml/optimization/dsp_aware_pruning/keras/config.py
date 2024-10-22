from qkeras import QConv2D, QDense
from tensorflow.keras.layers import Conv2D, Dense

'''
Optimizable layers in Keras / QKeras
Any new layers need to be registered here first
Additional logic in the source files may need to be written (e.g. recurrent layers should also optimize recurrent kernels)
'''
SUPPORTED_LAYERS = (Dense, Conv2D, QDense, QConv2D)


'''
Supported ranking metrics, for classifying redundant (groups of) weights

1. l1 - groups of weights are ranked by their l1 norm
2. l2 - groups of weights are ranked by their l2 norm
3. oracle - abs(dL / dw * w), introduced by Molchanov et al. (2016)
    Pruning Convolutional Neural Networks for Resource Efficient Inference
4. saliency - (d^2L / dw^2 * w)^2, introduced by Lecun et al. (1989) Optimal Brain Damage
'''
SUPPORTED_METRICS = ('l1', 'l2', 'oracle', 'saliency')

'''
Temporary directory for storing best models, tuning results etc.
'''
TMP_DIRECTORY = 'hls4ml-optimization-keras'
