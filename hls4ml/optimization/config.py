from enum import Enum

'''
A list of currently supported structures when optimizing (pruning, weight sharing)
For more information, see attributes.py

1. Unstructured:
    - Pruning: Y
    - Weight sharing: N
    - Description: Removes (zeroes out) individual weights
    - Supports: All layers in SUPPORTED_LAYERS (hls4ml.optimization.keras)

2. Structured:
    - Pruning: Y
    - Weight sharing: Y
    - Description: Zeroes out or quantizes all the weights in a structure:
        - Dense: Neurons, determined by their outgoing connections (columns in Keras weight tensors)
        - Conv2D: Filters (structures of size filt_width x filt_height x n_chan)
        - Notes:
            - For Dense, it was also possible optimize by incoming connections (rows);
                However, removing zero neurons becomes harder because of Keras Surgeon
            - For Conv2D, significant literature explored pruning channels; currently not supported
    - Supports: All layers in SUPPORTED_LAYERS (hls4ml.optimization.keras)

3. Pattern:
    - Pruning: Y
    - Weight sharing: Y
    - Description: Zeroes out or quantizes all the weights in a group
       Groups are determined by a variable, n, and every n-th weight in the flattened,
       Transposed (Resource) weight tensor is collected and stored in the same group
       Equivalent to pruning/quantizing weight processed by the same DSP in hls4ml
    - Supports: All layers in SUPPORTED_LAYERS (hls4ml.optimization.keras)

4. Block:
    - Pruning: Y
    - Weight sharing: Y
    - Description: Zeroes out or quantizes all the weights in a block of size (w, h)
    - Supports: All rank-2 (e.g. Dense, but not Conv2D) layers in SUPPORTED_LAYERS (hls4ml.optimization.keras)

'''


class SUPPORTED_STRUCTURES(Enum):
    UNSTRUCTURED = 'unstructured'
    STRUCTURED = 'structured'
    PATTERN = 'pattern'
    BLOCK = 'block'
