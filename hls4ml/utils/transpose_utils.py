import math

import numpy as np


def transpose_config_gen(name: str, shape: tuple[int, ...], perm: tuple[int, ...]):
    """
    Generate new shape and perm_strides for a permute operation. Operates by mapping the output index
    to input input index by:
    - unravel the output index
    - map each dimension to the corresponding stride in the input tensor, sum
    The operation can be expressed as:

    new_shape = tuple(shape[i] for i in perm)
    strides = np.cumprod((shapes[1:] + (1,))[::-1])[::-1]
    perm_strides = [strides[i] for i in perm]
    out[index] = inp[np.dot(np.unravel_index(index, new_shape), perm_strides)]

    Args:
        name (str): The name of the configuration.
        shape (tuple[int, ...]): The shape of the input tensor.
        perm (tuple[int, ...]): The permutation of the dimensions.

    Returns:
        dict: Dictionary containing the configuration.
    """
    new_shape = tuple(shape[i] for i in perm)
    strides = np.cumprod((shape[1:] + (1,))[::-1])[::-1]
    perm_strides = tuple(int(strides[i]) for i in perm)
    return dict(
        dims=len(shape),
        N=math.prod(shape),
        from_shape=', '.join(str(x) for x in shape),
        perm=', '.join(str(x) for x in perm),
        perm_strides=', '.join(str(x) for x in perm_strides),
        to_shape=', '.join(str(x) for x in new_shape),
        config_name=name,
    )
