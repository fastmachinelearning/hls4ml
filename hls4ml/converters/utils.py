import math


def parse_data_format(input_shape, data_format='channels_last'):
    """Parses the given input shape according to the specified data format.

    This function can be used to ensure the shapes of convolutional and pooling layers is correctly parsed. If the first
    element of the given ``input_shape`` is ``None`` it is interpreted as a batch dimension and discarded.The returned tuple
    will have the channels dimension last.

    Args:
        input_shape (list or tuple): Input shape of 2D or 3D tensor with optional batch dimension of ``None``.
        data_format (str, optional): Data format type, one of ``channels_first`` or ``channels_last``. (case insensitive).
            Defaults to 'channels_last'.

    Raises:
        Exception: Raised if the data format type is unknown.

    Returns:
        tuple: The input shape (without the batch dimension) in ``channels_last`` format.
    """
    if input_shape[0] is None:
        # Ignore batch size
        input_shape = input_shape[1:]

    if data_format.lower() == 'channels_last':
        if len(input_shape) == 2:  # 1D, (n_in, n_filt)
            return (input_shape[0], input_shape[1])
        elif len(input_shape) == 3:  # 2D, (in_height, in_width, n_filt)
            return (input_shape[0], input_shape[1], input_shape[2])

    elif data_format.lower() == 'channels_first':
        if len(input_shape) == 2:  # 1D, (n_filt, n_in)
            return (input_shape[1], input_shape[0])
        elif len(input_shape) == 3:  # 2D, (n_filt, in_height, in_width)
            return (input_shape[1], input_shape[2], input_shape[0])
    else:
        raise Exception(f'Unknown data format: {data_format}')


def compute_padding_1d(pad_type, in_size, stride, filt_size):
    """Computes the amount of padding required on each side of the 1D input tensor.

    In case of ``same`` padding, this routine tries to pad evenly left and right, but if the amount of columns to be added
    is odd, it will add the extra column to the right.

    Args:
        pad_type (str): Padding type, one of ``same``, `valid`` or ``causal`` (case insensitive).
        in_size (int): Input size.
        stride (int): Stride length.
        filt_size (int): Length of the kernel window.

    Raises:
        Exception: Raised if the padding type is unknown.

    Returns:
        tuple: Tuple containing the padded input size, left and right padding values.
    """
    if pad_type.lower() == 'same':
        n_out = int(math.ceil(float(in_size) / float(stride)))
        if in_size % stride == 0:
            pad_along_size = max(filt_size - stride, 0)
        else:
            pad_along_size = max(filt_size - (in_size % stride), 0)
        pad_left = pad_along_size // 2
        pad_right = pad_along_size - pad_left
    elif pad_type.lower() == 'valid':
        n_out = int(math.ceil(float(in_size - filt_size + 1) / float(stride)))
        pad_left = 0
        pad_right = 0
    elif pad_type.lower() == 'causal':
        n_out = int(math.ceil(float(in_size) / float(stride)))
        if in_size % stride == 0:
            pad_along_size = max(filt_size - stride, 0)
        else:
            pad_along_size = max(filt_size - (in_size % stride), 0)
        pad_left = pad_along_size
        pad_right = 0
    else:
        raise Exception(f'Unknown padding type: {pad_type}')

    return (n_out, pad_left, pad_right)


def compute_padding_2d(pad_type, in_height, in_width, stride_height, stride_width, filt_height, filt_width):
    """Computes the amount of padding required on each side of the 2D input tensor.

    In case of ``same`` padding, this routine tries to pad evenly left and right (top and bottom), but if the amount of
    columns to be added is odd, it will add the extra column to the right/bottom.

    Args:
        pad_type (str): Padding type, one of ``same`` or ``valid`` (case insensitive).
        in_height (int): The height of the input tensor.
        in_width (int): The width of the input tensor.
        stride_height (int): Stride height.
        stride_width (int): Stride width.
        filt_height (int): Height of the kernel window.
        filt_width (int): Width of the kernel window.

    Raises:
        Exception: Raised if the padding type is unknown.

    Returns:
        tuple: Tuple containing the padded input height, width, and top, bottom, left and right padding values.
    """
    if pad_type.lower() == 'same':
        # Height
        out_height = int(math.ceil(float(in_height) / float(stride_height)))
        if in_height % stride_height == 0:
            pad_along_height = max(filt_height - stride_height, 0)
        else:
            pad_along_height = max(filt_height - (in_height % stride_height), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        # Width
        out_width = int(math.ceil(float(in_width) / float(stride_width)))
        if in_width % stride_width == 0:
            pad_along_width = max(filt_width - stride_width, 0)
        else:
            pad_along_width = max(filt_width - (in_width % stride_width), 0)
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
    elif pad_type.lower() == 'valid':
        out_height = int(math.ceil(float(in_height - filt_height + 1) / float(stride_height)))
        out_width = int(math.ceil(float(in_width - filt_width + 1) / float(stride_width)))

        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
    else:
        raise Exception(f'Unknown padding type: {pad_type}')

    return (out_height, out_width, pad_top, pad_bottom, pad_left, pad_right)


def compute_padding_1d_pytorch(pad_type, in_size, stride, filt_size, dilation):
    if isinstance(pad_type, str):
        if pad_type.lower() == 'same':
            n_out = int(
                math.floor((float(in_size) + 2 - float(dilation) * (float(filt_size) - 1) - 1) / float(stride) + 1)
            )  # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            if in_size % stride == 0:
                pad_along_size = max(filt_size - stride, 0)
            else:
                pad_along_size = max(filt_size - (in_size % stride), 0)
            pad_right = pad_along_size // 2
            pad_left = pad_along_size - pad_right
        elif pad_type.lower() == 'valid':
            n_out = int(
                math.floor((float(in_size) - float(dilation) * (float(filt_size) - 1) - 1) / float(stride) + 1)
            )  # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            pad_left = 0
            pad_right = 0
        else:
            raise Exception(f'Unknown padding type: {pad_type}')
    else:
        if pad_type > 0:
            n_out = int(
                math.floor(
                    (float(in_size) + 2 * pad_type - float(dilation) * (float(filt_size) - 1) - 1) / float(stride) + 1
                )
            )  # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            pad_right = pad_type
            pad_left = pad_type
        else:
            n_out = int(
                math.floor((float(in_size) - float(dilation) * (float(filt_size) - 1) - 1) / float(stride) + 1)
            )  # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            pad_left = 0
            pad_right = 0

    return (n_out, pad_left, pad_right)


def compute_padding_2d_pytorch(
    pad_type, in_height, in_width, stride_height, stride_width, filt_height, filt_width, dilation_height, dilation_width
):
    if isinstance(pad_type, str):
        if pad_type.lower() == 'same':
            # Height
            out_height = int(
                math.floor(float(in_height + 2 - dilation_height * (filt_height - 1) - 1) / float(stride_height) + 1)
            )
            if in_height % stride_height == 0:
                pad_along_height = max(filt_height - stride_height, 0)
            else:
                pad_along_height = max(filt_height - (in_height % stride_height), 0)
            pad_bottom = pad_along_height // 2
            pad_top = pad_along_height - pad_bottom
            pad_top = 1
            # Width
            out_width = int(
                math.floor(float(in_width + 2 - dilation_width * (filt_width - 1) - 1) / float(stride_width) + 1)
            )
            if in_width % stride_width == 0:
                pad_along_width = max(filt_width - stride_width, 0)
            else:
                pad_along_width = max(filt_width - (in_width % stride_width), 0)
            pad_right = pad_along_width // 2
            pad_left = pad_along_width - pad_right
        elif pad_type.lower() == 'valid':
            out_height = int(
                math.floor(float(in_height - dilation_height * (filt_height - 1) - 1) / float(stride_height) + 1)
            )
            out_width = int(math.floor(float(in_width - dilation_width * (filt_width - 1) - 1) / float(stride_width) + 1))

            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0
        else:
            raise Exception(f'Unknown padding type: {pad_type}')

    else:
        if pad_type[0] == 0 and pad_type[1] == 0:
            out_height = int(
                math.floor(float(in_height - dilation_height * (filt_height - 1) - 1) / float(stride_height) + 1)
            )
            out_width = int(math.floor(float(in_width - dilation_width * (filt_width - 1) - 1) / float(stride_width) + 1))

            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0

        else:
            # Height
            pad_height = pad_type[0]
            pad_width = pad_type[1]
            out_height = int(
                math.floor(
                    float(in_height + 2 * pad_height - dilation_height * (filt_height - 1) - 1) / float(stride_height) + 1
                )
            )
            pad_bottom = pad_height
            pad_top = pad_height
            # Width
            out_width = int(
                math.floor(float(in_width + 2 * pad_width - dilation_width * (filt_width - 1) - 1) / float(stride_width) + 1)
            )
            pad_right = pad_width
            pad_left = pad_width

    return (out_height, out_width, pad_top, pad_bottom, pad_left, pad_right)
