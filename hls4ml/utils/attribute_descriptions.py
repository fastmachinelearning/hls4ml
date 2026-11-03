"""Strings holding attribute descriptions."""

# Common attributes

reuse_factor = (
    'The number of times each multiplier is used by controlling the amount of pipelining/unrolling. '
    'A lower number results in more parallelism and lower latency at the expense of the resources used. '
    'Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.'
)

index = 'Internal node counter used for bookkeeping and variable/tensor naming.'
trace = 'Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)'

result_type = 'The datatype (precision) of the output tensor.'
accum_type = 'The datatype (precision) used to store intermediate results of the computation within the layer.'

# Activation-related attributes

table_size = 'The size of the lookup table used to approximate the function.'
table_type = 'The datatype (precision) used for the values of the lookup table.'

softmax_implementation = (
    'Choice of implementation of softmax function. '
    '"latency" provides good latency at the expense of extra resources. It performs well with a small number of classes. '
    '"stable" may require extra clock cycles but has better accuracy. '
    '"legacy" is the older implementation which has bad accuracy, but is fast and has low resource use. '
    'It is superseded by the "latency" implementation for most applications. '
    '"argmax" is a special implementation that can be used if only the output with the highest probability is important. '
    'Using this implementation will save resources and clock cycles.'
)
softmax_skip = 'If enabled, skips the softmax node and returns the raw outputs.'

# Convolution-related attributes

conv_pf = (
    'The number of outputs computed in parallel. Essentially the number of multiplications of input window with the '
    'convolution kernel occurring in parallel. '
    'A higher number results in more parallelism (lower latency and II) at the expense of resources used. '
    'Currently only supported in io_parallel.'
)
conv_implementation = (
    '"LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.'
)
conv_transpose_slot_groups = (
    'Set by the backend. The output is computed in blocks of stride positions, one block for each input position. For '
    'each kernel element, in the order of the rows of the weight matrix, the position in the block that it adds to.'
)
conv_transpose_slot_shifts = (
    'Set by the backend. For each kernel element, in the order of the rows of the weight matrix, how many blocks '
    'before the current block its input is, as (height, width).'
)
conv_transpose_n_buf = (
    'Set by the backend. The number of inputs multiplied for each block of outputs, which is the number of kernel '
    'elements times the number of input channels.'
)
conv_transpose_first_block = (
    'Set by the backend. The index of the first block of outputs along this axis that cropping does not remove entirely.'
)
conv_transpose_n_blocks = (
    'Set by the backend. The number of blocks of outputs along this axis that cropping does not remove entirely.'
)

# Recurrent-related attributes

recurrent_static = (
    'If set to True, will reuse the same recurrent block for computation, resulting in lower resource '
    'usage at the expense of serialized computation and higher latency/II. '
    'Only applies to io_parallel, as io_stream always reuses the same block. '
    'On Vivado and Vitis, the Resource strategy also always reuses the same block.'
)

# LayerNorm-related attributes

table_range_power2 = (
    'The negative power of 2 that represents the range of the lookup table, '
    'e.g. a value of 1 would represent a range of 0.5.'
)

time_distributed_loop = (
    'Controls the amount and type of parallelism in the loop over time steps. If set to "off", no parallelism will be used. '
    'If set to "unroll", the loop will be unrolled. This may result in excessive resource use and cannot be used in '
    '"io_stream" mode. If set to "pipeline", the loop will be pipelined.'
)
