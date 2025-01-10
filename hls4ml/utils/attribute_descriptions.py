"""Strings holding attribute descriptions."""

# Common attributes

reuse_factor = (
    'The number of times each multiplier is used by controlling the amount of pipelining/unrolling. '
    'Lower number results in more parallelism and lower latency at the expense of the resources used.'
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
    '"latency" provides good latency at the expense of extra resources. performs well on small number of classes. '
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
    'convolution kernel occuring in parallel. '
    'Higher number results in more parallelism (lower latency and II) at the expense of resources used.'
    'Currently only supported in io_parallel.'
)
conv_implementation = (
    '"LineBuffer" implementation is preferred over "Encoded" for most use cases. '
    'This attribute only applies to io_stream.'
)

# Recurrent-related attributes

recurrent_static = (
    'If set to True, will reuse the the same recurrent block for computation, resulting in lower resource '
    'usage at the expense of serialized computation and higher latency/II.'
)
