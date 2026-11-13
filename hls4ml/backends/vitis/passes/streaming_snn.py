import numpy as np

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Input, StreamingSNN
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, Source

streaming_snn_config_template = """struct config{index} {{
    static const unsigned n_steps = {n_steps};
    typedef {weight_t.name} weight_t;
    typedef {bias_t.name} bias_t;
    typedef {accum_t.name} accum_t;
    typedef {membrane_t.name} membrane_t;
}};\n"""

streaming_snn_function_template = '{namespace}::streaming_snn_{index}<{input_t}, {output_t}, {config}>({input}, {output});'

# The block follows the validated SpikePID firmware (arXiv:2611.00104) statement by statement: its casts, statement
# order and pragmas set the results and the schedule (one time step per clock cycle).

block_template = """
template <class data_T, class res_T, typename CONFIG_T>
void streaming_snn_{index}(data_T x[CONFIG_T::n_steps], res_T count[1]) {{
    #pragma HLS INLINE
    typedef data_T input_t;
    typedef res_T count_t;
    typedef typename CONFIG_T::weight_t weight_t;
    typedef typename CONFIG_T::bias_t bias_t;
    typedef typename CONFIG_T::accum_t acc_t;
    typedef typename CONFIG_T::membrane_t mem_t;
{weights}
{constants}
{state}
    mem_t mo=0; ap_uint<1> po=0;
{init}
    count_t total=0;
    TICKS: for (int t=0;t<{n_steps};t++) {{
        #pragma HLS PIPELINE II=1
{body}
    }}
    count[0]=total;
}}
"""

# LIF update of neuron {n} of stage {i}: subtractive reset, one time step after the spike
lif_template = """            mem_t rst = p{i}[{n}] ? THR{i} : (mem_t)0;
            mem_t v = BETA{i}*m{i}[{n}] + (mem_t)a - rst;
            ap_uint<1> sp = (v>=THR{i})?1:0; m{i}[{n}]=v; p{i}[{n}]=sp; s{i}[{n}]=sp;"""

output_lif = """        mem_t rst = po ? THRo : (mem_t)0;
        mem_t v = BETAo*mo + (mem_t)a - rst;
        ap_uint<1> sp = (v>=THRo)?1:0; mo=v; po=sp;
        total += sp;"""

dense_input_template = """        input_t xt = x[t];
        ap_uint<1> s0[{n_out}];
        for (int h=0;h<{n_out};h++) {{
            acc_t a = (acc_t)fc0_b[h] + (acc_t)fc0_w[h]*(acc_t)xt;
{lif}
        }}"""

dense_template = """        ap_uint<1> s{i}[{n_out}];
        for (int h=0;h<{n_out};h++) {{
            acc_t a = (acc_t)fc{i}_b[h];
            for (int j=0;j<{n_in};j++) a += s{prev}[j] ? (acc_t)fc{i}_w[h][j] : (acc_t)0;
{lif}
        }}"""

dense_output_template = """        acc_t a = (acc_t)fcout_b;
        for (int j=0;j<{n_in};j++) a += s{prev}[j] ? (acc_t)fcout_w[j] : (acc_t)0;
{lif}"""

conv_input_template = """        for (int j=0;j<{k}-1;j++) xbuf[j]=xbuf[j+1];
        xbuf[{k}-1]=x[t];
        ap_uint<1> s0[{n_out}];
        for (int c=0;c<{n_out};c++) {{
            acc_t a=(acc_t)conv0_b[c];
            for (int j=0;j<{k};j++) a += (acc_t)conv0_w[c][j]*(acc_t)xbuf[j];
{lif}
        }}"""

conv_template = """        for (int j=0;j<{k}-1;j++) for (int c=0;c<{n_in};c++) {buf}[j][c]={buf}[j+1][c];
        for (int c=0;c<{n_in};c++) {buf}[{k}-1][c]=s{prev}[c];
        ap_uint<1> s{i}[{n_out}];
        for (int c=0;c<{n_out};c++) {{
            acc_t a=(acc_t)conv{i}_b[c];
            for (int cc=0;cc<{n_in};cc++) for (int j=0;j<{k};j++) a += {buf}[j][cc] ? (acc_t)conv{i}_w[c][cc][j] : (acc_t)0;
{lif}
        }}"""

conv_output_template = """        acc_t a=(acc_t)convout_b;
        for (int c=0;c<{n_in};c++) a += s{prev}[c] ? (acc_t)convout_w[c] : (acc_t)0;
{lif}"""


def _literal(value):
    # Exact, so that values on the grid of the type are not truncated one step down
    return repr(float(value))


def _array(values):
    values = np.asarray(values)
    if values.ndim == 1:
        return '{' + ', '.join(_literal(value) for value in values) + '}'
    return '{' + ', '.join(_array(row) for row in values) + '}'


def _dense_code(hidden, out):
    weights, state, init, body = [], [], [], []
    for i, stage in enumerate(hidden):
        n_in, n_out = stage['n_in'], stage['n_out']
        lif = lif_template.format(i=i, n='h')
        if i == 0:
            weights.append(f'    static const weight_t fc0_w[{n_out}] = {_array(stage["weight"][:, 0, 0])};')
            body.append(dense_input_template.format(n_out=n_out, lif=lif))
        else:
            weights.append(f'    static const weight_t fc{i}_w[{n_out}][{n_in}] = {_array(stage["weight"][:, :, 0])};')
            body.append(dense_template.format(i=i, prev=i - 1, n_in=n_in, n_out=n_out, lif=lif))
        weights.append(f'    static const bias_t fc{i}_b[{n_out}] = {_array(stage["bias"])};')
        state.append(f'    mem_t m{i}[{n_out}]; ap_uint<1> p{i}[{n_out}];')
        init.append(f'    for (int h=0;h<{n_out};h++) {{ m{i}[h]=0; p{i}[h]=0; }}')
    weights.append(f'    static const weight_t fcout_w[{out["n_in"]}] = {_array(out["weight"][0, :, 0])};')
    weights.append(f'    static const bias_t fcout_b = {_literal(out["bias"][0])};')
    body.append(dense_output_template.format(n_in=out['n_in'], prev=len(hidden) - 1, lif=output_lif))
    return weights, state, init, body


def _conv_code(hidden, out):
    weights, state, init, body = [], [], [], []
    for i, stage in enumerate(hidden):
        n_in, n_out, k = stage['n_in'], stage['n_out'], stage['kernel_size']
        lif = lif_template.format(i=i, n='c')
        if i == 0:
            weights.append(f'    static const weight_t conv0_w[{n_out}][{k}] = {_array(stage["weight"][:, 0, :])};')
            body.append(conv_input_template.format(k=k, n_out=n_out, lif=lif))
        else:
            weights.append(f'    static const weight_t conv{i}_w[{n_out}][{n_in}][{k}] = {_array(stage["weight"])};')
            body.append(conv_template.format(i=i, prev=i - 1, buf=f's{i - 1}buf', k=k, n_in=n_in, n_out=n_out, lif=lif))
        weights.append(f'    static const bias_t conv{i}_b[{n_out}] = {_array(stage["bias"])};')
        state.append(f'    mem_t m{i}[{n_out}]; ap_uint<1> p{i}[{n_out}];')
        init.append(f'    for (int c=0;c<{n_out};c++) {{ m{i}[c]=0; p{i}[c]=0; }}')
    # Line buffers: the last input samples for the first stage, the last spikes of the previous stage for the others
    k = hidden[0]['kernel_size']
    state.append(f'    input_t xbuf[{k}];')
    init.append(f'    for (int j=0;j<{k};j++) xbuf[j]=0;')
    for i, stage in enumerate(hidden[1:], start=1):
        k, n_in = stage['kernel_size'], stage['n_in']
        state.append(f'    ap_uint<1> s{i - 1}buf[{k}][{n_in}];')
        init.append(f'    for (int j=0;j<{k};j++) for (int c=0;c<{n_in};c++) s{i - 1}buf[j][c]=0;')
    weights.append(f'    static const weight_t convout_w[{out["n_in"]}] = {_array(out["weight"][0, :, 0])};')
    weights.append(f'    static const bias_t convout_b = {_literal(out["bias"][0])};')
    body.append(conv_output_template.format(n_in=out['n_in'], prev=len(hidden) - 1, lif=output_lif))
    return weights, state, init, body


def _streaming_snn_code(node):
    stages = node.get_attr('stages')
    hidden, out = stages[:-1], stages[-1]
    code = _dense_code if node.get_attr('family') == 'dense' else _conv_code
    weights, state, init, body = code(hidden, out)
    constants = []
    for name, stage in zip([*range(len(hidden)), 'o'], stages):
        constants.append(f'    const mem_t BETA{name} = (mem_t){_literal(stage["beta"])};')
        constants.append(f'    const mem_t THR{name}  = (mem_t){_literal(stage["threshold"])};')
    return block_template.format(
        index=node.index,
        n_steps=node.get_attr('n_steps'),
        weights='\n'.join(weights),
        constants='\n'.join(constants),
        state='\n'.join(state),
        init='\n'.join(init),
        body='\n'.join(body),
    )


def _check_types(node):
    # The membrane can go negative, and beta and threshold are constants of its type: neither may wrap or truncate to 0
    membrane = node.get_attr('membrane_t').precision
    if isinstance(membrane, (FixedPrecisionType, IntegerPrecisionType)):
        if not membrane.signed:
            raise Exception(f'Layer {node.name}: the membrane type {membrane.definition_cpp()} must be signed.')
        lsb = 2.0**-membrane.fractional
        largest = 2.0 ** (membrane.integer - membrane.signed) - lsb
        for stage in node.get_attr('stages'):
            beta, threshold = stage['beta'], stage['threshold']
            if not ((beta == 0 or lsb <= beta <= largest) and lsb <= threshold <= largest):
                raise Exception(
                    f'Layer {node.name}: beta {beta} or threshold {threshold} does not fit the membrane type '
                    f'{membrane.definition_cpp()}.'
                )

    result = node.get_output_variable().type.precision
    if isinstance(result, (FixedPrecisionType, IntegerPrecisionType)):
        if 2 ** (result.integer - result.signed) - 1 < node.get_attr('n_steps'):
            raise Exception(
                f'Layer {node.name}: the result type {result.definition_cpp()} cannot count '
                f'{node.get_attr("n_steps")} spikes. Set its "result" precision to "auto" or a wider type.'
            )


class ValidateStreamingSNN(OptimizerPass):
    name = 'validate_streaming_snn'

    def match(self, node):
        return isinstance(node, StreamingSNN)

    def transform(self, model, node):
        others = [layer for layer in model.get_layers() if layer is not node and not isinstance(layer, Input)]
        if others or node.inputs != model.inputs or node.outputs != model.outputs:
            raise Exception(
                f'Layer {node.name}: a model with a StreamingSNN layer may contain only its input and that layer.'
            )
        if model.config.pipeline_style not in (None, 'auto'):
            raise Exception(f'Layer {node.name}: a model with a StreamingSNN layer supports only PipelineStyle "auto".')
        return False


class GenerateStreamingSNN(OptimizerPass):
    """Generates the C++ of the block into nnet_code_gen.h, since its structure (number and kind of stages, their widths
    and kernel sizes) varies with the model."""

    name = 'generate_streaming_snn'

    def match(self, node):
        return isinstance(node, StreamingSNN) and 'streaming_snn_codegen' not in node.attributes

    def transform(self, model, node):
        _check_types(node)
        node.set_attr('streaming_snn_codegen', Source(_streaming_snn_code(node)))
        return False


class SetStreamingSNNInterface(OptimizerPass):
    """The block reads one input sample per clock cycle and pipelines its own loop over time steps, so the top function
    has no array pragmas and no top-level PIPELINE. Runs in the writer flow, since loading a saved model sets them again.
    """

    name = 'set_streaming_snn_interface'

    def match(self, node):
        return isinstance(node, StreamingSNN)

    def transform(self, model, node):
        node.get_input_variable().pragma = None
        node.get_output_variable().pragma = None
        model.config.pipeline_style = 'none'
        return False


class StreamingSNNConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(StreamingSNN)
        self.template = streaming_snn_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class StreamingSNNFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(StreamingSNN, include_header=[])
        self.template = streaming_snn_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['namespace'] = node.model.config.get_writer_config().get('Namespace', None) or 'nnet'
        return self.template.format(**params)
