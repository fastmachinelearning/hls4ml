import re
import typing
from math import prod
from pathlib import Path

if typing.TYPE_CHECKING:
    from hls4ml.model import ModelGraph


def create_jit_bridge_fn(model: 'ModelGraph'):
    inp_vars = model.get_input_variables()
    out_vars = model.get_output_variables()
    prj_name = model.config.config['ProjectName']

    inp_shapes = [tuple(v.shape) for v in inp_vars]
    out_shapes = [tuple(v.shape) for v in out_vars]

    inp_names = [v.name for v in inp_vars]
    out_names = [v.name for v in out_vars]

    inp_sizes = [prod(s) for s in inp_shapes]
    out_sizes = [prod(s) for s in out_shapes]

    n_out = len(out_names)

    input_def = '\n    '.join(f'std::vector<T> {v.name}, ' for v in inp_vars)[:-2]

    inp_size_def = '\n    '.join(f'constexpr size_t {n}_size = {s};' for n, s in zip(inp_names, inp_sizes))
    out_size_def = '\n    '.join(f'constexpr size_t {n}_size = {s};' for n, s in zip(out_names, out_sizes))

    ptr_buf_def = '\n    '.join(f'T* {v.name}_ptr = {v.name}.data();' for v in inp_vars + out_vars)
    n_samples_def = f'{inp_vars[0].name}.size() / {inp_vars[0].name}_size'

    assertions_def = ' ||\n    '.join(f'({n}.size() != {n}_size * n_samples)' for n in inp_names)

    inp_args_def_list = [f'{n}_ptr + i * {n}_size,' for n in inp_names]
    out_args_def_list = [f'{n}_ptr + i * {n}_size,' for n in out_names]
    args_def = ('\n' + ' ' * 12).join(inp_args_def_list + out_args_def_list)[:-1]

    out_var_def = '\n    '.join(f'std::vector<T> {v.name}({v.name}_size * n_samples);' for v in out_vars)

    _ret_template_arg = ('std::vector<T>, ' * n_out)[:-2]
    _ret_tuple_arg = ', '.join(out_names)
    return_def = f'std::tuple<{_ret_template_arg}>({_ret_tuple_arg})'

    cpp_fn = f"""
template <typename T>
auto batch_inference(
    {input_def}
){{
    {inp_size_def}
    {out_size_def}

    size_t n_samples = {n_samples_def};

    if (
        {assertions_def}
    )
        throw std::runtime_error("Invalid input sizes: number of samples or input sizes do not match");

    {out_var_def}

    {ptr_buf_def}

    #pragma omp parallel for
    for (int i = 0; i < n_samples; i++) {{
        if (std::is_same<T, double>::value) {{
            {prj_name}_double(
                {args_def}
            );
        }} else if (std::is_same<T, float>::value) {{
            {prj_name}_float(
                {args_def}
            );
        }} else {{
            throw std::runtime_error("Unsupported type");
        }}
    }}
    
    return {return_def};
}}
"""
    return cpp_fn


def create_jit_weight_filler(model: 'ModelGraph'):
    filler_fn = """
template <typename T>
void fill_weight(T weight[], std::vector<float> vec) {{
    for (size_t i = 0; i < vec.size(); i++) {{
        weight[i] = vec[i];
    }}
}}
"""
    return filler_fn


class Writer:
    def __init__(self):
        pass

    def write_hls(self, model):
        raise NotImplementedError

    def write_jit_bridge(self, model: 'ModelGraph'):
        """Write the Python-C++ bridge for JIT compilation (myproject_bridge_jit.cpp)"""

        prj_name = model.config.get_project_name()
        prj_path = Path(model.config.get_output_dir())
        path_c_bridge = prj_path / f'{prj_name}_bridge.cpp'
        path_jit_bridge = prj_path / f'{prj_name}_jit_bridge.cpp'

        namespace = model.config.get_writer_config()['Namespace'] or 'nnet'

        cpp_source_bridge = path_c_bridge.read_text()

        # Remove the unsigned short &const_size_... arguments from the function signature
        # For Quartus
        m = re.compile(r',\s*unsigned short &const_size_\w+', re.MULTILINE)
        cpp_source_bridge = m.sub('', cpp_source_bridge)

        jit_bridge_fn = create_jit_bridge_fn(model)
        weight_filler_fn = create_jit_weight_filler(model)

        _plugin_code = f'namespace {namespace} {{\n' + jit_bridge_fn + '\n\n' + weight_filler_fn + '\n}\n#endif'

        cpp_source_bridge = cpp_source_bridge.replace('extern "C" {', f'namespace {namespace} {{')
        cpp_source_bridge = cpp_source_bridge.replace('#endif', _plugin_code)

        path_jit_bridge.write_text(cpp_source_bridge)


writer_map = {}


def register_writer(name, writer_cls):
    if name in writer_map:
        raise Exception(f'Writer {name} already registered')

    writer_map[name] = writer_cls


def get_writer(name):
    return writer_map[name]()
