---
name: kernels
description: >-
  Write or modify hls4ml's C++ compute kernels. Use whenever the task touches a backend's nnet_utils headers
  or adds the kernel behind a Strategy. The kernel-contract sections describe the Vivado/Vitis family
  convention, which other backends follow only partly; the closing section says what holds for a backend that
  looks nothing like it. For the types flowing through these kernels, see precision-and-debugging.md.
globs:
  - "hls4ml/templates/**"
---

# hls4ml compute kernels

Paths are relative to the package directory `hls4ml/hls4ml/`.

## Scope: how much of this is a convention, not a rule

The C++ side is **not uniform across backends**. What follows describes the Vivado/Vitis convention, which
Libero follows closely and Catapult partly. Check the backend you are actually working on:

| Backend | `strategy` in the config struct | `kernel` typedef dispatch | header location | Dense weight layout |
| --- | --- | --- | --- | --- |
| Vivado, Vitis, Libero | yes | yes | `templates/<b>/nnet_utils/` | native `weights[i*n_out+j]` |
| Catapult | yes | no (branches inside the header) | `templates/<b>/nnet_utils/` | native |
| Quartus, oneAPI | no | no (`dense_rf_gt` / `dense_rf_lt` chosen by reuse factor) | `templates/<b>/firmware/nnet_utils/` | padded and rounded, `reuse_factor_rounded * block_factor_rounded` |
| symbolic | no nnet_utils at all | | | |

oneAPI further replaces `hls::stream` with pipes (variables carry `pipe_name`) and adds its own template
kinds in `backends/oneapi/oneapi_template.py`. A backend under development may share none of this.

So: read the target backend's own `nnet_common.h` and `nnet_dense.h` before assuming any of the sections
below. If the backend has no such files, skip to "If your backend does not look like this" at the end.

## The kernel contract (Vivado/Vitis family)

Kernels live in `templates/<backend>/nnet_utils/*.h` and are copied verbatim into
`<output_dir>/firmware/nnet_utils/` by the writer's `write_nnet_utils`. A backend can add a file from
elsewhere with `backend.register_source(abs_path, destination_dir='nnet_utils')`.

Generated C++ per layer is: a config struct in `parameters.h`, and a one-line call in `myproject.cpp`. For
Dense:

```cpp
struct config2 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 32;
    static const unsigned strategy = nnet::resource;
    typedef accum_default_t accum_t;
    typedef weight2_t weight_t;
    template<class data_T, class res_T, class CONFIG_T> using kernel = nnet::DenseResource_rf_leq_nin<...>;
    template<class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};
nnet::dense<input_t, layer2_t, config2>(input, layer2_out, w2, b2);
```

- `nnet::dense` in `nnet_dense.h` does nothing except dispatch:
  `CONFIG_T::template kernel<data_T, res_T, CONFIG_T>::dense(...)`.
- A kernel is a class deriving from `nnet::DenseKernel` in `nnet_function_stubs.h` with one static `dense`
  method of the fixed signature `(data[n_in], res[n_out], weights[n_in*n_out], biases[n_out])`.
  `Conv1DKernel` and `DepthwiseDenseKernel` follow the same pattern.
- A new kernel is therefore three things: the class in a header, a `dense_function` branch in the config
  template (`backends/vivado/passes/core_templates.py`), and an `init_dense` branch that sets `strategy`.
- Dense weight layout in this family is `weights[i * n_out + j]` (input-major), exactly as the frontend
  produced it. This is a per-backend decision, not a global one — Quartus and oneAPI pass a padded, rounded
  layout instead. If your kernel wants a different layout, rearrange it in a pass, not at runtime.
- Never hardcode `ap_fixed` inside a kernel. Use `typename CONFIG_T::accum_t` for accumulators,
  `CONFIG_T::template product<data_T, weight_t>::product(x, w)` for the multiply (this is what makes binary
  and ternary weights work), and `cast<data_T, res_T, CONFIG_T>(acc)` from `nnet_mult.h` for the output
  conversion.
- Shared helpers in `nnet_common.h`: the `io_parallel` / `io_stream` and `latency` / `resource` /
  `resource_unrolled` / `distributed_arithmetic` enums, `DIV_ROUNDUP`, `PRAGMA_DATA_PACK`, and the balanced
  tree `reduce<T, N, Op_add<T>>` for summing a fully unrolled array.

## io_parallel and io_stream

Two io types with these two representations is a Vivado-family arrangement. oneAPI carries data in pipes
instead, and a new backend may define something else entirely.

- **io_parallel:** tensors are plain C arrays, partitioned by a pragma chosen in `transform_types.py`. The
  kernel signature above is used directly.
- **io_stream:** tensors are `hls::stream<nnet::array<T, N>>` (see `nnet_types.h`). For Dense, the wrapper in
  `nnet_dense_stream.h` reads the input stream into a local array (`data_prepare`), calls the same
  array-based kernel, and writes the result out (`res_write`), so one kernel serves both io types. Conv and
  pooling do not work this way: their io_stream form is a separate line-buffer implementation.
- **The trap in that wrapper:** it branches on `CONFIG_T::strategy`. Under `latency` it calls
  `dense_latency_wrapper`, which carries `#pragma HLS PIPELINE II=CONFIG_T::reuse_factor` and therefore fully
  unrolls whatever kernel is underneath. A dataflow-style kernel must reach the `resource` branch, or its
  cost model is destroyed. When adding a strategy, check every place that compares against
  `nnet::latency` or `nnet::resource`, including `backends/vivado/passes/pipeline_style.py`.

## Dense is the simplest kernel, not a representative one

Dense is used above because it is the shortest complete example. It is a poor model for most other work,
because it has one input, one weight set, a fixed shape relation, and an io_stream path that is only a
wrapper around the array kernel. Before working on another family, read that family's own headers and
template file; the list below says what each one adds that Dense does not show.

| Family | Headers / templates | What is different |
| --- | --- | --- |
| Conv1D/2D | `nnet_conv*.h`, `nnet_conv_stream.h`, `convolution_templates.py` | io_stream has its own line-buffer implementation rather than a wrapper; per-layer C++ is **generated** by `im2col_codegen.py` and injected at the `// hls4ml insert code` marker in `nnet_code_gen.h` by `write_generated_code` |
| Separable / depthwise | `nnet_sepconv*.h`, `nnet_depthwise_product.h` | two weight sets and two chained kernels for one logical layer |
| Pooling | `nnet_pooling*.h`, `pooling_templates.py` | no weights; window arithmetic and padding dominate the config |
| Activations | `nnet_activation*.h`, activation part of `core_templates.py` | no weights; lookup tables sized by the `table_size` and `table_t` attributes registered in `FPGABackend._register_layer_attributes`; softmax additionally has the `fix_softmax_table_size` pass in `backends/fpga/passes/` |
| Merge, Concatenate | `nnet_merge*.h`, `merge_templates.py` | several inputs, so the template must resolve `get_input_variable(name)` per input rather than assuming `inputs[0]` |
| Recurrent (LSTM, GRU) | `nnet_recurrent.h`, `recurrent_templates.py` | separate `recurrent_reuse_factor` and `static` attributes, state carried across time steps, a nested Dense-like kernel per gate |
| Reshape, Flatten, Transpose | `nnet_transpose*.h`, `reshaping_templates.py` | may produce an `InplaceTensorVariable` — no data movement and no kernel call at all |
| Distributed arithmetic | `distributed_arithmetic.py`, `nnet_da_wrappers.h` | the kernel body itself is generated per layer from the weight values, so there is no fixed header to edit |
| Einsum, GarNet, SNN | `nnet_einsum*.h`, `nnet_garnet.h`, `nnet_snn*.h` | their own config vocabulary; useful precedents when a layer does not fit the matrix-vector shape |

Two consequences worth carrying into any kernel work: a change to the matrix-vector code path affects the
conv family too, because they share it; and a family whose io_stream form is a separate implementation
(conv, pooling) must be validated in both io types separately, since fixing one does not fix the other.

## Pragmas that matter

| Pragma | Effect | Cost when misused |
| --- | --- | --- |
| `PIPELINE II=n` | one result every n cycles; unrolls everything inside | full unroll of a large loop nest, huge LUT and DSP use |
| `UNROLL factor=k` | k copies of the loop body | silent resource blow-up if k is derived from a layer dimension |
| `ARRAY_PARTITION complete` | every element in its own register | fine for tens of elements, not for thousands |
| `ARRAY_RESHAPE cyclic factor=k` | k-wide access per cycle from one memory | mismatch with the loop's access pattern gives no speedup |
| `DATAFLOW` | run stages concurrently | only helps if stages communicate through streams or single-writer arrays |
| `INLINE` / `INLINE recursive` | remove the function boundary | large inlined bodies slow synthesis considerably |

Aim for one clear parallelism knob per kernel (hls4ml's is `reuse_factor`) and make the pragmas derive from
it, so the cost is predictable across shapes.

## If your backend does not look like this

For a backend under development, or one of the non-Vivado backends, treat everything above as one worked
example rather than a specification. Only the Python-side contract is fixed, and it is short:

- Each node must end up with the attributes the backend's own writer reads. For the Vivado family those are
  `config_cpp` and `function_cpp`; oneAPI adds `stream_function_cpp` and `task_sequence_cpp` through its own
  `Template` subclasses in `backends/oneapi/oneapi_template.py`. **Defining new template kinds is a supported
  extension point**, not a workaround — subclass `Template` with your own `attribute_name`.
- The backend must supply variable and type converters that turn `TensorVariable` and `NamedType` into
  whatever its language needs, applied by its own `transform_types` pass. Arrays and `hls::stream` are the
  Vivado answer; pipes are the oneAPI answer; a new backend may have a third.
- The writer is registered per backend and may emit any layout it wants. There is no required project
  structure, only the one each writer creates.
- Everything else — the `nnet::` namespace, `nnet_common.h`, the strategy enum, the `kernel` typedef, the
  `dense_config` base struct, io_parallel and io_stream as the only two io types — is a convention of the
  Vivado family that other backends adopt as far as it suits them.

When shared machinery does not fit a new backend, the intended response is for that backend to define its
own version, not to bend the design to what the shared code allows. Say out loud which convention you are
diverging from and why.

## Anti-checklist

- Do not write a standalone C++ harness that calls the kernel directly to measure it. Passing weights as an
  argument makes the tool infer a limited-port memory interface, throttling whichever kernel reads more
  weights per cycle. Compare through real hls4ml, where the scaffolding around both kernels is identical;
  see [**evaluating implementations**](evaluating-implementations.md).
- Do not carry a Vivado/Vitis convention into another backend without checking it there. The `strategy`
  field, the `kernel` typedef, the header path and the weight layout all differ across backends already.
- Do not generalize from Dense. Check whether the family you are touching has a separate io_stream
  implementation, generated per-layer code, several inputs, or several weight sets.
- Do not assume the io_stream wrapper will leave your kernel alone; check which strategy branch it takes.
- Do not hardcode a type, a shape, or a parallelism factor inside a kernel; everything comes from `CONFIG_T`.

For the precision rules these kernels obey, and for reading the project they generate, see
[precision and debugging](precision-and-debugging.md).
