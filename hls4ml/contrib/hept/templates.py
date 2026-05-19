from __future__ import annotations

import math

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate

from .layers import HEPTAttention


_HEPT_CONFIG_TEMPLATE = """struct config{index} : nnet::hept_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned embed_dim = {embed_dim};
    static const unsigned num_heads = {num_heads};
    static const unsigned dim_per_head = {dim_per_head};
    static const unsigned coords_dim = {coords_dim};
    static const unsigned num_hashes = {num_hashes};
    static const unsigned num_w_per_dist = {num_w_per_dist};
    static const unsigned block_size = {block_size};
    static const unsigned inter_reuse_factor = {inter_reuse_factor};
    static const unsigned qkv_reuse_factor = {qkv_reuse_factor};
    static const unsigned par_factor = {par_factor};
    static const unsigned par_factor_sort = {par_factor_sort};

    static const unsigned norm_inv_sqrt_table_size = {norm_inv_sqrt_table_size};
    static const unsigned norm_inv_sqrt_range = {norm_inv_sqrt_range};
    static const unsigned qkv_exp_table_size = {qkv_exp_table_size};
    static const unsigned qkv_exp_range = {qkv_exp_range};
    static const unsigned qkv_sqrt_table_size = {qkv_sqrt_table_size};
    static const unsigned qkv_sqrt_range = {qkv_sqrt_range};
    static const unsigned e2lsh_exp_table_size = {e2lsh_exp_table_size};
    static const unsigned e2lsh_exp_range = {e2lsh_exp_range};

    typedef {input_t} input_t;
    typedef {coords_t} coords_t;
    typedef {shift_t} shift_t;
    typedef {output_t} output_t;
    typedef {weight_t} weight_t;
    typedef {accum_t} accum_t;

    typedef {norm_accum_t} norm_accum_t;
    typedef {norm_sum_t} norm_sum_t;
    typedef {norm_sum_sqr_t} norm_sum_sqr_t;
    typedef {norm_mean_t} norm_mean_t;
    typedef {norm_var_table_t} norm_var_table_t;
    typedef {qkv_exp_table_t} qkv_exp_table_t;
    typedef {qkv_sqrt_table_t} qkv_sqrt_table_t;
    typedef {e2lsh_exp_table_t} e2lsh_exp_table_t;
    typedef {hash_mul_accum_t} hash_mul_accum_t;

    typedef {norm_out_t} norm_out_t;
    typedef {prepare_qk_t} prepare_qk_t;
    typedef {combined_shift_out_t} combined_shift_out_t;
    typedef {sorted_index_t} sorted_index_t;
    typedef {batched_index_select_out_t} batched_index_select_out_t;
    typedef {qkv_res_out_denom_t} qkv_res_out_denom_t;
    typedef {qkv_res_out_so_t} qkv_res_out_so_t;
    typedef {batched_index_select_unsort_out_o_t} batched_index_select_unsort_out_o_t;
    typedef {batched_index_select_unsort_out_logits_t} batched_index_select_unsort_out_logits_t;
    typedef {o_divide_logits_out_t} o_divide_logits_out_t;
}};
"""

_HEPT_FUNCTION_TEMPLATE = """nnet::hept_attention<{input_t}, {coords_t}, {shift_t}, {output_t}, {config}>(
    {input},
    {coords},
    {combined_shifts},
    {output},
    {layer_norm_weight},
    {layer_norm_bias},
    {q_weight},
    {k_weight},
    {v_weight},
    {sqrt_w},
    {alpha},
    {out_linear_weight},
    {out_linear_bias}
);"""


_DEFAULT_TYPE_STRINGS = {
    "input_t": "ap_fixed<16,6>",
    "coords_t": "ap_fixed<16,6>",
    "shift_t": "ap_fixed<16,6>",
    "output_t": "ap_fixed<16,6>",
    "accum_t": "ap_fixed<62,38>",
    "norm_accum_t": "ap_fixed<32,10>",
    "norm_sum_t": "ap_fixed<18,8,AP_RND_CONV,AP_WRAP,0>",
    "norm_sum_sqr_t": "ap_fixed<18,8,AP_RND_CONV,AP_WRAP,0>",
    "norm_mean_t": "ap_fixed<18,8,AP_RND_CONV,AP_WRAP,0>",
    "norm_var_table_t": "ap_ufixed<64,8,AP_RND_CONV,AP_SAT,0>",
    "qkv_exp_table_t": "ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0>",
    "qkv_sqrt_table_t": "ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0>",
    "e2lsh_exp_table_t": "ap_ufixed<64,20,AP_RND_CONV,AP_SAT,0>",
    "hash_mul_accum_t": "ap_fixed<32,16,AP_RND_CONV,AP_WRAP,0>",
}


def _type_name(obj, default: str) -> str:
    if obj is None:
        return default
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)


def _var_name(obj) -> str:
    return getattr(obj, "name", str(obj))


class HEPTConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HEPTAttention)
        self.template = _HEPT_CONFIG_TEMPLATE

    def format(self, node):
        if node.model.config.get_config_value("IOType") != "io_stream":
            raise ValueError("HEPTAttention currently requires IOType='io_stream'.")

        params = self._default_config_params(node)
        params["weight_t"] = node.get_weights("q_weight").type.name

        params["input_t"] = _type_name(node.get_attr("input_t", None), _DEFAULT_TYPE_STRINGS["input_t"])
        params["coords_t"] = _type_name(node.get_attr("coords_t", None), _DEFAULT_TYPE_STRINGS["coords_t"])
        params["shift_t"] = _type_name(node.get_attr("shift_t", None), _DEFAULT_TYPE_STRINGS["shift_t"])
        params["output_t"] = _type_name(node.get_attr("output_t", None), _DEFAULT_TYPE_STRINGS["output_t"])

        params["accum_t"] = _type_name(node.get_attr("accum_t", None), _DEFAULT_TYPE_STRINGS["accum_t"])
        params["norm_accum_t"] = _type_name(node.get_attr("norm_accum_t", None), _DEFAULT_TYPE_STRINGS["norm_accum_t"])
        params["norm_sum_t"] = _type_name(node.get_attr("norm_sum_t", None), _DEFAULT_TYPE_STRINGS["norm_sum_t"])
        params["norm_sum_sqr_t"] = _type_name(
            node.get_attr("norm_sum_sqr_t", None), _DEFAULT_TYPE_STRINGS["norm_sum_sqr_t"]
        )
        params["norm_mean_t"] = _type_name(node.get_attr("norm_mean_t", None), _DEFAULT_TYPE_STRINGS["norm_mean_t"])
        params["norm_var_table_t"] = _type_name(
            node.get_attr("norm_var_table_t", None), _DEFAULT_TYPE_STRINGS["norm_var_table_t"]
        )
        params["qkv_exp_table_t"] = _type_name(
            node.get_attr("qkv_exp_table_t", None), _DEFAULT_TYPE_STRINGS["qkv_exp_table_t"]
        )
        params["qkv_sqrt_table_t"] = _type_name(
            node.get_attr("qkv_sqrt_table_t", None), _DEFAULT_TYPE_STRINGS["qkv_sqrt_table_t"]
        )
        params["e2lsh_exp_table_t"] = _type_name(
            node.get_attr("e2lsh_exp_table_t", None), _DEFAULT_TYPE_STRINGS["e2lsh_exp_table_t"]
        )
        params["hash_mul_accum_t"] = _type_name(
            node.get_attr("hash_mul_accum_t", None), _DEFAULT_TYPE_STRINGS["hash_mul_accum_t"]
        )

        params["norm_out_t"] = _type_name(node.get_attr("norm_out_t", None), params["input_t"])
        params["prepare_qk_t"] = _type_name(node.get_attr("prepare_qk_t", None), params["input_t"])
        params["combined_shift_out_t"] = _type_name(node.get_attr("combined_shift_out_t", None), params["input_t"])

        seq_len = int(node.get_attr("seq_len"))
        index_width = max(1, math.ceil(math.log2(max(2, seq_len))))
        params["sorted_index_t"] = _type_name(node.get_attr("sorted_index_t", None), f"ap_uint<{index_width}>")

        params["batched_index_select_out_t"] = _type_name(
            node.get_attr("batched_index_select_out_t", None), params["input_t"]
        )
        params["qkv_res_out_denom_t"] = _type_name(node.get_attr("qkv_res_out_denom_t", None), params["input_t"])
        params["qkv_res_out_so_t"] = _type_name(node.get_attr("qkv_res_out_so_t", None), params["input_t"])
        params["batched_index_select_unsort_out_o_t"] = _type_name(
            node.get_attr("batched_index_select_unsort_out_o_t", None), params["input_t"]
        )
        params["batched_index_select_unsort_out_logits_t"] = _type_name(
            node.get_attr("batched_index_select_unsort_out_logits_t", None), params["input_t"]
        )
        params["o_divide_logits_out_t"] = _type_name(
            node.get_attr("o_divide_logits_out_t", None), params["output_t"]
        )

        params["norm_inv_sqrt_table_size"] = int(node.get_attr("norm_inv_sqrt_table_size", 1024))
        params["norm_inv_sqrt_range"] = int(node.get_attr("norm_inv_sqrt_range", 16))
        params["qkv_exp_table_size"] = int(node.get_attr("qkv_exp_table_size", 1024))
        params["qkv_exp_range"] = int(node.get_attr("qkv_exp_range", 8))
        params["qkv_sqrt_table_size"] = int(node.get_attr("qkv_sqrt_table_size", 1024))
        params["qkv_sqrt_range"] = int(node.get_attr("qkv_sqrt_range", 8))
        params["e2lsh_exp_table_size"] = int(node.get_attr("e2lsh_exp_table_size", 1024))
        params["e2lsh_exp_range"] = int(node.get_attr("e2lsh_exp_range", 8))

        return self.template.format(**params)


class HEPTFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HEPTAttention, include_header=["nnet_utils/nnet_hept.h"])
        self.template = _HEPT_FUNCTION_TEMPLATE

    def format(self, node):
        if node.model.config.get_config_value("IOType") != "io_stream":
            raise ValueError("HEPTAttention currently requires IOType='io_stream'.")

        params = self._default_function_params(node)

        coords_var = node.get_input_variable(node.inputs[1])
        shift_var = node.get_input_variable(node.inputs[2])

        params["coords_t"] = coords_var.type.name
        params["shift_t"] = shift_var.type.name
        params["coords"] = coords_var.name
        params["combined_shifts"] = shift_var.name

        for name in (
            "layer_norm_weight",
            "layer_norm_bias",
            "q_weight",
            "k_weight",
            "v_weight",
            "sqrt_w",
            "alpha",
            "out_linear_weight",
            "out_linear_bias",
        ):
            params[name] = _var_name(node.get_weights(name))

        return self.template.format(**params)
