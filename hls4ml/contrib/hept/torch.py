from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import torch
from einops import rearrange


def _strip_batch(shape):
    if shape is None:
        return None
    if len(shape) > 0 and shape[0] is None:
        return tuple(shape[1:])
    return tuple(shape)


def _get_weight(data_reader, layer_name: str, candidates: Iterable[str]) -> np.ndarray:
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = data_reader.get_weights_data(layer_name, candidate)
        except Exception as exc:  # noqa: BLE001 - converter backends vary
            errors.append((candidate, exc))
            data = None
        if data is not None:
            return np.asarray(data).reshape(-1)

    names = ", ".join(repr(name) for name in candidates if name)
    if errors:
        raise KeyError(
            f"None of the candidate HEPT weights for layer {layer_name!r} were found: {names}. "
            f"Last reader error: {errors[-1][1]!r}"
        )
    raise KeyError(f"None of the candidate HEPT weights for layer {layer_name!r} were found: {names}.")


def _expected_alpha_size(num_w_per_dist: int, coords_dim: int, num_hashes: int, num_heads: int) -> int:
    return (num_w_per_dist + 2 * coords_dim) * num_hashes * num_heads


def parse_hept_attention_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, cfg):
    """Parse a pytorch custom HEPTAttention layer into the hls4ml IR.

    Expected torch config keys
    --------------------------
    Required:
      * seq_len
      * embed_dim
      * num_heads
      * dim_per_head
      * coords_dim
      * num_hashes
      * par_factor
      * par_factor_sort

    Optional:
      * num_w_per_dist       (defaults to dim_per_head - coords_dim)
      * block_size           (defaults to par_factor)
      * inter_reuse_factor   (defaults to seq_len // num_heads)
      * qkv_reuse_factor     (defaults to seq_len // par_factor)
      * weight_names         (mapping overriding Keras variable names)

    Default weight aliases
    ----------------------
      * layer_norm_weight : layer_norm_weight | gamma | layer_norm1_weight
      * layer_norm_bias   : layer_norm_bias   | beta  | layer_norm1_bias
      * q_weight          : q_weight | weight_q
      * k_weight          : k_weight | weight_k
      * v_weight          : v_weight | weight_v
      * sqrt_w            : sqrt_w
      * alpha             : alpha | e2lsh
      * out_linear_weight : out_linear_weight | out_proj_weight
      * out_linear_bias   : out_linear_bias   | out_proj_bias
    """
    
    if input_names is None or len(input_names) != 3:
        raise ValueError(
            f"HEPTAttention parser expects 3 inputs (x, coords, combined_shifts); got {input_names!r}."
        )

    x_shape = _strip_batch(input_shapes[0])
    coords_shape = _strip_batch(input_shapes[1])
    shift_shape = _strip_batch(input_shapes[2])

    if x_shape is None or coords_shape is None or shift_shape is None:
        raise ValueError("HEPTAttention requires concrete input shapes for all three inputs.")

    seq_len = int(cfg.get("seq_len", x_shape[0]))
    embed_dim = int(cfg.get("embed_dim", x_shape[-1]))
    coords_dim = int(cfg.get("coords_dim", coords_shape[-1]))
    num_heads = class_object.attn.num_heads
    dim_per_head = class_object.attn.dim_per_head
    num_hashes = class_object.attn.n_hashes
    num_w_per_dist = class_object.attn.num_w_per_dist
    par_factor = int(cfg.get('HLSConfig',{}).get('Model',{}).get("par_factor", 1))
    par_factor_sort = int(cfg.get("par_factor_sort", par_factor))
    block_size = int(cfg.get("block_size", par_factor))
    inter_reuse_factor = int(cfg.get("inter_reuse_factor", max(1, seq_len // max(1, num_heads))))
    qkv_reuse_factor = int(cfg.get("qkv_reuse_factor", max(1, seq_len // max(1, par_factor))))

    if embed_dim != dim_per_head:
        raise ValueError(
            "The imported HEPT helper kernels currently require embed_dim == dim_per_head. "
            f"Got embed_dim={embed_dim}, dim_per_head={dim_per_head}."
        )

    if coords_shape[-1] != coords_dim:
        raise ValueError(f"coords width ({coords_shape[-1]}) does not match coords_dim ({coords_dim}).")

    expected_shift_width = num_heads * num_hashes
    if shift_shape[-1] != expected_shift_width:
        raise ValueError(
            f"combined_shifts width ({shift_shape[-1]}) does not match "
            f"num_heads * num_hashes ({expected_shift_width})."
        )

    weight_names = cfg.get("weight_names", {})
    def aliases(key: str, defaults: list[str]) -> list[str]:
        alias = weight_names.get(key)
        if alias is None:
            return defaults
        if isinstance(alias, str):
            return [alias]
        return list(alias)

    layer = {
        "class_name": "HEPTAttention",
        "name": layer_name,
        "inputs": input_names,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "dim_per_head": dim_per_head,
        "coords_dim": coords_dim,
        "num_hashes": num_hashes,
        "num_w_per_dist": num_w_per_dist,
        "block_size": block_size,
        "par_factor": par_factor,
        "par_factor_sort": par_factor_sort,
        "inter_reuse_factor": inter_reuse_factor,
        "qkv_reuse_factor": qkv_reuse_factor,
    }

    layer["layer_norm_weight_data"] = class_object.norm1.weight.detach().numpy()
    layer["layer_norm_bias_data"] = class_object.norm1.bias.detach().numpy()
    # The HLS qkv kernel (compute_qkv_pre_qk_merged) indexes weight as [in][out]
    # (flat = in * num_heads*dim_per_head + out), i.e. the transpose of the PyTorch
    # nn.Linear weight, which is stored [out][in]. Transpose here so the projection
    # q = norm(x) @ W matches the reference. (out_linear keeps PyTorch [out][in] layout.)
    layer["q_weight_data"] = class_object.w_q.weight.detach().numpy().T.copy()
    layer["k_weight_data"] = class_object.w_k.weight.detach().numpy().T.copy()
    layer["v_weight_data"] = class_object.w_v.weight.detach().numpy().T.copy()
    layer["alpha_data"] = class_object.attn.e2lsh.alpha.detach().numpy()
    layer["out_linear_weight_data"] = class_object.attn.out_linear.weight.detach().numpy()
    layer["out_linear_bias_data"] = class_object.attn.out_linear.bias.detach().numpy()


    w_rpe= class_object.w_rpe.weight

    w = rearrange(
        w_rpe,
        "(h d) (r k) -> h d r k",
        h=num_heads,
        d=dim_per_head,
        k=num_w_per_dist,
    )

    # Must match prep_qk() in HEPT/.../attention/hept.py, which builds the coords
    # augmentation as sqrt(2 * new_qw_expand_dim) * coords. The HLS kernel only
    # multiplies sqrt_w * coords, so the sqrt(2 * .) transform has to be baked in here.
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)
    layer["sqrt_w_data"] = torch.sqrt(2 * new_qw_expand_dim).detach().numpy()

    expected_lengths = {
        "layer_norm_weight": embed_dim,
        "layer_norm_bias": embed_dim,
        "q_weight": dim_per_head * dim_per_head * num_heads,
        "k_weight": dim_per_head * dim_per_head * num_heads,
        "v_weight": dim_per_head * dim_per_head * num_heads,
        "sqrt_w": num_heads * coords_dim,
        "alpha": _expected_alpha_size(num_w_per_dist, coords_dim, num_hashes, num_heads),
        "out_linear_weight": num_heads * dim_per_head * dim_per_head,
        "out_linear_bias": dim_per_head,
    }

    for name, expected_length in expected_lengths.items():
        actual_length = int(layer[f"{name}_data"].size)
        if actual_length != expected_length:
            warnings.warn(
                f"HEPTAttention layer {layer_name!r}: {name} has flattened length {actual_length}, "
                f"expected {expected_length} from the helper-kernel contract. "
                "This may be fine if your training-side layer packs weights differently, "
                "but you should verify the generated HLS indexing.",
                stacklevel=2,
            )

    return layer, [dim for dim in input_shapes[0]]
