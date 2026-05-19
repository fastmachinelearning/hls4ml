from __future__ import annotations

import hls4ml
from hls4ml.model.attributes import Attribute


_HEPT_WEIGHT_SPECS: tuple[tuple[str, str], ...] = (
    ("layer_norm_weight", "layer_norm_weight{index}"),
    ("layer_norm_bias", "layer_norm_bias{index}"),
    ("q_weight", "q_weight{index}"),
    ("k_weight", "k_weight{index}"),
    ("v_weight", "v_weight{index}"),
    ("sqrt_w", "sqrt_w{index}"),
    ("alpha", "alpha{index}"),
    ("out_linear_weight", "out_linear_weight{index}"),
    ("out_linear_bias", "out_linear_bias{index}"),
)


class HEPTAttention(hls4ml.model.layers.Layer):
    """IR node for a HEPT attention block.

    Input order:
      1. x                : [seq_len, embed_dim]
      2. coords           : [seq_len, coords_dim]
      3. combined_shifts  : [seq_len, num_heads * num_hashes]

    Output:
      * out               : [seq_len, embed_dim]

    Notes
    -----
    The first implementation deliberately mirrors the assumptions already present
    in Howard1011/HEPT_HLS:

      * `embed_dim == dim_per_head`
      * `combined_shifts` carries one value per `(head, hash)` pair
      * the backend uses `IOType = io_stream`

    Those restrictions come from the current low-level HEPT kernels; the hls4ml
    side is now parameterized, but the imported HEPT helper headers still impose
    these interface contracts.
    """

    _expected_attributes = [
        Attribute("seq_len"),
        Attribute("embed_dim"),
        Attribute("num_heads"),
        Attribute("dim_per_head"),
        Attribute("coords_dim"),
        Attribute("num_hashes"),
        Attribute("num_w_per_dist"),
        Attribute("block_size"),
        Attribute("par_factor"),
        Attribute("par_factor_sort"),
        Attribute("inter_reuse_factor"),
        Attribute("qkv_reuse_factor"),
    ]

    def initialize(self):
        if len(self.inputs) != 3:
            raise ValueError(
                f"{self.__class__.__name__} expects exactly 3 inputs "
                f"(x, coords, combined_shifts); got {len(self.inputs)}."
            )

        x = self.get_input_variable(self.inputs[0])
        coords = self.get_input_variable(self.inputs[1])
        combined_shifts = self.get_input_variable(self.inputs[2])

        x_shape = list(x.shape)
        coords_shape = list(coords.shape)
        shift_shape = list(combined_shifts.shape)

        if len(x_shape) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects x to have rank 2 "
                f"[seq_len, embed_dim], got shape {x_shape}."
            )
        if len(coords_shape) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects coords to have rank 2 "
                f"[seq_len, coords_dim], got shape {coords_shape}."
            )
        if len(shift_shape) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects combined_shifts to have rank 2 "
                f"[seq_len, num_heads * num_hashes], got shape {shift_shape}."
            )

        if x_shape[0] != coords_shape[0] or x_shape[0] != shift_shape[0]:
            raise ValueError(
                "All HEPT inputs must share the same sequence length, got "
                f"x={x_shape}, coords={coords_shape}, combined_shifts={shift_shape}."
            )

        self.set_attr("seq_len", int(self.get_attr("seq_len")))
        self.set_attr("embed_dim", int(self.get_attr("embed_dim")))
        self.set_attr("coords_dim", int(self.get_attr("coords_dim")))
        self.set_attr("num_heads", int(self.get_attr("num_heads")))
        self.set_attr("dim_per_head", int(self.get_attr("dim_per_head")))
        self.set_attr("num_hashes", int(self.get_attr("num_hashes")))
        self.set_attr("num_w_per_dist", int(self.get_attr("num_w_per_dist")))
        self.set_attr("block_size", int(self.get_attr("block_size")))
        self.set_attr("par_factor", int(self.get_attr("par_factor")))
        self.set_attr("par_factor_sort", int(self.get_attr("par_factor_sort")))
        self.set_attr("inter_reuse_factor", int(self.get_attr("inter_reuse_factor")))
        self.set_attr("qkv_reuse_factor", int(self.get_attr("qkv_reuse_factor")))

        if self.get_attr("embed_dim") != self.get_attr("dim_per_head"):
            raise ValueError(
                "The imported HEPT helper kernels currently require embed_dim == dim_per_head. "
                f"Got embed_dim={self.get_attr('embed_dim')} and "
                f"dim_per_head={self.get_attr('dim_per_head')}."
            )

        if coords_shape[1] != self.get_attr("coords_dim"):
            raise ValueError(
                f"coords width ({coords_shape[1]}) does not match coords_dim "
                f"({self.get_attr('coords_dim')})."
            )

        expected_shift_width = self.get_attr("num_heads") * self.get_attr("num_hashes")
        if shift_shape[1] != expected_shift_width:
            raise ValueError(
                f"combined_shifts width ({shift_shape[1]}) does not match "
                f"num_heads * num_hashes ({expected_shift_width})."
            )

        self.add_output_variable(x_shape)

        for attr_name, var_name in _HEPT_WEIGHT_SPECS:
            self.add_weights_variable(name=attr_name, var_name=var_name, data=attr_name)
