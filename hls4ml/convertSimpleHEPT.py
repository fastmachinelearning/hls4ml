"""Minimal end-to-end demo of the HEPT attention contrib kernel in hls4ml.

It builds a single-layer HEPT attention block in PyTorch, converts it to an HLS
project with the `hls4ml.contrib.hept` extension, runs the C-simulation, and
compares the HLS output against the PyTorch reference element-by-element.

Note on scope: the generated HLS kernel implements only the attention sub-block
    norm1 -> q/k/v projections -> HEPT attention -> out_linear   (== `aggr_out`)
NOT the full transformer block (input residual + norm2 + feed-forward + second
residual). The reference below therefore computes just `aggr_out` so the two
sides are comparing the same quantity.

Usage:
    python convertSimpleHEPT.py [-m hept] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

import hls4ml
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model
from hls4ml.contrib.hept.torch import parse_hept_attention_layer
from hls4ml.contrib.hept.registration import register

from HEPT.src.models.baselines.transformer import HEPT
from HEPT.src.models.model_utils.hash_utils import get_regions, compute_combined_shifts

# Demo input dimensions.
SEQ_LEN = 600
COORDS_DIM = 6
PAR_FACTOR = 5
OUTPUT_DIR = "hept_test"
BACKEND = "Vitis"
IO_TYPE = "io_stream"


class PseudoHEPT(nn.Module):
    """Wraps a single HEPT attention block with an HLS-friendly forward signature."""

    def __init__(self, attn_type, coords_dim, **kwargs):
        super().__init__()
        self.attn = HEPT(attn_type, coords_dim, **kwargs)
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]),
            requires_grad=False,
        )
        self.block_size = kwargs["block_size"]

    def forward(self, data, coords, combined_shifts):
        # data:            (seq_len, embed_dim)
        # coords:          (seq_len, coords_dim)
        # combined_shifts: (seq_len, num_heads * num_hashes)  — precomputed region indices
        return self.attn(data, coords, combined_shifts)


def reference_attention(model, x, coords, combined_shifts):
    """PyTorch reference for the attention sub-block the HLS kernel implements (`aggr_out`)."""
    hept = model.attn
    with torch.no_grad():
        x_normed = hept.norm1(torch.tensor(x))  # pe_type == "none" -> no positional encoding
        q = hept.w_q(x_normed)
        k = hept.w_k(x_normed)
        v = hept.w_v(x_normed)
        out = hept.attn(
            q, k, v,
            coords=torch.tensor(coords),
            combined_shifts=torch.tensor(combined_shifts),
            raw_size=x.shape[0],
            key_padding_mask=None,
            edge_index=None,
            w_rpe=hept.w_rpe,
        )
    return out.detach().numpy()


def report(reference, hls_out):
    """Print a concise numerical comparison of the two outputs."""
    ref = reference.astype(np.float64)
    hls = hls_out.astype(np.float64)
    diff = hls - ref
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / np.maximum(np.abs(ref), 1e-12)
    corr = np.corrcoef(hls.ravel(), ref.ravel())[0, 1]

    print("\n=== PyTorch vs HLS numerical comparison ===")
    print(f"output shape        : {reference.shape}")
    print(f"max |abs diff|      : {abs_diff.max():.3e}")
    print(f"mean |abs diff|     : {abs_diff.mean():.3e}")
    print(f"RMS abs diff        : {np.sqrt(np.mean(diff ** 2)):.3e}")
    print(f"max relative diff   : {rel_diff.max():.3e}")
    print(f"mean relative diff  : {rel_diff.mean():.3e}")
    print(f"ref range [min,max] : [{ref.min():.3e}, {ref.max():.3e}]")
    print(f"Pearson correlation : {corr:.6f}")
    print("\ntoken 0 (first 6 dims):")
    print(f"  reference : {np.array2string(reference[0, :6], precision=4, floatmode='fixed')}")
    print(f"  hls       : {np.array2string(hls_out[0, :6], precision=4, floatmode='fixed')}")


def main():
    parser = argparse.ArgumentParser(description="Convert a single HEPT attention block to HLS and compare outputs.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible inputs/weights.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_dir = Path(f"./HEPT/src/configs/tracking/tracking_trans_{args.model}.yaml")
    model_kwargs = yaml.safe_load(config_dir.open("r").read())["model_kwargs"]
    embed_dim = model_kwargs["h_dim"]
    num_heads_x_hashes = model_kwargs["num_heads"] * model_kwargs["n_hashes"]

    # Register the HEPT contrib kernel (layer class, parser, backend templates, headers).
    register()
    hls4ml.converters.register_pytorch_layer_handler("HEPT", parse_hept_attention_layer)

    model = PseudoHEPT(args.model, COORDS_DIM, **model_kwargs)
    model.eval()  # disable dropout so the reference forward is deterministic

    # Random inputs. combined_shifts are the per-token region indices the kernel expects.
    x = np.random.rand(1, SEQ_LEN, embed_dim).astype(np.float32)
    coords = np.random.rand(1, SEQ_LEN, COORDS_DIM).astype(np.float32)
    combined_shifts = compute_combined_shifts(
        torch.tensor(coords[0]), model.regions, model.block_size
    ).unsqueeze(0).numpy()  # (1, seq_len, num_heads * num_hashes)

    # PyTorch reference (attention sub-block only).
    reference = reference_attention(model, x[0], coords[0], combined_shifts[0])

    # Convert to HLS and run the C-simulation.
    hls_config = config_from_pytorch_model(
        model,
        input_shape=[
            (None, SEQ_LEN, embed_dim),
            (None, SEQ_LEN, COORDS_DIM),
            (None, SEQ_LEN, num_heads_x_hashes),
        ],
        channels_last_conversion="off",
        transpose_outputs=False,
    )
    hls_config["Model"]["par_factor"] = PAR_FACTOR

    hls_model = convert_from_pytorch_model(
        model,
        hls_config=hls_config,
        output_dir=OUTPUT_DIR,
        backend=BACKEND,
        io_type=IO_TYPE,
    )
    hls_model.compile()

    # hls4ml returns the output flattened per sample; reshape to (seq_len, embed_dim).
    hls_prediction = np.asarray(hls_model.predict([x, coords, combined_shifts])).reshape(reference.shape)

    report(reference, hls_prediction)


if __name__ == "__main__":
    main()
