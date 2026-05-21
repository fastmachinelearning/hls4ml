import argparse
import numpy as np
import torch
from pathlib import Path
import yaml
from torch import nn
from HEPT.src.utils.get_data import get_data_loader, get_dataset
from HEPT.src.utils.get_model import get_model
from HEPT.src.models.baselines.transformer import HEPT

import hls4ml
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model
from hls4ml.contrib.hept.torch import parse_hept_attention_layer
from hls4ml.contrib.hept.registration import register

class pseudoHEPT(nn.Module):
	def __init__(self, attn_type, coords_dim, **kwargs):
		super().__init__()
		self.attn = HEPT(attn_type, coords_dim, **kwargs)
		
	def forward(self, data, coords, combined_shifts):
		return self.attn(data, coords, combined_shifts)	

def main():
    parser = argparse.ArgumentParser(description="Train a model for tracking.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    args = parser.parse_args()

    if args.model in ["gcn", "gatedgnn", "dgcnn", "gravnet"]:
        config_dir = Path(f"./HEPT/src/configs/tracking/tracking_gnn_{args.model}.yaml")
    else:
        config_dir = Path(f"./HEPT/src/configs/tracking/tracking_trans_{args.model}.yaml")

    output_dir = "hept_test"
    backend = "Vitis"
    io_type = "io_stream"

    register()

    hls4ml.converters.register_pytorch_layer_handler('HEPT', parse_hept_attention_layer)    

    config = yaml.safe_load(config_dir.open("r").read())

    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    # ~ model = get_model(model_name, config["model_kwargs"], dataset_name)
    print (config["model_kwargs"])
    model = pseudoHEPT(args.model,6,**config["model_kwargs"])
    config = config_from_pytorch_model(model, input_shape = [(None, 600, 16), (None, 600, 6), (None, 600, 6)], channels_last_conversion="internal",transpose_outputs=False)
    print (config)

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)
    hls_model.compile()


if __name__ == "__main__":
    main()

