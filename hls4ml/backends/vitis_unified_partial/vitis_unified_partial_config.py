import json
import os

import numpy as np

from hls4ml.backends.vitis_unified.vitis_unified_config import VitisUnifiedConfig


class VitisUnifiedPartialConfig(VitisUnifiedConfig):


    def __init__(self, config, model_inputs, model_outputs):

        super().__init__(config, model_inputs, model_outputs)

        self.free_interim_input = self.config.get('MultiGraphConfig', {}).get('IOInterimType', {}).get(
            "Input") == "io_free_stream"
        self.free_interim_output = self.config.get('MultiGraphConfig', {}).get('IOInterimType', {}).get(
            "Output") == "io_free_stream"

        self.amt_graph = self.config.get('MultiGraphConfig', {}).get('amtGraph', -1)
        self.graph_idx = self.config.get('MultiGraphConfig', {}).get('graphIdx', -1)

        self.mgs_meta = self.config.get('MultiGraphConfig', {}).get('MgsMeta', None)




    def is_free_interim_input(self):
        return self.free_interim_input

    def is_free_interim_output(self):
        return self.free_interim_output

    def get_amt_graph(self):
        return self.amt_graph

    def get_graph_idx(self):
        return self.graph_idx

    def get_mgs_meta_list(self):
        ### it is supposed to return [(dataWidth, IndexWidth, ...), ... ]
        return self.mgs_meta

