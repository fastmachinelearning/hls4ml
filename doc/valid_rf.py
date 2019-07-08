#
# This script returns the valid values of reuse-factor for a given MLP model. A
# reuse-factor value is valid if it passes some conditions. Those can be either
# for the functional correctness of the nnet_large_layer.h implementation or
# for a succesful Vivado HLS execution.
#
# The script loads the JSON file of an MLP model and, for each dense layer,
# returns the values that pass the following conditions.

# For the current nnet_large_layer.h implementation, a reuse factor is valid
# iff
#   (multiplier_limit % n_out) == 0 or (rf >= n_in)
# where
#   multiplier_limit = ceil ((n_in * n_out) / min(n_in, rf))
#
# We also enforce
#   ((n_in * n_out) % rf) == 0
# which provides higher chances of a succesful run of Vivado HLS.

from __future__ import print_function

import sys
import math
import keras
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.models import model_from_json

#
# Define and check the conditions.
#
# THIS IS THE MOST IMPORTANT FUNCTION HERE!
#
def check_conditions(n_in, n_out, rf):
    multfactor = min(n_in, rf)
    multiplier_limit = math.ceil((n_in * n_out) / float(multfactor))
    #
    # THIS ASSERTION IS FOR THE FUNCTIONAL CORRECTNESS OF THE LARGE LAYER
    #
    _assert = ((int(multiplier_limit % n_out) == 0) or (rf >= n_in))
    #
    # THIS ASSERTION IS FOR QoR AND EXECUTION TIME OF VIVADO HLS
    #
    _assert = _assert and (((n_in * n_out) % rf) == 0)

    return _assert

# Get the list of reuse-factor values that pass the conditions. The
# reuse-factor tested are in the range [1, max_rf].
def get_valid_rf_per_layer(layer, max_rf):
    layer_input_shape = layer.input_shape
    layer_output_shape = layer.output_shape
    n_in = layer_input_shape[1]
    n_out = layer_output_shape[1]
    # Create an empty list to populate.
    valid_reuse_factors = np.empty((0,), dtype=int)
    for rf in range(1, max_rf):
        _assert = check_conditions(n_in, n_out, rf)
        if _assert:
            valid_reuse_factors = np.append(valid_reuse_factors, int(rf))
    return valid_reuse_factors

# Print on the console the valid reuse-factor values and other information:
# - layer name
# - max and min input and output dimensions in the model
def print_reuse_factors(layer, max_n_in, max_n_out, reuse_factors):
    layer_input_shape = layer.input_shape
    layer_output_shape = layer.output_shape
    name = layer.name
    n_in = layer_input_shape[1]
    n_out = layer_output_shape[1]
    print('INFO: ============================================================')
    print('INFO: layer #', name, '(', n_in, 'x', n_out, ')')
    print('INFO: total reuse factors:', max_n_in*max_n_out, "=", max_n_in, "*", max_n_out)
    print('INFO: valid reuse factors:', reuse_factors)
    print('INFO: valid reuse factor count:', len(reuse_factors))
    print('INFO: ============================================================')

# Traverse the dense layers of an MLP model and return the maximum input and
# output dimensions.
def get_max_n_in_n_out(model):
    max_n_in = 0
    max_n_out = 0
    for layer in model.layers:
        layer_name = layer.name
        layer_class = layer.__class__.__name__
        if layer_class == 'Dense':
            layer_input_shape = layer.input_shape
            layer_output_shape = layer.output_shape
            n_in = layer_input_shape[1]
            n_out = layer_output_shape[1]
            max_n_in = max(n_in, max_n_in)
            max_n_out = max(n_out, max_n_out)
    return (max_n_in, max_n_out)

# The top of the hill :-)
def main():
    argc = len(sys.argv)
    if (argc != 2):
        print("ERROR: usage: python", str(sys.argv[0]), "<JSON file>")
        raise SystemExit
    json_filename = str(sys.argv[1])
    with open(json_filename, 'r') as f:
        model = model_from_json(f.read())
        max_n_in, max_n_out = get_max_n_in_n_out(model)
        rf_per_layer = []
        for layer in model.layers:
            layer_name = layer.name
            layer_class = layer.__class__.__name__
            if layer_class == 'Dense':
                reuse_factors = get_valid_rf_per_layer(layer, max_n_in * max_n_out)
                rf_per_layer.append(reuse_factors)
                print_reuse_factors(layer, max_n_in, max_n_out, reuse_factors)
                print('INFO:')
        rf_per_model = rf_per_layer[0]
        if (len(rf_per_layer) > 1):
            for i in range(1, len(rf_per_layer)):
                rf_per_model = np.intersect1d(rf_per_model, rf_per_layer[i])
        print('INFO: ============================================================')
        print('INFO: model')
        print('INFO: valid reuse factor per model:', rf_per_model)
        print('INFO: valid reuse factor per model count:', len(rf_per_model))
        print('INFO: ============================================================')

if __name__ ==  "__main__":
    main();
