from __future__ import print_function

import numpy as np
from math import ceil

def block_partition(data, n_in, n_out, block_factor):
    rf = int(ceil((n_in * n_out) / float(block_factor)))
    return np.resize(weights, (block_factor, rf))

n_in = 16
n_out = 8
RF = 8

block_factor = int(ceil((n_in * n_out) / float(RF)))

print("INFO: n_in = ", n_in)
print("INFO: n_out = ", n_out)
print("INFO: RF = ", RF)
print("INFO: block_factor = ", block_factor)
print("INFO: n_in * n_out = ", n_in * n_out)

# Create and initialize arrays
data = np.arange(n_in)
biases = np.arange(n_out)
weights = np.arange( n_in * n_out ).reshape(n_in, n_out)

def nnet_large_layer(data, weights, biases):
    #weights = block_partition(weights, n_in, n_out, block_factor)
    print("INFO: weights.shape = ", weights.shape)
    imp_in_index = 0
    for ir in range(RF):
        print("INFO: ir", ir, " =============================")
        for im in range(block_factor):
            w_index = ir + (RF * im)
            d_index = w_index % n_in
            if (w_index >= n_in * n_out):
                pass
            print("INFO: weights[", w_index, "], data[", d_index, "]")
            #print("INFO: weights[", im, "][", ir, "], weights[", w_index, "], data[", d_index, "]")

nnet_large_layer(data, weights, biases)
