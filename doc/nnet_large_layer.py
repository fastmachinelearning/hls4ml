from __future__ import print_function

import numpy as np
from math import ceil

def DIV_ROUNDUP(n, d):
    return int(ceil((n / float(d))))

def MIN(a, b):
    return a if (a < b) else b

n_in = 16
n_out = 8
RF = 8

block_factor = DIV_ROUNDUP(n_in * n_out , RF)
multfactor = MIN(n_in, RF)
multiplier_limit = DIV_ROUNDUP(n_in * n_out, multfactor)
multscale = multiplier_limit / n_out

print("INFO: n_in = ", n_in)
print("INFO: n_out = ", n_out)
print("INFO: RF = ", RF)
print("INFO: block_factor = ", block_factor)
print("INFO: n_in * n_out = ", n_in * n_out)

# Create and initialize arrays
data = np.arange(n_in).astype(float)
biases = np.arange(n_out).astype(float)
weights = np.arange( n_in * n_out).astype(float)

# transpose the weights matrix
weights_T = weights.reshape(n_in, n_out).transpose().reshape(n_in * n_out, 1)

# Python implementation of nnet_utils/nnet_large_layer.h
def nnet_large_layer(data, weights, biases):

    acc = np.zeros(n_out)
    for iacc in range(n_out):
        acc[iacc] = biases[iacc]

    for ir in range(RF):
        print("INFO: --- reuse --- ir", ir, " -----------------------------")
        tmpmult = np.zeros(block_factor)

        for im in range(block_factor):
            w_index = ir + RF * im
            in_index = w_index % n_in
            if (w_index >= n_in * n_out):
                continue
            print("INFO: data[", in_index, "], weights[", w_index, "]")
            tmpmult[im] = data[in_index] * weights[w_index]

        mult = np.zeros(multiplier_limit)

        for im in range(block_factor):
            w_index = ir + RF * im
            out_index = int(w_index / multfactor)
            if (out_index >= multiplier_limit):
                continue
            mult[out_index] = mult[out_index] + tmpmult[im]

        for im in range(multiplier_limit):
            out_index = int(im / multscale)
            acc[out_index] = acc[out_index] + mult[im]

    res = np.zeros(n_out)

    for ires in range(n_out):
        res[ires] = acc[ires]

    return res

# A reference implementation of a FC layer
def fully_connected_layer(data, weights, biases):
    return np.matmul(data, weights.reshape(n_in, n_out)) + biases

implementation_results = nnet_large_layer(data, weights_T, biases)
reference_results = fully_connected_layer(data, weights, biases)

print("INFO:===============================================================================")
print("INFO: implementation: ", implementation_results)
print("INFO: reference     : ", reference_results)
