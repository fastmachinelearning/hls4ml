from __future__ import print_function
import logging

import numpy as np
from math import ceil

def DIV_ROUNDUP(n, d):
    return int(ceil((n / float(d))))

def MIN(a, b):
    return a if (a < b) else b

logging.basicConfig(level=logging.INFO)

n_in = 16
n_out = 8
RF = 4

block_factor = DIV_ROUNDUP(n_in * n_out , RF)
multfactor = MIN(n_in, RF)
multiplier_limit = DIV_ROUNDUP(n_in * n_out, multfactor)
multscale = multiplier_limit / n_out

if (multiplier_limit % n_out != 0):
    logging.info("ERROR: RF = %d is not acceptable (multiplier_limit % n_out = %d (!= 0))", RF, multiplier_limit % n_out)
    exit(1)

logging.info("INFO:===============================================================================")
logging.info("n_in = %d", n_in)
logging.info("n_out = %d", n_out)
logging.info("n_in * n_out = %d", n_in * n_out)
logging.info("RF = %d", RF)
logging.info("block_factor = %d", block_factor)
logging.info("multfactor = %d", multfactor)
logging.debug("INFO:===============================================================================")

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
        logging.debug("--- reuse --- ir %d -----------------------------", ir)
        tmpmult = np.zeros(block_factor)

        logging.debug("MultLoop:")
        for im in range(block_factor):
            w_index = ir + RF * im
            in_index = w_index % n_in
            if (w_index >= n_in * n_out):
                continue
            logging.debug("  data[ %d ], weights[ %d ]", in_index, w_index)
            tmpmult[im] = data[in_index] * weights[w_index]

        mult = np.zeros(multiplier_limit)

        logging.debug("AccumLoop1:")
        for im in range(block_factor):
            w_index = ir + RF * im
            out_index = int(w_index / multfactor)
            logging.debug("  w_index = %d, multfactor = %d", w_index, multfactor)

            if (out_index >= multiplier_limit):
                continue
            logging.debug("  mult[ %d ] += tmpmult[ %d ]", out_index, im)
            mult[out_index] = mult[out_index] + tmpmult[im]

        logging.debug("AccumLoop2:")
        for im in range(multiplier_limit):
            out_index = int(im / multscale)
            logging.debug("  acc[ %d ] += mult[ %d ]", out_index, im)
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

validation_result = np.array_equal(implementation_results, reference_results)

logging.info("INFO:===============================================================================")
print("implementation:", implementation_results)
print("reference     :", reference_results)
if (validation_result):
    logging.info("validation: PASS")
else:
    logging.info("validation: FAIL")
