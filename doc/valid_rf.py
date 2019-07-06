#
# This script explores the valid values of reuse-factor for the MNIST MLP
# model.
#
# For the current large_layer implementation, the reuse-factor is valid iff
#   (multiplier_limit % N_IN) == 0
# where
#   multiplier_limit = ceil ((N_IN * N_OUT) / min(N_IN, RF))
#
# MNIST has 3 dense layers: 784 x 512 x 512 x 10
#

from __future__ import print_function

import math
import numpy as np

def print_valid_rf_per_layer(name, n_in, n_out):
    print('INFO: layer #', name, '(', n_in, 'x', n_out, ')')
    count = 0
    valid_reuse_factors = np.empty((0,), dtype=int)
    for rf in range(1, n_in * n_out):
        multfactor = min(n_in, rf)
        multiplier_limit = math.ceil((n_in * n_out) / float(multfactor))
        _assert = ((int(multiplier_limit % n_in) == 0) or (rf >= n_in)) and (((n_in * n_out) % rf) == 0)
        if _assert:
            valid_reuse_factors = np.append(valid_reuse_factors, int(rf))
    print('INFO: valid reuse factors:', valid_reuse_factors)
    print('INFO: valid reuse factor count:', len(valid_reuse_factors))

print_valid_rf_per_layer(1, 784, 512)
print_valid_rf_per_layer(2, 512, 512)
print_valid_rf_per_layer(3, 512, 10)


