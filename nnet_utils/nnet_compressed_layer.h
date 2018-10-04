//
//    hls4ml: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2018 Giuseppe Di Guglielmo
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_COMPRESSED_LAYER_H_
#define NNET_COMPRESSED_LAYER_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void compute_compressed_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::compressed_weight_t  weights[CONFIG_T::n_nonzeros],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    typename CONFIG_T::compressed_weight_t mult[CONFIG_T::n_nonzeros];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Do the compressed matrix-multiply
COMPRESSED_MAT_MULT_L:
	for(unsigned i = 0; i < CONFIG_T::n_nonzeros; i++) {
        // TODO: remove this division
		unsigned j = weights[i].index / CONFIG_T::n_out;
        mult[i].index = weights[i].index;
        mult[i].weight = weights[i].weight * data[j];
    }

    // Initialize accumulator with input biases
ACCUMULATOR_INIT_L:
	for(unsigned i = 0; i < CONFIG_T::n_out; i++) {
        acc[i] = (typename CONFIG_T::accum_t) (biases[i]);
    }

COMPRESSED_ACCUMULATOR_L:
    for(unsigned i = 0; i < CONFIG_T::n_nonzeros; i++) {
		unsigned j = mult[i].index % CONFIG_T::n_out;
        acc[j] += mult[i].weight;
    }

    // Cast to "res_t" type
RESULT_L:
	for(unsigned i = 0; i < CONFIG_T::n_out; i++){
        res[i] = (res_T) (acc[i]);
    }
}

}

#endif
