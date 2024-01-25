//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
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

#ifndef NNET_LAYERNAME_H_
#define NNET_LAYERNAME_H_

#include "ap_fixed.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
//#include <math.h>

namespace nnet {

struct layername_config {
    // Internal data type definitions (replaced by the types in the python description of the layer)
    deftypedef

        // Add here default values for layer attributes and variables needed in function implementation (accessible through
        // the CONFIG_T namespace)

        static const unsigned n_in = 10;
    exampledef

        // Resource reuse info
        static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;

    // Example of function taken from the nnet_common.h header file
    // template<class x_T, class y_T>
    // using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void layername(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in], arglist) {
    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    // #pragma HLS function_instantiate variable=eps

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    // #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
    // #pragma HLS ARRAY_PARTITION variable=scale complete
    // #pragma HLS ARRAY_PARTITION variable=bias complete

    ////////////////INSERT FUNCTION HERE
}

} // namespace nnet

#endif
