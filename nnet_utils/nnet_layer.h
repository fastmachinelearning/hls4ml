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

#ifndef NNET_LAYER_H_
#define NNET_LAYER_H_

#include "nnet_default.h"
#include "hls_stream.h"

namespace nnet {

struct layer_t
{
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;
    static const bool fully_unrolled = true;
    static const unsigned roll_factor = 1;
};

template<class data_T, class res_T, class weight_T, class bias_T, class acc_T, typename CONFIG_T>
void compute_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    weight_T  weights[CONFIG_T::n_in][CONFIG_T::n_out],
    bias_T    biases[CONFIG_T::n_out])
{

    data_T data_cache;
    acc_T acc[CONFIG_T::n_out];

    // is there a way to cyclically unroll multiple dimensions?
    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=acc complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    // Optional... Cuts down on a few of the BRAMs
    // #if CONFIG_T::n_out > 16
    //     #pragma HLS RESOURCE variable=acc core=RAM_2P_LUTRAM
    // #endif

    int unroll_factor_in  = CONFIG_T::n_in / CONFIG_T::roll_factor;
    int unroll_factor_out = CONFIG_T::n_out / CONFIG_T::roll_factor;
    //int unroll_factor_in  = CONFIG_T::n_in;
    //int unroll_factor_out = CONFIG_T::n_out;
    std::cout << "Unroll: " << unroll_factor_in << " " << unroll_factor_out << " " << CONFIG_T::roll_factor  << std::endl;

    Reset: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
      #pragma HLS UNROLL factor=unroll_factor_out
      //#pragma HLS UNROLL 
        acc[iacc] = 0;
    }

    NewInput: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        #pragma HLS UNROLL factor=unroll_factor_in
        //#pragma HLS UNROLL 
        data_cache = data[ii];
        Product: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
        #pragma HLS UNROLL factor=unroll_factor_out
        //#pragma HLS UNROLL
            acc[jj] += data_cache * weights[ii][jj];
        }
    }

    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        #pragma HLS UNROLL factor=unroll_factor_out
        //#pragma HLS UNROLL
        res[ires] = (res_T) (acc[ires] + (acc_T) biases[ires]);
    }

}

}

#endif
