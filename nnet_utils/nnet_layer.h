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

struct layer_settings {
    int roll_factor; 
} ;

template<class data_T, class res_T, class weight_T, class bias_T, class acc_T, int N_IN, int N_OUT>
void compute_layer(
    data_T    data[N_IN],
    res_T     res[N_OUT],
    weight_T  weights[N_IN][N_OUT],
    bias_T    biases[N_OUT]),
    layer_settings settings)
{

    data_T data_cache;
    acc_T acc[N_OUT];

    // is there a way to cyclically unroll multiple dimensions?
    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=acc complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    // Optional... Cuts down on a few of the BRAMs
    #if N_OUT > 16
        #pragma HLS RESOURCE variable=acc core=RAM_2P_LUTRAM
    #endif

    int unroll_factor_in  = N_IN / settings.roll_factor; 
    int unroll_factor_out = N_OUT / settings.roll_factor; 
    //int unroll_factor_in  = N_IN;
    //int unroll_factor_out = N_OUT;
    std::cout << unroll_factor_in << " " << unroll_factor_out << " " << settings.roll_factor  << std::endl;

    Reset: for(int iacc = 0; iacc < N_OUT; iacc++) {
      #pragma HLS UNROLL factor=unroll_factor_out 
      //#pragma HLS UNROLL 
        acc[iacc] = 0;
    }

    NewInput: for(int ii = 0; ii < N_IN; ii++) {
        #pragma HLS UNROLL factor=unroll_factor_in 
      //#pragma HLS UNROLL 
        data_cache = data[ii];
        Product: for(int jj = 0; jj < N_OUT; jj++) {
	  #pragma HLS UNROLL factor=unroll_factor_out
	  //#pragma HLS UNROLL 
            acc[jj] += data_cache * weights[ii][jj];
        }
    }

    Result: for(int ires = 0; ires < N_OUT; ires++)
	   #pragma HLS UNROLL factor=unroll_factor_out
      //#pragma HLS UNROLL
        res[ires] = (res_T) (acc[ires] + (acc_T) biases[ires]);

}

}

#endif
