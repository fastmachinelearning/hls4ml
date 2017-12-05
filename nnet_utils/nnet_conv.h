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

#ifndef NNET_CONV_H_
#define NNET_CONV_H_

#include "nnet_common.h"

namespace nnet {

struct conv_config
{
    // Internal data type definitions                                                                                      
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    static const unsigned y_in = 10;
    static const unsigned n_chan = 1;
    static const unsigned y_filt = 2;

    static const bool fully_unrolled = true;
    static const unsigned roll_factor_in = 1;
    static const unsigned roll_factor_out = 1;
    static const bool store_weights_in_bram = false;
    // partitioning arrays cyclically to go with roll factors?
};

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d(
	     data_T    data[CONFIG_T::y_in][CONFIG_T::n_chan],
	     res_T     res[CONFIG_T::y_in][CONFIG_T::n_chan],
	     typename CONFIG_T::weight_t  weights[CONFIG_T::y_filt][CONFIG_T::n_chan],
	     typename CONFIG_T::bias_t    biases[CONFIG_T::n_chan])
{
    // conv_1d: 1-dimensional convolution
    //   - Also includes multiple input channels
    //   - Only allows new data on each ROW (i.e. this is NOT a 2D convolution)
    // Only ONE output channel per input channel

    // Initial directives used from HLS User guide, pg 381
    // (https://www.xilinx.com/support/documentation/sw_manuals/xilinx2015_4/ug902-vivado-high-level-synthesis.pdf)

    // TODO: Figure out how to correctly pipeline FiltLoop-- It sort of needs to be pipelined
    // across iterations of ChanLoop. Otherwise it does not want to consistently hit 10 ns timing
    data_T buffer[CONFIG_T::y_filt][CONFIG_T::n_chan];
    for(int ii = 0; ii < CONFIG_T::y_filt; ii++) {
      for(int chan = 0; chan < CONFIG_T::n_chan; chan++){
      #pragma HLS UNROLL
	// Initialize buffer to zero: effecively a form of "same" zero padding
	buffer[ii][chan] = 0;
      }
    }
    
    typename CONFIG_T::accum_t int_accum[CONFIG_T::n_chan];

    #pragma HLS ARRAY_PARTITION variable=buffer complete
    #pragma HLS ARRAY_PARTITION variable=weights complete

    // NOTE: Currently we only output data after the kernel is full
    //         (ie: row >= CONFIG_T::y_filt-1)
    // NOTE UPDATE: Now, we output data with "same" zero padding on the left 
    // (0's are used if buffer is not full)
    // TODO: Find out what states get saved between runs!

    RowLoop:for(int row = 0; row < CONFIG_T::y_in; row++) {
        ChanLoop:for(int chan = 0; chan < CONFIG_T::n_chan; chan++){
	    // data_T val = data.read();
	    data_T val = data[row][chan];

            // std::cout << "Read " << val << std::endl;

            BuffLoop:for(int ii = 0; ii < CONFIG_T::y_filt; ii++) {
            #pragma HLS UNROLL
                // Shift operation for buffer
                buffer[ii][chan] = ii < CONFIG_T::y_filt - 1 ? buffer[ii + 1][chan] : val;
            }

            int_accum[chan] = 0;

            FiltLoop:for(int ii = 0; ii < CONFIG_T::y_filt; ii++){
            #pragma HLS UNROLL factor=4
                int_accum[chan] += buffer[ii][chan] * weights[ii][chan];
                // std::cout << "\tFilter/ChIn: " << ii << "/" << chan << ", Buffer: " << buffer[ii][chan] << std::endl;
                // std::cout << "\tAccum: " << int_accum[chan] << std::endl;
                // std::cout << "\tWeight: " << weights[ii][chan] << std::endl;
            }
            // When we hit the last filter sample, add bias term and output
	    //            if (row >= CONFIG_T::y_filt-1) {
	    // res << int_accum[chan] + biases[chan];
	    res[row][chan] = int_accum[chan] + biases[chan];
	    // std::cout << "\tResult: " << int_accum[jj] + biases[chan][jj]] << std::endl;
	    //            }
        }
    }
}

}

#endif
