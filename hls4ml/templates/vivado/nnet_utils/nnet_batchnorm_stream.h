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

#ifndef NNET_BATCHNORM_STREAM_H_
#define NNET_BATCHNORM_STREAM_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

// ****************************************************
//       Streaming Batch Normalization
// ****************************************************

template<class data_T, class res_T, typename CONFIG_T>
void normalize(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::scale_t scale[CONFIG_T::n_filt],
    typename CONFIG_T::bias_t  bias[CONFIG_T::n_filt]
) {
    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    int multiplier_limit = ceil(float(CONFIG_T::n_in) / float(CONFIG_T::reuse_factor));
    #pragma HLS ALLOCATION instances=product limit=multiplier_limit function

    BatchNormLoop: for (int i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        data_T in_data = data.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            int norm_index;
            if (CONFIG_T::n_filt==-1) {
                norm_index = i * data_T::size + j;
            } else {
                norm_index = j % CONFIG_T::n_filt;
            }
            out_data[j] = product<typename data_T::value_type, typename CONFIG_T::scale_t, typename res_T::value_type>(in_data[j], scale[norm_index])
                    + bias[norm_index];
        }

        res.write(out_data);
    }
}

}

#endif
