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
#include "hls_stream.h"
#include <math.h>

namespace nnet {

// ****************************************************
//       Streaming Batch Normalization
// ****************************************************

template<class data_T, class res_T, typename CONFIG_T>
void normalize_no_filt(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   bias[CONFIG_T::n_filt]
)
{
    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    BatchNormLoop: for (int i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data[j] * scale[i * data_T::size + j] + bias[i * data_T::size + j];
        }

        res.write(out_data);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void normalize_filt(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   bias[CONFIG_T::n_filt]
)
{
    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    BatchNormLoop: for (int i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_part = data.read();
        res_T out_part;
        #pragma HLS DATA_PACK variable=out_part

        NormFiltLoop: for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            int norm_index = j % CONFIG_T::n_filt;
            out_part.data[j] = in_part.data[j] * scale[norm_index] + bias[norm_index];
        }
        res.write(out_part);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void normalize(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   bias[CONFIG_T::n_filt]
)
{
    if (CONFIG_T::n_filt==-1) {
        normalize_no_filt<data_T, res_T, CONFIG_T>(data, res, scale, bias);
	} else {
        normalize_filt<data_T, res_T, CONFIG_T>(data, res, scale, bias);
    }
}

}

#endif
