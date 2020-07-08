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

#ifndef NNET_BATCHNORM_H_
#define NNET_BATCHNORM_H_

#include "nnet_common.h"

namespace nnet {

struct batchnorm_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float scale_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_filt = -1;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
};

template<class data_T, class res_T, typename CONFIG_T>
void normalize(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_in],
    const typename CONFIG_T::scale_t  scale[CONFIG_T::n_in],
    const typename CONFIG_T::bias_t   bias[CONFIG_T::n_in]
)
{
    // Calcuate result
    Result:
    #pragma unroll
    for (int ires = 0; ires < CONFIG_T::n_in; ires++) {
        if (CONFIG_T::n_filt==-1) {
            res[ires] = data[ires] * scale[ires] + bias[ires];
	      }
        else {
            int norm_index = ires%CONFIG_T::n_filt;
            res[ires] = data[ires] * scale[norm_index] + bias[norm_index];
        }
	}
}

// ****************************************************
//       Merged Batch Normalization and Quantized Tanh
// ****************************************************
struct batchnorm_quantized_tanh_config
{
    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_filt = -1;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
};

template<class data_T, typename CONFIG_T>
void  normalize_binary_tanh(data_T data[CONFIG_T::n_in], ac_int<1, false> res[CONFIG_T::n_in], const data_T threshold[CONFIG_T::n_in])
{
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        ac_int<1, false> cache;
        data_T datareg = data[ii];
        if( datareg > threshold[ii] ) cache = 1;
        else cache = 0;

        res[ii] = (ac_int<1, false>) cache;
    }
}

template<class data_T, typename CONFIG_T>
void  normalize_ternary_tanh(data_T data[CONFIG_T::n_in], ac_int<2, true> res[CONFIG_T::n_in], const data_T threshold_hi[CONFIG_T::n_in], const data_T threshold_lo[CONFIG_T::n_in])
{
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        ac_int<2, true> cache;
        data_T datareg = data[ii];
        if( datareg > threshold_hi[ii] ) cache = 1;
        else if( datareg <= threshold_lo[ii]) cache = -1;
        else cache = 0;
        res[ii] = (ac_int<2, true>) cache;
    }
}

}

#endif
