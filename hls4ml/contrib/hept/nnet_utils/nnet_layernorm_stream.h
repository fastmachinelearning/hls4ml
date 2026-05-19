// -----------------------------------------------------------------------------
// Vendored helper header for the HEPT hls4ml extension.
//
// Upstream source repository:
//   Howard1011/HEPT_HLS
//   https://github.com/Howard1011/HEPT_HLS/tree/main/HEPT/firmware/nnet_utils
// Upstream file URL at retrieval time:
//   https://raw.githubusercontent.com/Howard1011/HEPT_HLS/main/HEPT/firmware/nnet_utils/nnet_layernorm_stream.h
// Retrieved on: 2026-03-30
//
// Keep this file in sync with upstream only after reviewing provenance and
// licensing for your intended use.
// -----------------------------------------------------------------------------

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

#ifndef NNET_LAYERNORM_SINGLE_STREAM_H_
#define NNET_LAYERNORM_SINGLE_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "hls_stream.h"
#include <math.h>
#include <iostream>
#include <iomanip>
#include "hls_math.h"

namespace nnet {

struct layernorm_config {
    static const unsigned seq_len = 30;
    static const unsigned embed_dim = 16;
    static const unsigned inv_sqrt_table_size = 1024;
    static const unsigned inv_sqrt_range = 16;
};

struct layernorm_origin_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static constexpr double inv_sqrt_range = 1;
    static const unsigned log_table_range = 10;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
};

template<typename CONFIG_T, int N_TABLE>
void init_n_invert_sqr_table(typename CONFIG_T::accum_t table_out[N_TABLE])
{
    //float inv_range = CONFIG_T::inv_sqrt_range;
    // Inversion function:
    //   result = 1/sqrt(x)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +0.01)
        double in_val;
        if (N_TABLE > CONFIG_T::inv_sqrt_range) {
            in_val = ii/typename CONFIG_T::accum_t(N_TABLE / CONFIG_T::inv_sqrt_range);
        } else {
            in_val = ii*typename CONFIG_T::accum_t(CONFIG_T::inv_sqrt_range / N_TABLE);
        }
        //float in_val = ii/float(N_TABLE / CONFIG_T::inv_sqrt_range);
        // Next, compute lookup table function
        if (in_val > 0.0) table_out[ii] = typename CONFIG_T::accum_t(1.0/sqrt(in_val));
        else table_out[ii] = typename CONFIG_T::accum_t(1024); // a large number
        // std::cout << "table_out[" << ii << "]: " << table_out[ii] << std::endl;
    }
}

template<typename CONFIG_T, int N_TABLE>
typename CONFIG_T::accum_t lookup_n_invert_sqr_table(
    typename CONFIG_T::accum_t variance)
{
    #ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::accum_t var_table[N_TABLE];
    #else
    static bool initialized = false;
    static typename CONFIG_T::accum_t var_table[N_TABLE];
    #endif

    if (!initialized) {
        init_n_invert_sqr_table<CONFIG_T, N_TABLE>(var_table);
        initialized = true;
    }

    int index;

    if (N_TABLE > CONFIG_T::inv_sqrt_range) {
        index = int(variance * (N_TABLE / CONFIG_T::inv_sqrt_range));
    } else {
        index = int(variance / (CONFIG_T::inv_sqrt_range / N_TABLE));
    }

    if (index < 0) index = 0;
    if (index >= N_TABLE) index = N_TABLE - 1;

    // std::cout << "index: " << index << " value: " << var_table[index] << std::endl;

    return var_table[index];
}


template<class data_T, class res_T, typename CONFIG_T>
void LayerNormalize_origin(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::scale_t  scale[CONFIG_T::embed_dim],
    typename CONFIG_T::bias_t   bias[CONFIG_T::embed_dim]
)
{
    float in_val[CONFIG_T::seq_len*CONFIG_T::embed_dim];
    float outval[CONFIG_T::seq_len*CONFIG_T::embed_dim];
    #pragma HLS DATAFLOW
    #pragma HLS BIND_STORAGE variable=in_val type=ram_s2p impl=bram
    #pragma HLS BIND_STORAGE variable=outval type=ram_s2p impl=bram

    const unsigned int tf_T = CONFIG_T::tiling_factor[0];
    const unsigned int tf_N = CONFIG_T::tiling_factor[1];
    const unsigned int T = CONFIG_T::seq_len/tf_T;
    const unsigned int N = CONFIG_T::embed_dim/tf_N;
    #pragma HLS ARRAY_RESHAPE variable=in_val cyclic factor=tf_N*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=outval cyclic factor=tf_N*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=scale cyclic factor=tf_N dim=1
    #pragma HLS ARRAY_RESHAPE variable=bias cyclic factor=tf_N dim=1
    
    float dim_inv = 1.0/CONFIG_T::embed_dim;
    float xsqrsum_1[CONFIG_T::tiling_factor[0]];
    float xsum_1[CONFIG_T::tiling_factor[0]];
    float prev_xsum_1[CONFIG_T::tiling_factor[0]];
    float xsqrsum_2[CONFIG_T::tiling_factor[0]];
    float xsum_2[CONFIG_T::tiling_factor[0]];
    float prev_xsum_2[CONFIG_T::tiling_factor[0]];
    float xsum[CONFIG_T::tiling_factor[0]];
    float xsqrsum[CONFIG_T::tiling_factor[0]];
    float xmean[CONFIG_T::tiling_factor[0]];
    float row_buffer[CONFIG_T::embed_dim*CONFIG_T::tiling_factor[0]];
    bool write_buffer1[tf_T];
    float deno_inver[tf_T];
    
    for (int jj=0; jj < tf_T; ++jj){
        write_buffer1[jj] = true;
    }

    #pragma HLS ARRAY_PARTITION variable=xsqrsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=deno_inver complete dim=1
    #pragma HLS ARRAY_PARTITION variable=write_buffer1 complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=row_buffer cyclic factor=tf_N*tf_T dim=1
    
    float tmp;
    data_T data_pack;
    res_T res_pack;
    float norm_tmp;

    layerNorm:  for (int j=0; j <= T; ++j){
                    for (int i=0; i < N; ++i){
                        #pragma HLS PIPELINE
                        if (j < T) {
                            data_pack = data.read();
                        }
                        for (int jj=0; jj < tf_T; ++jj){
                            #pragma HLS UNROLL
                            if (i == 0){
                                if (write_buffer1[jj] == true){
                                    xsqrsum_1[jj] = 0;
                                    xsum_1[jj] = 0;
                                } else {
                                    xsqrsum_2[jj] = 0;
                                    xsum_2[jj] = 0;
                                }
                            }
                            for (int ii=0; ii < tf_N; ++ii){
                                #pragma HLS UNROLL
                                if (j < T){
                                    tmp = (float)data_pack[jj*tf_N+ii];
                                    float tmp2 = tmp*tmp*dim_inv;
                                    if (write_buffer1[jj] == true){
                                        xsum_1[jj] = xsum_1[jj] + tmp;
                                        xsqrsum_1[jj] = xsqrsum_1[jj] + tmp2;
                                    } else {
                                        xsum_2[jj] = xsum_2[jj] + tmp;
                                        xsqrsum_2[jj] = xsqrsum_2[jj] + tmp2;
                                    }
                                }
                                if (j > 0){
                                    if (write_buffer1[jj] == false){
                                        xsum[jj] = xsum_1[jj];
                                    } else {
                                        xsum[jj] = xsum_2[jj];
                                    }
                                    xmean[jj] = xsum[jj]*dim_inv;
                                    norm_tmp = (row_buffer[i*tf_T*tf_N + ii*tf_T + jj] - xmean[jj])*deno_inver[jj];
                                    float scale_float = (float)scale[i*tf_N + ii];
                                    float bias_float = (float)bias[i*tf_N + ii];
                                    res_pack[jj*tf_N+ii] = (typename res_T::value_type)(norm_tmp*scale_float + bias_float);
                                }
                                if (j < T){
                                    row_buffer[i*tf_T*tf_N + ii*tf_T + jj] = tmp;
                                }
                            }
                            if (i == (N-1)){
                                write_buffer1[jj] = !write_buffer1[jj];
                                if (write_buffer1[jj] == false){
                                    xsqrsum[jj] = xsqrsum_1[jj];
                                    xsum[jj] = xsum_1[jj];
                                } else {
                                    xsqrsum[jj] = xsqrsum_2[jj];
                                    xsum[jj] = xsum_2[jj];
                                }
                                xmean[jj] = xsum[jj]*dim_inv;
                                float var = xsqrsum[jj]-xmean[jj]*xmean[jj];
                                deno_inver[jj] = 1.0f/sqrtf(var + 1e-5f);
                            }
                        }
                        if (j > 0){
                           res.write(res_pack);
                        }
                    }
                }
}


template<class data_T, class res_T, typename CONFIG_T>
void LayerNormalize(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::scale_t  scale[CONFIG_T::embed_dim],
    typename CONFIG_T::bias_t   bias[CONFIG_T::embed_dim]
) {
    typename CONFIG_T::accum_t in_val[CONFIG_T::seq_len * CONFIG_T::embed_dim];
    res_T res_pack;
    data_T data_pack;
    
    #pragma HLS BIND_STORAGE variable=in_val type=ram_s2p impl=bram

    typename CONFIG_T::accum_t dim_inv = 1.0 / CONFIG_T::embed_dim;
    OUTER_LOOP:
    for (int j = 0; j < CONFIG_T::seq_len; j++) {
        #pragma HLS PIPELINE II=1 rewind
        typename CONFIG_T::accum_t xsum = 0, xsqrsum = 0;
        typename CONFIG_T::accum_t tmp;
        READ_CAL_LOOP:
        for (int i = 0; i < CONFIG_T::embed_dim / data_T::size; i++) {
            //#pragma HLS PIPELINE
            data_pack = data.read();
            READ_CAL_INTER_LOOP:
            for (int k = 0; k < data_T::size; k++) {
                #pragma HLS UNROLL
                tmp = (typename CONFIG_T::accum_t)data_pack[k];
                in_val[j * CONFIG_T::embed_dim + i * data_T::size + k] = tmp;
                xsum += tmp;
                xsqrsum += tmp * tmp;
            }
        }

        typename CONFIG_T::accum_t mean = xsum * dim_inv;
        typename CONFIG_T::accum_t variance = xsqrsum * dim_inv - mean * mean;
        typename CONFIG_T::accum_t deno_inv = lookup_n_invert_sqr_table<CONFIG_T, CONFIG_T::inv_sqrt_table_size>(variance);
        WRITE_CAL_OUTER_LOOP:
        for (int i = 0; i < CONFIG_T::embed_dim / res_T::size; i++) {
            #pragma HLS UNROLL
            WRITE_CAL_LOOP:
            for(int k = 0; k < res_T::size; k++) {
                #pragma HLS UNROLL
                typename CONFIG_T::accum_t norm_tmp = (in_val[j * CONFIG_T::embed_dim + i * res_T::size + k] - mean) * deno_inv;
                res_pack[k] = (typename res_T::value_type)(norm_tmp * (typename CONFIG_T::accum_t)scale[i * res_T::size + k] + (typename CONFIG_T::accum_t)bias[i * res_T::size + k]);
            }
            res.write(res_pack);
        }
    }
}
template<class data_T, class res_T, typename CONFIG_T>
void LayerNormalize(
    hls::stream<data_T>    data[CONFIG_T::par_factor*CONFIG_T::embed_dim],
    hls::stream<res_T>     res[CONFIG_T::par_factor*CONFIG_T::embed_dim],
    typename CONFIG_T::scale_t  scale[CONFIG_T::embed_dim],
    typename CONFIG_T::bias_t   bias[CONFIG_T::embed_dim]
) {
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW

    typename CONFIG_T::accum_t in_val[CONFIG_T::embed_dim];
    #pragma HLS ARRAY_RESHAPE variable=in_val complete dim=1
    #pragma HLS BIND_STORAGE variable=in_val type=ram_1p impl=lutram

    typename CONFIG_T::accum_t dim_inv = 1.0 / CONFIG_T::embed_dim;

    OUTER_LOOP:
    for (int j = 0; j < CONFIG_T::seq_len / CONFIG_T::par_factor; j++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int pf = 0; pf < CONFIG_T::par_factor; pf++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t xsum = 0;
            typename CONFIG_T::accum_t xsqrsum = 0;
            typename CONFIG_T::accum_t tmp;

            READ_CAL_LOOP:
            for (int i = 0; i < CONFIG_T::embed_dim; i++) {
                #pragma HLS UNROLL
                data_T data_pack = data[pf*CONFIG_T::embed_dim+i].read();
                tmp = (typename CONFIG_T::accum_t)data_pack;
                in_val[i] = tmp;
                xsum += data_pack;
                xsqrsum += data_pack * data_pack;
            }

            typename CONFIG_T::accum_t mean = xsum * dim_inv;
            typename CONFIG_T::accum_t variance = xsqrsum * dim_inv - mean * mean;
            typename CONFIG_T::accum_t deno_inv = lookup_n_invert_sqr_table<CONFIG_T, CONFIG_T::inv_sqrt_table_size>(variance);

        
            WRITE_CAL_LOOP:
            for (int i = 0; i < CONFIG_T::embed_dim; i++) {
                #pragma HLS UNROLL
                typename CONFIG_T::accum_t norm_tmp = (in_val[i] - mean) * deno_inv;
                res_T res_pack = (res_T)(norm_tmp * (typename CONFIG_T::accum_t)scale[i] + (typename CONFIG_T::accum_t)bias[i]);
                res[pf*CONFIG_T::embed_dim+i].write(res_pack);
            }
        }
    }
}



}

#endif