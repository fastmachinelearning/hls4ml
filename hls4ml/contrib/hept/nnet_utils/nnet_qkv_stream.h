// -----------------------------------------------------------------------------
// Vendored helper header for the HEPT hls4ml extension.
//
// Upstream source repository:
//   Howard1011/HEPT_HLS
//   https://github.com/Howard1011/HEPT_HLS/tree/main/HEPT/firmware/nnet_utils
// Upstream file URL at retrieval time:
//   https://raw.githubusercontent.com/Howard1011/HEPT_HLS/main/HEPT/firmware/nnet_utils/nnet_qkv_stream.h
// Retrieved on: 2026-03-30
//
// Keep this file in sync with upstream only after reviewing provenance and
// licensing for your intended use.
// -----------------------------------------------------------------------------

#ifndef NNET_QKV_SS_H_
#define NNET_QKV_SS_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "hls_stream.h"
#include <iostream>
#include <math.h>
#include "nnet_helpers.h"
#include "hls_streamofblocks.h"
//#include "nnet_activation.h"

namespace nnet {

struct qkv_config {
    static const unsigned num_head = 2;
    static const unsigned dim_per_head = 16;
    static const unsigned seq_len = 30;
    static const unsigned num_w_per_dist = 10;
    static const unsigned exp_table_size = 1024;
    static const unsigned coords_dim = 2;
};

template <typename CONFIG_T> 
void init_exp_table_qkv(typename CONFIG_T::exp_table_t table_out[CONFIG_T::exp_table_size])
{
    for (int ii = 0; ii < CONFIG_T::exp_table_size; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        double in_val = 2 * double(CONFIG_T::exp_range) * (ii - double(CONFIG_T::exp_table_size) / 2.0) / double(CONFIG_T::exp_table_size);
        // Next, compute lookup table function
        typename CONFIG_T::exp_table_t real_val = std::exp(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <typename CONFIG_T> 
void init_sqrt_table_qkv(typename CONFIG_T::sqrt_table_t table_out[CONFIG_T::sqrt_table_size])
{
    for (int ii = 0; ii < CONFIG_T::sqrt_table_size; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        double in_val = double(CONFIG_T::sqrt_range) * ii / double(CONFIG_T::sqrt_table_size);
        // Next, compute lookup table function
        if (in_val > 0.0)
            table_out[ii] = std::sqrt(in_val);
        else
            table_out[ii] = 0.0;
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << table_out[ii] << std::endl;
    }
}

template<typename CONFIG_T>
typename CONFIG_T::sqrt_table_t lookup_sqrt_qkv(
    typename CONFIG_T::accum_t data)
{
    #ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::sqrt_table_t sqrt_table[CONFIG_T::sqrt_table_size];
    #else
    static bool initialized = false;
    static typename CONFIG_T::sqrt_table_t sqrt_table[CONFIG_T::sqrt_table_size];
    #endif

    if (!initialized) {
        init_sqrt_table_qkv<CONFIG_T>(sqrt_table);
        initialized = true;
    }
    //std::cout << "fixed point data before: " << data << std::endl;
    int index = int(data*(CONFIG_T::sqrt_table_size/CONFIG_T::sqrt_range));
    //std::cout << "index: " << index << std::endl;
    if (index < 0)   index = 0;
    if (index > CONFIG_T::sqrt_table_size-1) index = CONFIG_T::sqrt_table_size-1;
    return sqrt_table[index];
}

template<typename CONFIG_T>
typename CONFIG_T::exp_table_t lookup_exp_qkv(
    typename CONFIG_T::accum_t data)
{
    #ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    #else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    #endif

    if (!initialized) {
        //init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_exp_table_qkv<CONFIG_T>(exp_table);
        initialized = true;
    }
    //std::cout << "fixed point data before: " << data << std::endl;
    int data_round = int(data*(CONFIG_T::exp_table_size/(CONFIG_T::exp_range*2)));
    //std::cout << "data_round: " << data_round << std::endl;
    //std::cout << "fixed point data: " << static_cast<typename CONFIG_T::accum_t>(data)*(CONFIG_T::exp_range*2)/CONFIG_T::exp_table_size << std::endl;
    int index = data_round + CONFIG_T::exp_range*(CONFIG_T::exp_table_size/(CONFIG_T::exp_range*2));
    //print index
    if (index < 0)   index = 0;
    if (index > CONFIG_T::exp_table_size-1) index = CONFIG_T::exp_table_size-1;
    return exp_table[index];
}

// num_head = 2;
// dim_per_head = 16;
// seq_len = 30;
template<class data_T, class res_T, typename CONFIG_T>
void compute_qkv(
    hls::stream<data_T>    data[16],
    hls::stream<res_T>     q[32],
    hls::stream<res_T>     k[32],
    hls::stream<res_T>     v[32],
    typename CONFIG_T::weight_t               weight_q[CONFIG_T::dim_per_head * CONFIG_T::dim_per_head * CONFIG_T::num_head],
    typename CONFIG_T::weight_t               weight_k[CONFIG_T::dim_per_head * CONFIG_T::dim_per_head * CONFIG_T::num_head],
    typename CONFIG_T::weight_t               weight_v[CONFIG_T::dim_per_head * CONFIG_T::dim_per_head * CONFIG_T::num_head])
{
    //#pragma HLS ARRAY_RESHAPE variable=weight   cyclic factor=CONFIG_T::dim_per_head*CONFIG_T::num_head dim=1
    //#pragma HLS ARRAY_RESHAPE variable=K cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    //#pragma HLS ARRAY_RESHAPE variable=V cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    //#pragma HLS ARRAY_RESHAPE variable=Q cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    //#pragma HLS ARRAY_RESHAPE variable=O cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    //#pragma HLS ARRAY_RESHAPE variable=row_buffer cyclic factor=tf_N*tf_T dim=1

    #pragma HLS INLINE
    #pragma HLS DATAFLOW

    typename CONFIG_T::accum_t row_buffer_q[CONFIG_T::dim_per_head * CONFIG_T::seq_len];
    typename CONFIG_T::accum_t row_buffer_k[CONFIG_T::dim_per_head * CONFIG_T::seq_len];
    typename CONFIG_T::accum_t row_buffer_v[CONFIG_T::dim_per_head * CONFIG_T::seq_len];

    // 30
    COMPUTE_QKV:
    for (int y = 0; y < CONFIG_T::seq_len; y++) {
        #pragma HLS pipeline II=1 rewind
        // 16
        COMPUTE_QKV_LOOP:
        for (int x = 0; x < CONFIG_T::dim_per_head; x++) {
            data_T data_pack;
            data_pack = data[x].read();
            // 32
            ROW_BUFFER_LOOP:
            for(int j = 0; j < CONFIG_T::dim_per_head * CONFIG_T::num_head; j++) {
                #pragma HLS UNROLL
                if(x == 0) {
                    row_buffer_q[j] = data_pack * weight_q[x * CONFIG_T::dim_per_head * CONFIG_T::num_head + j];
                    row_buffer_k[j] = data_pack * weight_k[x * CONFIG_T::dim_per_head * CONFIG_T::num_head + j];
                    row_buffer_v[j] = data_pack * weight_v[x * CONFIG_T::dim_per_head * CONFIG_T::num_head + j];
                }
                else {
                    row_buffer_q[j] += data_pack * weight_q[x * CONFIG_T::dim_per_head * CONFIG_T::num_head + j];
                    row_buffer_k[j] += data_pack * weight_k[x * CONFIG_T::dim_per_head * CONFIG_T::num_head + j];
                    row_buffer_v[j] += data_pack * weight_v[x * CONFIG_T::dim_per_head * CONFIG_T::num_head + j];
                }
            }
        }
        // 32
        WRITE_RESULT_LOOP:
        for(int i = 0; i < CONFIG_T::dim_per_head * CONFIG_T::num_head; i++) {
            res_T res_pack_q;
            res_T res_pack_k;
            res_T res_pack_v;
            res_pack_q = row_buffer_q[i];
            res_pack_k = row_buffer_k[i];
            res_pack_v = row_buffer_v[i];
            q[i].write(res_pack_q);
            k[i].write(res_pack_k);
            v[i].write(res_pack_v);
        }
    }
}

// num_head = 2;
// dim_per_head = 16;
// seq_len = 30;
// num_w_per_dist = 10;
template<class data_T, class data_T_coords, class res_T, typename CONFIG_T>
void prepare_qk(
    hls::stream<data_T>    q[32],
    hls::stream<data_T>    k[32],
    hls::stream<data_T_coords>    coords[2],
    hls::stream<res_T>     q_hat[36],
    hls::stream<res_T>     k_hat[36],
    typename CONFIG_T::weight_t               sqrt_w[4])
{
    //#pragma HLS ARRAY_RESHAPE variable=weight   cyclic factor=CONFIG_T::dim_per_head*CONFIG_T::num_head dim=1
    int res_size = 0;
    //#pragma HLS ARRAY_RESHAPE variable=K cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    //#pragma HLS ARRAY_RESHAPE variable=V cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    //#pragma HLS ARRAY_RESHAPE variable=Q cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    //#pragma HLS ARRAY_RESHAPE variable=O cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    //#pragma HLS ARRAY_RESHAPE variable=row_buffer cyclic factor=tf_N*tf_T dim=1

    #pragma HLS INLINE
    #pragma HLS DATAFLOW

    data_T data_pack_q;
    data_T data_pack_k;
    data_T_coords data_pack_coords;
    res_T res_pack_q;
    res_T res_pack_k;
    
    // sqrt_w * coords
    typename CONFIG_T::accum_t sqrt_w_r_coords[4][CONFIG_T::seq_len];

    COMPUTE_COORD_MUL:
    for(int i = 0; i < CONFIG_T::seq_len; i++)
    {
        #pragma HLS PIPELINE II=1 rewind
    }
    
    typename CONFIG_T::accum_t data_pack_tmp_q[CONFIG_T::num_head*(CONFIG_T::dim_per_head+2)];
    typename CONFIG_T::accum_t data_pack_tmp_k[CONFIG_T::num_head*(CONFIG_T::dim_per_head+2)];
    COORD_APPEND_RES:
    for(int i = 0; i < CONFIG_T::seq_len; i++) {
        #pragma HLS pipeline II=1 rewind
        for(int j = 0; j < CONFIG_T::num_head; j++) {
            for(int x = 0; x < CONFIG_T::dim_per_head; x++) {
                //std::cout << "index: " << j*(CONFIG_T::dim_per_head+2)+x << std::endl;
                data_pack_q = q[j * CONFIG_T::dim_per_head + x].read();
                data_pack_k = k[j * CONFIG_T::dim_per_head + x].read();
                q_hat[j * (CONFIG_T::dim_per_head + 2) + x].write(data_pack_q);
                k_hat[j * (CONFIG_T::dim_per_head + 2) + x].write(data_pack_k);
            }
            //std::cout << "index: " << j*(CONFIG_T::dim_per_head+2)+CONFIG_T::dim_per_head << std::endl;
            q_hat[j*(CONFIG_T::dim_per_head+2)+CONFIG_T::dim_per_head].write(sqrt_w_r_coords[j*2][i]);
            k_hat[j*(CONFIG_T::dim_per_head+2)+CONFIG_T::dim_per_head].write(sqrt_w_r_coords[j*2][i]);
            q_hat[j*(CONFIG_T::dim_per_head+2)+CONFIG_T::dim_per_head+1].write(sqrt_w_r_coords[j*2+1][i]);
            k_hat[j*(CONFIG_T::dim_per_head+2)+CONFIG_T::dim_per_head+1].write(sqrt_w_r_coords[j*2+1][i]);
            //std::cout << "16: " << sqrt_w_r_coords[j*2][i] << " 17: " << sqrt_w_r_coords[j*2+1][i] << std::endl;
        }
    }
}

// num_head = 2;
// dim_per_head = 16;
// seq_len = 30;
template<class data_T, class data_T_coords, class res_T, typename CONFIG_T>
void compute_qkv_pre_qk_merged(
    hls::stream<data_T>    data[CONFIG_T::par_factor*CONFIG_T::dim_per_head],
    hls::stream<data_T_coords>    coords[CONFIG_T::par_factor*CONFIG_T::coords_dim],
    hls::stream<res_T>     q[CONFIG_T::par_factor*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)*CONFIG_T::num_head],
    hls::stream<res_T>     k[CONFIG_T::par_factor*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)*CONFIG_T::num_head],
    hls::stream<res_T>     v[CONFIG_T::par_factor*CONFIG_T::dim_per_head * CONFIG_T::num_head],
    typename CONFIG_T::weight_t               weight_q[CONFIG_T::dim_per_head * CONFIG_T::dim_per_head * CONFIG_T::num_head],
    typename CONFIG_T::weight_t               weight_k[CONFIG_T::dim_per_head * CONFIG_T::dim_per_head * CONFIG_T::num_head],
    typename CONFIG_T::weight_t               weight_v[CONFIG_T::dim_per_head * CONFIG_T::dim_per_head * CONFIG_T::num_head],
    typename CONFIG_T::weight_t               sqrt_w[ CONFIG_T::num_head * CONFIG_T::coords_dim])
{
    //#pragma HLS ARRAY_RESHAPE variable=weight   cyclic factor=CONFIG_T::dim_per_head*CONFIG_T::num_head dim=1

    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW
    #pragma HLS bind_op op=mul impl=dsp
    #pragma HLS bind_op op=add impl=dsp


    #pragma HLS ARRAY_RESHAPE variable=weight_q type=complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=weight_k type=complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=weight_v type=complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=sqrt_w type=complete dim=1
    

    // 30
    COMPUTE_QKV:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS pipeline II=1 rewind
        for(int pf = 0; pf < CONFIG_T::par_factor; pf++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t row_buffer_q[CONFIG_T::dim_per_head * CONFIG_T::num_head];
            typename CONFIG_T::accum_t row_buffer_k[CONFIG_T::dim_per_head * CONFIG_T::num_head];
            typename CONFIG_T::accum_t row_buffer_v[CONFIG_T::dim_per_head * CONFIG_T::num_head];
            
            INIT_ROW_BUFFER_LOOP:
            for(int j = 0; j < CONFIG_T::dim_per_head * CONFIG_T::num_head; j++) {
                #pragma HLS UNROLL
                row_buffer_q[j] = 0;
                row_buffer_k[j] = 0;
                row_buffer_v[j] = 0;
            }
            
            typename CONFIG_T::accum_t sqrt_w_r_coords[CONFIG_T::coords_dim][CONFIG_T::num_head];
            COMPUTE_SQRT_W_R_COORDS:
            for(int j = 0; j < CONFIG_T::coords_dim; j++) {
                data_T_coords data_pack_coords;
                data_pack_coords = coords[pf*CONFIG_T::coords_dim+j].read();
                for(int x = 0; x < CONFIG_T::num_head; x++) {
                    sqrt_w_r_coords[j][x] = sqrt_w[x+j*CONFIG_T::num_head] * data_pack_coords;
                    // std::cout <<  j << " " << x <<" sqrt_w[x+j*CONFIG_T::num_head]: " << sqrt_w[x+j*CONFIG_T::num_head] << " data_pack_coords: "<< data_pack_coords << " sqrt_w_r_coords: " << sqrt_w_r_coords[j][x] << std::endl;
                }
            }
            // 16
            COMPUTE_QKV_LOOP:
            for (int x = 0; x < CONFIG_T::dim_per_head; x++) {
                data_T data_pack = data[pf*CONFIG_T::dim_per_head+x].read();
                // 32
                ROW_BUFFER_LOOP:
                for(int j = 0; j < CONFIG_T::dim_per_head * CONFIG_T::num_head; j++) {
                    #pragma HLS UNROLL
                    row_buffer_q[j] += data_pack * weight_q[x * CONFIG_T::dim_per_head * CONFIG_T::num_head + j];
                    row_buffer_k[j] += data_pack * weight_k[x * CONFIG_T::dim_per_head * CONFIG_T::num_head + j];
                    row_buffer_v[j] += data_pack * weight_v[x * CONFIG_T::dim_per_head * CONFIG_T::num_head + j];
                }
            }
            // 32
            WRITE_RESULT_LOOP:
            for(int j = 0; j < CONFIG_T::num_head; j++) {
                for(int x = 0; x < CONFIG_T::dim_per_head; x++) {
                    res_T res_pack_q = row_buffer_q[j * CONFIG_T::dim_per_head + x];
                    res_T res_pack_k = row_buffer_k[j * CONFIG_T::dim_per_head + x];
                    res_T res_pack_v = row_buffer_v[j * CONFIG_T::dim_per_head + x];
                    q[pf*(CONFIG_T::dim_per_head + CONFIG_T::coords_dim)*CONFIG_T::num_head+j * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim) + x].write(res_pack_q);
                    k[pf*(CONFIG_T::dim_per_head + CONFIG_T::coords_dim)*CONFIG_T::num_head+j * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim) + x].write(res_pack_k);
                    v[pf*CONFIG_T::dim_per_head*CONFIG_T::num_head+j * CONFIG_T::dim_per_head + x].write(res_pack_v);
                }
                for(int x = 0; x < CONFIG_T::coords_dim; x++) {
                    res_T res_pack_q = sqrt_w_r_coords[x][j];
                    res_T res_pack_k = sqrt_w_r_coords[x][j];
                    // std::cout << "sqrt_w_r_coords[x][j] " << x << " " << j << ": " << sqrt_w_r_coords[x][j] << std::endl;
                    q[pf*(CONFIG_T::dim_per_head + CONFIG_T::coords_dim)*CONFIG_T::num_head+j * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim) + CONFIG_T::dim_per_head + x].write(res_pack_q);
                    k[pf*(CONFIG_T::dim_per_head + CONFIG_T::coords_dim)*CONFIG_T::num_head+j * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim) + CONFIG_T::dim_per_head + x].write(res_pack_k);
                }
            }
        }
    }
}


}

#endif
