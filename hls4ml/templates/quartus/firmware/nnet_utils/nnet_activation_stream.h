#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"

namespace nnet{

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(stream<data_T> &data, stream<res_T> &res) {
    #include "activation_tables/exp_table.tb"
    #include "activation_tables/invert_table.tb"

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;

    hls_register typename data_T::value_type data_array[data_T::size];
    
    SoftmaxArrayLoop: 
    #pragma ii pipeline
    for(unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {    
        data_T in_pack = data.read();
        
        SoftmaxArrayPackLoop: 
        #pragma unroll 
        for(unsigned j = 0; j < data_T::size; j++) {
            data_array[j] = in_pack[j];
        }

        // Find the max and compute all delta(x_i, x_max)
        Op_max<typename data_T::value_type> op_max;
        hls_register typename data_T::value_type x_max = reduce<typename data_T::value_type, data_T::size, Op_max<typename data_T::value_type>>(data_array, op_max);

        // For the diffs, use the same type as the input but force rounding and saturation
        hls_register ac_fixed<data_T::value_type::width, data_T::value_type::i_width, true, AC_RND, AC_SAT> d_xi_xmax[data_T::size];
        #pragma unroll
        for(unsigned j = 0; j < data_T::size; j++){
            d_xi_xmax[j] = data_array[j] - x_max;
        }

        // Calculate all the e^x's
        hls_register typename CONFIG_T::exp_table_t exp_res[data_T::size];
        #pragma unroll
        for(unsigned j = 0; j < data_T::size; j++) {
            exp_res[j] = exp_table[softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T>(d_xi_xmax[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        hls_register typename CONFIG_T::exp_table_t exp_sum = reduce<typename CONFIG_T::exp_table_t, data_T::size, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        hls_register typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];
        res_T out_pack;
        
        SoftmaxInvPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < res_T::size; j++){
            
            // TODO - Find Quartus-equivalent pragma
            // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(stream<data_T> &data, stream<res_T> &res){
    #include "activation_tables/exp_table_latency.tb"
    #include "activation_tables/invert_table_latency.tb"
    
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;

    // Calculate all the e^x's
    hls_register typename CONFIG_T::exp_table_t exp_res[data_T::size];
    
    SoftmaxExpLoop: 
    #pragma ii pipeline
    for(unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        data_T in_pack = data.read();
        
        SoftmaxExpPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < data_T::size; j++) {
            exp_res[j] = exp_table_latency[softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T>(in_pack[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        hls_register typename CONFIG_T::exp_table_t exp_sum = reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        // Multiply previously calculated exponetials with the reciprocal of the sum
        hls_register typename CONFIG_T::inv_table_t inv_exp_sum = invert_table_latency[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];

        res_T out_pack;
        SoftmaxInvPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < res_T::size; j++){
            // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }

        res.write(out_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(stream<data_T> &data, stream<res_T> &res) {
    #include "activation_tables/exp_table_legacy.tb"
    #include "activation_tables/invert_table_legacy.tb"
    
    // Index into the lookup table based on data for exponentials
    hls_register typename CONFIG_T::table_t exp_res[data_T::size];
    hls_register typename CONFIG_T::table_t exp_diff_res;
    hls_register typename data_T::value_type data_cache[data_T::size];

    SoftmaxInitLoop: 
    #pragma ii 1
    for(unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        data_T in_pack = data.read();
        
        SoftmaxInitPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < data_T::size; j++) {
            data_cache[j] = in_pack[j];
            exp_res[j] = 0;
        }

        SoftmaxExpLoop: 
        #pragma unroll
        for (int i = 0; i < data_T::size; i++) {
            SoftmaxExpInner: 
            #pragma unroll
            for (int j = 0; j < data_T::size; j++) {
                if (i == j) {
                    exp_diff_res = 1;
                } else {
                    int data_round = ((data_cache[j] - data_cache[i])*CONFIG_T::table_size/16).to_int();
                    int index = data_round + 8 * CONFIG_T::table_size / 16;
                    if (index < 0) index = 0;
                    if (index > CONFIG_T::table_size - 1) index = CONFIG_T::table_size - 1;
                    exp_diff_res = exp_table_legacy[index];
                }
                exp_res[i] += exp_diff_res;
            }
        }

        res_T out_pack;
        SoftmaxInvPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < res_T::size; j++) {
            int exp_res_index = (exp_res[j]*CONFIG_T::table_size/64).to_int();
            if (exp_res_index < 0) exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1) exp_res_index = CONFIG_T::table_size - 1;
            out_pack[j] = (typename res_T::value_type) invert_table_legacy[exp_res_index];
        }

        res.write(out_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void softmax(stream<data_T> &data, stream<res_T> &res) {
    switch(CONFIG_T::implementation) {
        case softmax_implementation::latency:
            softmax_latency<data_T, res_T, CONFIG_T>(data, res);
            break;
        case softmax_implementation::stable:
            softmax_stable<data_T, res_T, CONFIG_T>(data, res);
            break;
        case softmax_implementation::legacy:
            softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
            break;
        default:
            softmax_stable<data_T, res_T, CONFIG_T>(data, res);
            break;
    }    
}

}

#endif