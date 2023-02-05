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

// Change History:
//   2022-06-30  dgburnette - Cleaned up code to separate AC Math from LUT code.
//                            Activation functions not implemented in AC Math will assert.
//   2022-06-28  dgburnette - Replaced AP Types with AC Datatypes. 

#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include <cmath>
#include "ac_fixed.h"
#include "ac_channel.h"
#include <ac_std_float.h>
#include <ac_math/ac_softmax_pwl.h>
#include <ac_math/ac_tanh_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include "nnet_common.h"
#include "nnet_types.h"
#include "nnet_stream.h"
#include "nnet_activation.h"

namespace nnet {


// *************************************************
//       LINEAR Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void linear(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    #pragma hls_pipeline_init_interval 1
    LinearActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        LinearPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            out_data[j] = in_data[j];
        }

        res.write(out_data);
    }
}


// *************************************************
//       RELU Activation
// *************************************************
#pragma hls_design block
template<class data_T, class res_T, typename CONFIG_T>
void relu(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    #pragma hls_pipeline_init_interval 1
    ReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        ReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if (in_data[j] > 0) out_data[j] = in_data[j];
            else out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************

#ifndef USE_AC_MATH

template<class data_T, class res_T, typename CONFIG_T>
void sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(sigmoid_table);
        initialized = true;
    }

    #pragma hls_pipeline_init_interval 1
    SigmoidActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        SigmoidPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            int data_round = in_data[j]*CONFIG_T::table_size/16;
            int index = data_round + 8*CONFIG_T::table_size/16;
            if (index < 0)   index = 0;
            else if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            out_data[j] = sigmoid_table[index];
        }

        res.write(out_data);
    }
}

#else

template<class data_T, class res_T, typename CONFIG_T>
void sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    SigmoidActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
        #pragma hls_unroll
        SigmoidPackLoop: for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_sigmoid_pwl(in_data[j],out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       Softmax Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(ac_channel<data_T> &data, ac_channel<res_T> &res){
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename data_T::value_type, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned ii = data_T::size / multiplier_limit;

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[data_T::size];
    //#pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);

    if constexpr(ii==1) {
      #pragma hls_pipeline_init_interval 1
    }
    if constexpr(ii!=1) {
      // future enhancement for Catapult
      #pragma hls_pipeline_init_interval ii
    }
    SoftmaxExpLoop: for(unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++){
        //#pragma HLS PIPELINE II=ii

        data_T in_pack = data.read();
        #pragma hls_unroll
        SoftmaxExpPackLoop: for(unsigned j = 0; j < data_T::size; j++){
            //#pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T>(in_pack[j]);
            exp_res[j] = exp_table[x];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        exp_sum = reduce<typename CONFIG_T::exp_table_t, data_T::size, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];

        res_T out_pack;
        //#pragma HLS DATA_PACK variable=out_pack
        #pragma hls_unroll
        SoftmaxInvPackLoop: for(unsigned j = 0; j < res_T::size; j++){
            //#pragma HLS UNROLL
            //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(ac_channel<data_T> &data, ac_channel<res_T> &res){
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename data_T::value_type, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned ii = data_T::size / multiplier_limit;

    typename data_T::value_type data_array[data_T::size];
    //#pragma HLS ARRAY_PARTITION variable=data_array complete

    if constexpr(ii==1) {
      #pragma hls_pipeline_init_interval 1
    }
    if constexpr(ii!=1) {
      // future enhancement for Catapult
      #pragma hls_pipeline_init_interval ii
    }
    SoftmaxArrayLoop: for(unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++){
        //#pragma HLS PIPELINE II=ii

        data_T in_pack = data.read();
        #pragma hls_unroll
        SoftmaxArrayPackLoop: for(unsigned j = 0; j < data_T::size; j++){
            //#pragma HLS UNROLL
            data_array[j] = in_pack[j];
        }

        // Find the max and compute all delta(x_i, x_max)
        Op_max<typename data_T::value_type> op_max;
        typename data_T::value_type x_max = reduce<typename data_T::value_type, data_T::size, Op_max<typename data_T::value_type>>(data_array, op_max);

        // For the diffs, use the same type as the input but force rounding and saturation
        ac_fixed<data_T::value_type::width, data_T::value_type::iwidth,true,AC_RND,AC_SAT> d_xi_xmax[data_T::size];
        #pragma hls_unroll
        for(unsigned j = 0; j < data_T::size; j++){
            //#pragma HLS UNROLL
            d_xi_xmax[j] = data_array[j] - x_max;
        }

        // Calculate all the e^x's
        typename CONFIG_T::exp_table_t exp_res[data_T::size];
        //#pragma HLS ARRAY_PARTITION variable=exp_res complete
        typename CONFIG_T::exp_table_t exp_sum(0);
        #pragma hls_unroll
        for(unsigned j = 0; j < data_T::size; j++){
            //#pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T>(d_xi_xmax[j]);
            exp_res[j] = exp_table[x];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        exp_sum = reduce<typename CONFIG_T::exp_table_t, data_T::size, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];

        res_T out_pack;
        //#pragma HLS DATA_PACK variable=out_pack
        #pragma hls_unroll
        SoftmaxInvPackLoop: for(unsigned j = 0; j < res_T::size; j++){
            //#pragma HLS UNROLL
            //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_invert_table_legacy<CONFIG_T, CONFIG_T::table_size>(invert_table);
        initialized = true;
    }

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[data_T::size];
    typename CONFIG_T::table_t exp_diff_res;
    typename data_T::value_type data_cache[data_T::size];

    #pragma hls_pipeline_init_interval 1
    SoftmaxInitLoop: for(unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        //#pragma HLS PIPELINE
        data_T in_pack = data.read();
        #pragma hls_unroll
        SoftmaxInitPackLoop: for(unsigned j = 0; j < data_T::size; j++) {
            //#pragma HLS UNROLL
            data_cache[j] = in_pack[j];
            exp_res[j] = 0;
        }

        #pragma hls_unroll
        SoftmaxExpLoop: for (int i = 0; i < data_T::size; i++) {
            //#pragma HLS UNROLL
            #pragma hls_unroll
            SoftmaxExpInner: for (int j = 0; j < data_T::size; j++) {
                //#pragma HLS UNROLL

                if (i == j) {
                    exp_diff_res = 1;
                } else {
                    int data_round = (data_cache[j] - data_cache[i]) * CONFIG_T::table_size / 16;
                    int index = data_round + 8 * CONFIG_T::table_size / 16;
                    if (index < 0) index = 0;
                    if (index > CONFIG_T::table_size - 1) index = CONFIG_T::table_size - 1;
                    exp_diff_res = exp_table[index];
                }

                exp_res[i] += exp_diff_res;
            }
        }

        res_T out_pack;
        //#pragma HLS DATA_PACK variable=out_pack
        #pragma hls_unroll
        SoftmaxInvPackLoop: for(unsigned j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL

            int exp_res_index = exp_res[j] * CONFIG_T::table_size / 64;
            if (exp_res_index < 0) exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1) exp_res_index = CONFIG_T::table_size - 1;

            out_pack[j] = (typename res_T::value_type) invert_table[exp_res_index];
        }
        res.write(out_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void softmax(ac_channel<data_T> &data, ac_channel<res_T> &res){
    assert(CONFIG_T::axis == -1);

    switch(CONFIG_T::implementation){
    case softmax_implementation::latency:
        softmax_latency<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::stable:
        softmax_stable<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::legacy:
        softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
        break;
    }    
}

#else

template<class data_T, class res_T, typename CONFIG_T>
void softmax(ac_channel<data_T> &data, ac_channel<res_T> &res)
{
    typename data_T::value_type data_cache[data_T::size];
    typename res_T::value_type res_cache[data_T::size];
    #pragma hls_pipeline_init_interval 1
    SoftmaxInitLoop: for(unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        data_T in_pack = data.read();

        #pragma hls_unroll
        SoftmaxInitPackLoop: for(unsigned j = 0; j < data_T::size; j++) { data_cache[j] = in_pack[j]; }

        res_T out_pack;
		  ac_math::ac_softmax_pwl(data_cache,res_cache);

        #pragma hls_unroll
        SoftmaxResPackLoop: for(unsigned j = 0; j < data_T::size; j++) { out_pack[j] = res_cache[j]; }

        res.write(out_pack);
    }
}

#endif


// *************************************************
//       TanH Activation
// *************************************************

#ifndef USE_AC_MATH

template<class data_T, class res_T, typename CONFIG_T>
void tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(tanh_table);
        initialized = true;
    }

    #pragma hls_pipeline_init_interval 1
    TanHActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        TanHPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            int data_round = in_data[j]*CONFIG_T::table_size/8;
            int index = data_round + 4*CONFIG_T::table_size/8;
            if (index < 0)   index = 0;
            else if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            out_data[j] = tanh_table[index];
        }

        res.write(out_data);
    }
}

#else

template<class data_T, class res_T, typename CONFIG_T>
void tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    #pragma hls_pipeline_init_interval 1
    TanHActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;
        #pragma hls_unroll
        TanHPackLoop: for (int j = 0; j < res_T::size; j++) {
            int data_round = in_data[j]*CONFIG_T::table_size/8;
            ac_math::ac_tanh_pwl(in_data[j],out_data[j]);
        }
        res.write(out_data);
    }
}

#endif


// *************************************************
//       Hard sigmoid Activation
// *************************************************

template<class data_T, class res_T, typename CONFIG_T>
void hard_sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    typename data_T::value_type slope = (typename data_T::value_type) 0.2;
    typename data_T::value_type shift = (typename data_T::value_type) 0.5;

    #pragma hls_pipeline_init_interval 1
    HardSigmoidActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        HardSigmoidPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            typename data_T::value_type datareg = slope * in_data[j] + shift;
            if (datareg > 1) datareg = 1;
            else if (datareg < 0) datareg = 0;
            out_data[j] = datareg;
        }

        res.write(out_data);
    }
}


// *************************************************
//       Leaky RELU Activation
// *************************************************

template<class data_T, class res_T, typename CONFIG_T>
void leaky_relu(ac_channel<data_T> &data, typename data_T::value_type alpha, ac_channel<res_T> &res) {
    #pragma hls_pipeline_init_interval 1
    LeakyReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        LeakyReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if (in_data[j] > 0) out_data[j] = in_data[j];
            else out_data[j] = alpha * in_data[j];
        }
        res.write(out_data);
    }
}


// *************************************************
//       Thresholded RELU Activation
// *************************************************

template<class data_T, class res_T, typename CONFIG_T>
void thresholded_relu(ac_channel<data_T> &data, typename data_T::value_type theta, ac_channel<res_T> &res) {
    #pragma hls_pipeline_init_interval 1
    ThresholdedReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        ThresholdedReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if (in_data[j] > theta) out_data[j] = in_data[j];
            else out_data[j] = 0;
        }

        res.write(out_data);
    }
}


// *************************************************
//       Softplus Activation
// *************************************************

#ifndef USE_AC_MATH

template<class data_T, class res_T, typename CONFIG_T>
void softplus(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softplus_table<CONFIG_T, CONFIG_T::table_size>(softplus_table);
        initialized = true;
    }

    #pragma hls_pipeline_init_interval 1
    SoftplusActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        SoftplusPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            int data_round = in_data[j]*CONFIG_T::table_size/16;
            int index = data_round + 8*CONFIG_T::table_size/16;
            if (index < 0)   index = 0;
            else if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            out_data[j] = softplus_table[index];
        }
        res.write(out_data);
    }
}

#else

template<class data_T, class res_T, typename CONFIG_T>
void softplus(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
assert("softplus stream not implemented for AC Math");
}

#endif


// *************************************************
//       Softsign Activation
// *************************************************

#ifndef USE_AC_MATH

template<class data_T, class res_T, typename CONFIG_T>
void softsign(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softsign_table<CONFIG_T, CONFIG_T::table_size>(softsign_table);
        initialized = true;
    }

    #pragma hls_pipeline_init_interval 1
    SoftsignActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        SoftsignPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            int data_round = in_data[j]*CONFIG_T::table_size/16;
            int index = data_round + 8*CONFIG_T::table_size/16;
            if (index < 0)   index = 0;
            else if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            out_data[j] = softsign_table[index];
        }
        res.write(out_data);
    }
}

#else

template<class data_T, class res_T, typename CONFIG_T>
void softsign(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
assert("softsign stream not implemented for AC Math");
}

#endif


// *************************************************
//       ELU Activation
// *************************************************

#ifndef USE_AC_MATH

template<class data_T, class res_T, typename CONFIG_T>
void elu(ac_channel<data_T> &data, typename data_T::value_type alpha, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_elu_table<CONFIG_T, CONFIG_T::table_size>(elu_table);
        initialized = true;
    }

    #pragma hls_pipeline_init_interval 1
    EluActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        EluPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            
            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = datareg;
            } else {
                int index = datareg*CONFIG_T::table_size/-8;
                if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
                out_data[j] = alpha * elu_table[index];
            }
        }
        res.write(out_data);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void elu(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    elu<data_T, res_T, CONFIG_T>(data, 1.0, res);
}

#else

template<class data_T, class res_T, typename CONFIG_T>
void elu(ac_channel<data_T> &data, ac_channel<res_T> &res)
{
assert("elu stream not implemented for AC Math");
}

#endif

// *************************************************
//       SELU Activation
// *************************************************

#ifndef USE_AC_MATH

template<class data_T, class res_T, typename CONFIG_T>
void selu(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_selu_table<CONFIG_T, CONFIG_T::table_size>(selu_table);
        initialized = true;
    }

    #pragma hls_pipeline_init_interval 1
    SeluActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        SeluPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL

            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = (typename data_T::value_type) 1.0507009873554804934193349852946 * datareg;
            } else {
                int index = datareg*CONFIG_T::table_size/-8;
                if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
                out_data[j] = selu_table[index];
            }
        }
        res.write(out_data);
    }
}

#else

template<class data_T, class res_T, typename CONFIG_T>
void selu(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
assert("selu stream not implemented for AC Math");
}

#endif

// *************************************************
//       PReLU Activation
// *************************************************

template<class data_T, class res_T, typename CONFIG_T>
void prelu(ac_channel<data_T> &data, typename data_T::value_type alpha[CONFIG_T::n_in], ac_channel<res_T> &res) {
    #pragma hls_pipeline_init_interval 1
    PReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        PReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if (in_data[j] > 0) out_data[j] = in_data[j];
            else out_data[j] = alpha[i*res_T::size+j] * in_data[j];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void binary_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    #pragma hls_pipeline_init_interval 1
    PReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        PReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if(in_data[j] > 0) out_data[j] = (typename res_T::value_type) 1;
            else out_data[j] = (typename res_T::value_type) -1;
        }
        res.write(out_data);
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void ternary_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    #pragma hls_pipeline_init_interval 1
    PReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        PReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            //#pragma HLS UNROLL
            if(in_data[j] > 1) out_data[j] = (typename res_T::value_type) 1;
            else if (in_data[j] <=-1) out_data[j] = (typename res_T::value_type) -1;
            else out_data[j] = (typename res_T::value_type) 0;
        }
        res.write(out_data);
    }
}



}

#endif
