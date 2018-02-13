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

#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

#include <math.h>
#include "ap_fixed.h"
#include "nnet_common.h"



namespace nnet {

struct activ_config
{
    // IO size
    static const unsigned n_in = 10;

    // Internal info
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    
    // Internal data type definitions    
    typedef ap_fixed<18,8> table_t;
};


// *************************************************
//       RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        // #pragma HLS UNROLL
        datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = 0;
    }
}

template<class data_T, class res_T, int MAX_INT, typename CONFIG_T>
void  relu_max(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        // #pragma HLS UNROLL
        datareg = data[ii];
        if (datareg < 0) res[ii] = 0;
        else if (datareg > MAX_INT) res[ii] = MAX_INT;
        else res[ii] = datareg;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  relu6(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    relu_max<data_T, res_T, 6, CONFIG_T>(data, res);
}


// *************************************************
//       Sigmoid Activation
// *************************************************
inline float sigmoid_fcn_float(float input) {
    return 1.0 / (1 + exp(-input));
}

template<class data_T, int N_TABLE>
void init_sigmoid_table(data_T table_out[N_TABLE])
{
    // Default logistic sigmoid function:
    //   result = 1/(1+e^(-x))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        data_T real_val = sigmoid_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
    res_T sigmoid_table[CONFIG_T::table_size];
    init_sigmoid_table<res_T, CONFIG_T::table_size>(sigmoid_table);

    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        // #pragma HLS UNROLL
        data_round = data[ii]*CONFIG_T::table_size/16;
        index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = sigmoid_table[index];
    }
}

// *************************************************
//       Softmax Activation
// *************************************************
inline float exp_fcn_float(float input) {
    return exp(input);
}


template<typename CONFIG_T, int N_TABLE>
void init_exp_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = exp_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<typename CONFIG_T, int N_TABLE>
void init_invert_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < N_TABLE; ii++) {
      // First, convert from table index to X-value (signed 8-bit, range 0 to +64)
	float in_val = 64.0*ii/float(N_TABLE);
        // Next, compute lookup table function
	if (in_val > 0.0) table_out[ii] = 1.0/in_val;
	else table_out[ii] = 0.0;
    }    
}

template<class data_T, class res_T, typename CONFIG_T>
void  softmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
    typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    init_exp_table<CONFIG_T, CONFIG_T::table_size>(exp_table);

    typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
    init_invert_table<CONFIG_T, CONFIG_T::table_size>(invert_table);

    if (CONFIG_T::io_type == io_parallel){
        // Note: This is going to be a resource hog to run with pipeline, but hey, whatever
        #pragma HLS PIPELINE
    }

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[CONFIG_T::n_in];// different, independent, fixed point precision
    typename CONFIG_T::table_t exp_res_sum=0;
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        // #pragma HLS UNROLL
        data_round = data[ii]*CONFIG_T::table_size/16;
        index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        exp_res[ii] = exp_table[index];
        exp_res_sum += exp_table[index];
    }


    int exp_res_sum_index = exp_res_sum*CONFIG_T::table_size/64;
    if (exp_res_sum_index < 0)   exp_res_sum_index = 0;
    if (exp_res_sum_index > CONFIG_T::table_size-1) exp_res_sum_index = CONFIG_T::table_size-1;
    typename CONFIG_T::table_t exp_res_sum_invert = invert_table[exp_res_sum_index];

    //std::cout << "exp_res_sum = " << exp_res_sum << std::endl;
    //std::cout << "1/exp_res_sum = " << 1.0/float(exp_res_sum) << std::endl;
    //std::cout << "exp_res_sum_invert = " << (float) exp_res_sum_invert << std::endl;    

    //Second loop to divide by sum of exponentials
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
      // #pragma HLS UNROLL
      //res[ii] = exp_res[ii]/exp_res_sum; //Note division used here!
      res[ii] = (res_T) exp_res[ii]*exp_res_sum_invert;
    }

}

// *************************************************
//       TanH Activation
// *************************************************
template<class data_T, int N_TABLE>
void init_tanh_table(data_T table_out[N_TABLE])
{
    // Implement tanh lookup
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
        float in_val = 2*4.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        data_T real_val = tanh(in_val);
        //std::cout << "Tanh:  Lookup table Index: " <<  ii<< " In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void  tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
    res_T tanh_table[CONFIG_T::table_size];
    init_tanh_table<res_T, CONFIG_T::table_size>(tanh_table);

    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        // #pragma HLS UNROLL
        data_round = data[ii]*CONFIG_T::table_size/8;
        index = data_round + 4*CONFIG_T::table_size/8;
        //std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = tanh_table[index];
    }
}

}

#endif
