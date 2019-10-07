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

#include <cmath>
#include "ap_fixed.h"
#include "nnet_common.h"

#ifdef MNTR_CATAPULT_HLS
#include <math/mgc_ac_math.h>
// TODO: use the provided functions
//#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_softmax_pwl.h>
#endif

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
//       LINEAR Activation -- See Issue 53
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  linear(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        res[ii] = data[ii];
    }
}



// *************************************************
//       RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = 0;
    }
}

template<class data_T, class res_T, int MAX_INT, typename CONFIG_T>
void  relu_max(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
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

template<class data_T, class res_T, typename CONFIG_T>
void  relu1(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    relu_max<data_T, res_T, 1, CONFIG_T>(data, res);
}

// *************************************************
//       Sigmoid Activation
// *************************************************
#ifdef MNTR_CATAPULT_HLS
inline ap_fixed<32,16> sigmoid_fcn_fixed(ap_fixed<10,4> input) {
    ap_fixed<32,16> tmp;
    mgc_ac_exp(-input, tmp);
    return (ap_fixed<32,16>(1.0) / (ap_fixed<32,16>(1) + tmp));
}
#else
inline float sigmoid_fcn_float(float input) {
    return 1.0 / (1 + std::exp(-input));
}
#endif

template<typename CONFIG_T, int N_TABLE>
void init_sigmoid_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default logistic sigmoid function:
    //   result = 1/(1+e^(-x))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
#ifdef MNTR_CATAPULT_HLS
        typename CONFIG_T::table_t real_val = sigmoid_fcn_fixed(in_val);
#else
        typename CONFIG_T::table_t real_val = sigmoid_fcn_float(in_val);
#endif
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
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
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    // Index into the lookup table based on data
#ifndef MNTR_CATAPULT_HLS
    int data_round;
    int index;
#else
    ac_int<32> data_round;
    ac_int<32> index;
#endif
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        data_round = (data[ii]*CONFIG_T::table_size/16).to_int();
        index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) sigmoid_table[index];
    }
}

// *************************************************
//       Softmax Activation
// *************************************************
#ifndef MNTR_CATAPULT_HLS
#ifdef MNTR_CATAPULT_HLS
inline ap_fixed<52,42> exp_fcn_fixed(ap_fixed<10,4> input) {
    ap_fixed<52,42> result;
    mgc_ac_exp(input, result);
    return result;
}
#else
inline float exp_fcn_float(float input) {
    return std::exp(input);
}
#endif

template<typename CONFIG_T, int N_TABLE>
void init_exp_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++) {
#ifdef MNTR_CATAPULT_HLS
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        ap_fixed<18,8> range(2 * 8.0);
        ap_fixed<32,30> N_TABLE_HALF(N_TABLE / 2.0);
        ap_fixed<32,16> in_val;
        div(ap_fixed<32,30>(range * ( ii - N_TABLE_HALF )), ap_fixed<32,32>(N_TABLE), in_val);
        // Next, compute lookup table function
        typename CONFIG_T::table_t fixed_val = exp_fcn_fixed(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = fixed_val;
#else
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = exp_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
#endif
    }
}

template<typename CONFIG_T, int N_TABLE>
void init_invert_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < N_TABLE; ii++) {
#ifdef MNTR_CATAPULT_HLS
	ap_fixed<18,8> in_val;
    ap_fixed<32,32> dividend(64.0*ii);
    ap_fixed<32,32> divisor(N_TABLE);
    div(dividend, divisor, in_val);
        // Next, compute lookup table function
	if (in_val > 0.0) {
        ap_fixed<18,8> one(1.0);
        div(one, in_val, table_out[ii]);
    }
	else table_out[ii] = 0.0;
#else
	float in_val = 64.0*ii/float(N_TABLE);
        // Next, compute lookup table function
	if (in_val > 0.0) table_out[ii] = 1.0/in_val;
	else table_out[ii] = 0.0;
#endif
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  softmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
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
        init_exp_table<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_invert_table<CONFIG_T, CONFIG_T::table_size>(invert_table);
        initialized = true;
    }
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        // Note: This is going to be a resource hog to run with pipeline, but hey, whatever
        #pragma HLS PIPELINE
    }
#endif
    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[CONFIG_T::n_in];// different, independent, fixed point precision
    typename CONFIG_T::table_t exp_diff_res;// different, independent, fixed point precision
    data_T data_cache[CONFIG_T::n_in];
#ifdef MNTR_CATAPULT_HLS
    ap_int<32> data_round;
    ap_int<32> index;
#else
    int data_round;
    int index;
#endif
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
      data_cache[ii] = data[ii];
      exp_res[ii] = 0;
    }
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
      if (CONFIG_T::io_type == io_serial){
          #pragma HLS PIPELINE
      }
#endif
      for (int jj=0; jj<CONFIG_T::n_in; jj++) {
	if (ii==jj) exp_diff_res = 1;
	else {
#ifdef MNTR_CATAPULT_HLS
      data_T _data_cache = (data_cache[jj]-data_cache[ii]);
      ap_int<32> _table_size_16 = ap_int<32>(CONFIG_T::table_size)/ap_int<32>(16);
	  data_round = (_data_cache * _table_size_16).to_int();
	  index = data_round + ap_int<32>(8)*(ap_int<32>(CONFIG_T::table_size)/ap_int<32>(16));
#else
	  data_round = (data_cache[jj]-data_cache[ii])*CONFIG_T::table_size/16;
	  index = data_round + 8*CONFIG_T::table_size/16;
#endif
	  if (index < 0)   index = 0;
	  if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
#ifdef MNTR_CATAPULT_HLS
	  exp_diff_res = exp_table[index.to_uint()];
#else
	  exp_diff_res = exp_table[index];
#endif
	}
	exp_res[ii] += exp_diff_res;
      }
    }

    //Second loop to invert
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifdef MNTR_CATAPULT_HLS
      int exp_res_index = (exp_res[ii]*ac_fixed<18,8>(CONFIG_T::table_size/64)).to_int();
#else
      int exp_res_index = exp_res[ii]*CONFIG_T::table_size/64;
#endif
      if (exp_res_index < 0)   exp_res_index = 0;
      if (exp_res_index > CONFIG_T::table_size-1) exp_res_index = CONFIG_T::table_size-1;
      //typename CONFIG_T::table_t exp_res_invert = invert_table[exp_res_index];
      res[ii] = (res_T) invert_table[exp_res_index];
    }

}
#else
template<class data_T, class res_T, typename CONFIG_T>
void  softmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{

    typedef ac_fixed<18, 6, true, AC_TRN, AC_SAT> input_type_t;
    typedef ac_fixed<18, 2, false, AC_TRN, AC_SAT> output_type_t;

    input_type_t _data[CONFIG_T::n_in];
    output_type_t _res[CONFIG_T::n_in];

    for (int i = 0; i < CONFIG_T::n_in; i++) {
        _data[i] = data[i];
    }

    ac_math::ac_softmax_pwl(_data, _res);


    for (int i = 0; i < CONFIG_T::n_in; i++) {
        res[i] = _res[i];
    }
}
#endif

// *************************************************
//       TanH Activation
// *************************************************
template<typename CONFIG_T, int N_TABLE>
void init_tanh_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Implement tanh lookup
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
        float in_val = 2*4.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = tanh(in_val);
        //std::cout << "Tanh:  Lookup table Index: " <<  ii<< " In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void  tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
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
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        data_round = data[ii]*CONFIG_T::table_size/8;
        index = data_round + 4*CONFIG_T::table_size/8;
        //std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) tanh_table[index];
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  hard_sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    data_T datareg;
    data_T slope = (data_T) 0.2;
    data_T shift = (data_T) 0.5;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        datareg = slope * data[ii] + shift;
        if (datareg > 1) datareg = 1;
        else if (datareg < 0) datareg = 0;
        res[ii] = datareg;
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  leaky_relu(data_T data[CONFIG_T::n_in], data_T alpha, res_T res[CONFIG_T::n_in])
{
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = alpha * datareg;
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  thresholded_relu(data_T data[CONFIG_T::n_in], data_T theta, res_T res[CONFIG_T::n_in])
{
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        datareg = data[ii];
        if (datareg > theta) res[ii] = datareg;
        else res[ii] = 0;
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
#ifdef MNTR_CATAPULT_HLS
inline ap_fixed<18,8> softplus_fcn_float(ap_fixed<18,8> input) {
    ap_fixed<18,8> _exp; //mgc_ac_exp(input, _exp);
    ap_fixed<18,8> _log; //mgc_ac_log(_exp, _log);
    return _log + ap_fixed<18,8>(1.);
}
#else
inline float softplus_fcn_float(float input) {
    return std::log(std::exp(input) + 1.);
}
#endif

template<typename CONFIG_T, int N_TABLE>
void init_softplus_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default softplus function:
    //   result = log(exp(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
#ifdef MNTR_CATAPULT_HLS
        ap_fixed<18,8>in_val; // = 2*8.0*(ii-ap_fixed<18,8>(N_TABLE)/ap_fixed<18,8>(2.0))/ap_fixed<18,8>(N_TABLE);
#else
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
#endif
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softplus_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  softplus(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
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
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        data_round = data[ii]*CONFIG_T::table_size/16;
        index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) softplus_table[index];
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
inline float softsign_fcn_float(float input) {
    return input / (std::abs(input) + 1.);
}

template<typename CONFIG_T, int N_TABLE>
void init_softsign_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default softsign function:
    //   result = x / (abs(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softsign_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  softsign(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
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
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        data_round = data[ii]*CONFIG_T::table_size/16;
        index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) softsign_table[index];
    }
}

// *************************************************
//       ELU Activation
// *************************************************
inline float elu_fcn_float(float input) {
    return std::exp(input) - 1.;
}

template<typename CONFIG_T, int N_TABLE>
void init_elu_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default ELU function:
    //   result = alpha * (e^(x) - 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0*ii/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = elu_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  elu(data_T data[CONFIG_T::n_in], const res_T alpha, res_T res[CONFIG_T::n_in])
{
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
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    data_T datareg;
    // Index into the lookup table based on data
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = datareg;
        } else {
            index = datareg*CONFIG_T::table_size/-8;
            if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            res[ii] = alpha * elu_table[index];
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  elu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
	elu<data_T, res_T, CONFIG_T>(data, 1.0, res);
}

// *************************************************
//       SELU Activation
// *************************************************
inline float selu_fcn_float(float input) {
    return 1.0507009873554804934193349852946 * (1.6732632423543772848170429916717 * (std::exp(input) - 1.));
}

template<typename CONFIG_T, int N_TABLE>
void init_selu_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default SELU function:
    //   result = 1.05 * (1.673 * (e^(x) - 1))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0*ii/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = selu_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  selu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
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
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    data_T datareg;
    // Index into the lookup table based on data
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = res_T(1.0507009873554804934193349852946) * datareg;
        } else {
            index = datareg*CONFIG_T::table_size/-8;
            if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            res[ii] = selu_table[index];
        }
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  prelu(data_T data[CONFIG_T::n_in], data_T alpha[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
#ifndef MNTR_CATAPULT_HLS
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }
#endif
    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
#endif
        datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = alpha[ii] * datareg;
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  binary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
#ifndef MNTR_CATAPULT_HLS
 if (CONFIG_T::io_type == io_parallel){
     #pragma HLS PIPELINE
 }
#endif
 data_T datareg;
 res_T cache;
 for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
  if (CONFIG_T::io_type == io_serial){
      #pragma HLS PIPELINE
  }
#endif
  datareg = data[ii];
  if( datareg > 0 ) cache = 1;
  else cache = -1;

  res[ii] = (res_T) cache;

 }

}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  ternary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
#ifndef MNTR_CATAPULT_HLS
 if (CONFIG_T::io_type == io_parallel){
     #pragma HLS PIPELINE
 }
#endif
 data_T datareg;
 res_T cache;
 for (int ii=0; ii<CONFIG_T::n_in; ii++) {
#ifndef MNTR_CATAPULT_HLS
  if (CONFIG_T::io_type == io_serial){
      #pragma HLS PIPELINE
  }
#endif
  datareg = 2*data[ii];
  if( datareg > 1 ) cache = 1;
  else if( datareg > -1 && datareg <= 1) cache=0;
  else cache = -1;

  res[ii] = (res_T) cache;

 }

}

}

#endif
