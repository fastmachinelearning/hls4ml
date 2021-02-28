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

//#include <cmath>
#include "nnet_common.h"

namespace nnet {

struct activ_config
{
    // IO size
    static const unsigned n_in = 10;

    // Internal info
    static const unsigned table_size = 512;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ac_fixed<16,8> table_t;
};

// *************************************************
//       LINEAR Activation -- See Issue 53
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  linear(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        res[ii] = datareg;
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = 0;
    }
}

template<class data_T, class res_T, int MAX_INT, typename CONFIG_T>
void  relu_max(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
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
template<class data_T, class res_T, typename CONFIG_T>
void  sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    #include "activation_tables/sigmoid_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        int data_round = (data[ii]*(CONFIG_T::table_size/16)).to_int();
        int index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) sigmoid_table[index];
    }
}

// *************************************************
//       Softmax Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  softmax(  data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
  #include "activation_tables/exp_table.tb"
  #include "activation_tables/invert_table.tb"

  hls_register int data_round[CONFIG_T::n_in];
  New_loop:
  #pragma unroll
  for (int ii=0; ii<CONFIG_T::n_in; ii++) {
      data_round[ii] = (data[ii] * (CONFIG_T::table_size/16)).to_int();
  }
  NN_Outer:
  #pragma unroll
  for (int ii=0; ii<CONFIG_T::n_in; ii++) {
      typename CONFIG_T::exp_table_t exp_res_temp = 0;
      NN_Inner:
      #pragma unroll
      for (int jj=0; jj<CONFIG_T::n_in; jj++)
      {
          if (ii==jj)
          {
              exp_res_temp += 1;
          }
          else
          {
              int _data_cache = (data_round[jj]-data_round[ii]);
              int index = _data_cache + 8*CONFIG_T::table_size/16;

              if (index < 0)   index = 0;
              if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;

              typename CONFIG_T::exp_table_t temp_exp = exp_table[index];
              exp_res_temp += temp_exp;
          }
      }
      int exp_res_index = (exp_res_temp * CONFIG_T::table_size/64).to_int();
      if (exp_res_index < 0)   exp_res_index = 0;
      if (exp_res_index > CONFIG_T::table_size-1) exp_res_index = CONFIG_T::table_size-1;
      res[ii] = invert_table[exp_res_index];
  }
}

// *************************************************
//       TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  dense_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
    #include "activation_tables/tanh_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        ac_int<16> data_round = (data[ii]*(CONFIG_T::table_size/8)).to_int();
        ac_int<16> index = data_round +  4*CONFIG_T::table_size/8;
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
    data_T slope = (data_T) 0.2;
    data_T shift = (data_T) 0.5;
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = slope * data[ii] + shift;
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
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
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
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        if (datareg > theta) res[ii] = datareg;
        else res[ii] = 0;
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void softplus(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
    #include "activation_tables/softplus_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        ac_int<16> data_round = (data[ii]*(CONFIG_T::table_size/16)).to_int();
        ac_int<16> index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) softplus_table[index];
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  softsign(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
    #include "activation_tables/softsign_table.tb"

    // Index into the lookup table based on data
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        ac_int<16> data_round = (data[ii]*(CONFIG_T::table_size/16)).to_int();
        ac_int<16> index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) softsign_table[index];
    }
}

// *************************************************
//       ELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  elu(data_T data[CONFIG_T::n_in], const res_T alpha, res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
    #include "activation_tables/elu_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = datareg;
        } else {
            ac_int<16> index = (datareg*(CONFIG_T::table_size/-8)).to_int();
            if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            res[ii] = alpha * elu_table[index];
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void elu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
	elu<data_T, res_T, CONFIG_T>(data, 1.0, res);
}

// *************************************************
//       SELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  selu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
    #include "activation_tables/selu_table.tb"
    // Index into the lookup table based on data
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = res_T(1.0507009873554804934193349852946) * datareg;
        } else {
            ac_int<16> index = (datareg*(CONFIG_T::table_size/-8)).to_int();
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
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
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
    #pragma unroll
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_T datareg = data[ii];
        res_T cache;
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
  #pragma unroll
  for (int ii=0; ii<CONFIG_T::n_in; ii++) {
    data_T datareg = 2*data[ii];
    res_T cache;
    if( datareg > 1 ) cache = 1;
    else if( datareg > -1 && datareg <= 1) cache=0;
    else cache = -1;

    res[ii] = (res_T) cache;
  }
}

}

#endif
