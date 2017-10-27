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
//#include "hls_stream.h"

namespace nnet {


// *************************************************
//       RELU Activation
// *************************************************
template<class data_T, class res_T, int N_IN>
void  relu(data_T data[N_IN], res_T res[N_IN])
{
    //believe these two lines are not needed
    //#pragma HLS ARRAY_PARTITION variable=data complete
    //#pragma HLS ARRAY_PARTITION variable=res complete

    //using unroll below instead
    //#pragma HLS pipeline II=1 

    data_T datareg;
    for (int ii=0; ii<N_IN; ii++) {
        #pragma HLS UNROLL 
        datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = 0;
    }
}

template<class data_T, class res_T, int N_IN, int MAX_INT>
void  relu_max(data_T data[N_IN], res_T res[N_IN])
{
    data_T datareg;
    for (int ii=0; ii<N_IN; ii++) {
        #pragma HLS UNROLL 
        datareg = data[ii];
        if (datareg < 0) res[ii] = 0;
        else if (datareg > MAX_INT) res[ii] = MAX_INT;
        else res[ii] = datareg;
    }
}

template<class data_T, class res_T, int N_IN>
void  relu6(data_T data[N_IN], res_T res[N_IN])
{
    relu_max<data_T, res_T, N_IN, 6>(data, res);
}


// *************************************************
//       Sigmoid Activation
// *************************************************
float sigmoid_fcn_float(float input) {
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

template<class data_T, class res_T, int N_IN, int TABLE_SIZE/*=1024*/>
void  sigmoid(data_T data[N_IN], res_T res[N_IN])
{
    // Initialize the lookup table
    res_T sigmoid_table[TABLE_SIZE];
    init_sigmoid_table<res_T, TABLE_SIZE>(sigmoid_table);

    // Index into the lookup table based on data
    data_T datareg;
    int data_round;
    int index;
    for (int ii=0; ii<N_IN; ii++) {
        #pragma HLS UNROLL 
        data_round = data[ii]*TABLE_SIZE/16;
        index = data_round + 8*TABLE_SIZE/16;
        if (index < 0)   index = 0;
        if (index > TABLE_SIZE-1) index = TABLE_SIZE-1;
        res[ii] = sigmoid_table[index];
    }
}

// Default table size provided here:
template<class data_T, class res_T, int N_IN>
void  sigmoid(data_T data[N_IN], res_T res[N_IN]){ sigmoid<data_T, res_T, N_IN, 1024>(data, res); }


// *************************************************
//       Softmax Activation
// *************************************************
float exp_fcn_float(float input) {
    return exp(input);
}

template<class data_T, int N_TABLE>
void init_exp_table(data_T table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        data_T real_val = exp_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, int N_IN, int TABLE_SIZE/*=1024*/>
void  softmax(data_T data[N_IN], res_T res[N_IN])
{
    // Initialize the lookup table
    res_T exp_table[TABLE_SIZE];
    init_exp_table<res_T, TABLE_SIZE>(exp_table);

    // Index into the lookup table based on data for exponentials
    res_T exp_res[N_IN];//same precision as rest?
    res_T exp_res_sum=0;
    data_T datareg;
    int data_round;
    int index;
    for (int ii=0; ii<N_IN; ii++) {
        #pragma HLS UNROLL 
        data_round = data[ii]*TABLE_SIZE/16;
        index = data_round + 8*TABLE_SIZE/16;
        if (index < 0)   index = 0;
        if (index > TABLE_SIZE-1) index = TABLE_SIZE-1;
        exp_res[ii] = exp_table[index];
        exp_res_sum += exp_table[index];
    }

    //Second loop to divide by sum of exponentials
    for (int ii=0; ii<N_IN; ii++) {
      #pragma HLS UNROLL
      res[ii] = exp_res[ii]/exp_res_sum; //Note division used here!
    }

}

// Default table size provided here:
template<class data_T, class res_T, int N_IN>
void  softmax(data_T data[N_IN], res_T res[N_IN]){ softmax<data_T, res_T, N_IN, 1024>(data, res); }


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
        std::cout << "Tanh:  Lookup table Index: " <<  ii<< " In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}


template<class data_T, class res_T, int N_IN, int TABLE_SIZE/*=1024*/>
void  tanh(data_T data[N_IN], res_T res[N_IN])
{
    // Initialize the lookup table
    res_T tanh_table[TABLE_SIZE];
    init_tanh_table<res_T, TABLE_SIZE>(tanh_table);

    // Index into the lookup table based on data
    data_T datareg;
    int data_round;
    int index;
    for (int ii=0; ii<N_IN; ii++) {
        #pragma HLS UNROLL 
        data_round = data[ii]*TABLE_SIZE/8;
        index = data_round + 4*TABLE_SIZE/8;
        //std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
        if (index < 0)   index = 0;
        if (index > TABLE_SIZE-1) index = TABLE_SIZE-1;
        res[ii] = tanh_table[index];
    }
}

// Default table size provided here:
template<class data_T, class res_T, int N_IN>
void  tanh(data_T data[N_IN], res_T res[N_IN]){ tanh<data_T, res_T, N_IN, 1024>(data, res); }

}

#endif
