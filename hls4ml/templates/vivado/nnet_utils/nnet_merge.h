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

#ifndef NNET_MERGE_H_
#define NNET_MERGE_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct merge_config
{
    static const unsigned n_elem = 10;
};


struct dot_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    typedef float accum_t;
};

struct concat_config {
    static const unsigned n_elem1_0 = 10;
    static const unsigned n_elem1_1 = 10;
    static const unsigned n_elem1_2 = 10;
    static const unsigned n_elem2_0 = 10;
    static const unsigned n_elem2_1 = 10;
    static const unsigned n_elem2_2 = 10;

    static const unsigned axis = -1;
};

template<class input1_T, class input2_T, class ret_T>
inline typename std::enable_if<std::is_same<input1_T, ap_uint<1>>::value
        and std::is_same<input2_T, ap_uint<1>>::value, ap_uint<1>>::type
product(ap_uint<1> a, ap_uint<1> w){
    // specialisation for 1-bit weights and incoming data
    #pragma HLS inline off
    return a == w;
}

template<class input1_T, class input2_T, class ret_T>
inline typename std::enable_if<(not std::is_same<input1_T, ap_uint<1>>::value)
        and std::is_same<input2_T, ap_uint<1>>::value, ret_T>::type
product(input1_T a, ap_uint<1> w){
    // Specialisation for 1-bit weights, arbitrary data
    #pragma HLS inline off
    return w == 0 ? (input1_T) -a : a;
}

template<class input1_T, class input2_T, class ret_T>
inline typename std::enable_if<(not std::is_same<input1_T, ap_uint<2>>::value)
        and std::is_same<input2_T, ap_int<2>>::value, ret_T>::type
product(input1_T a, ap_int<2> w){
    // Specialisation for 2-bit weights, arbitrary data
    #pragma HLS inline off
    if (w == 0) return (input1_T) 0;
    else if(w == -1) return (input1_T) -a;
    else return (input1_T) a; // if(w == 1)
}

template<class input1_T, class input2_T, class ret_T>
inline typename std::enable_if<(not std::is_same<input1_T, ap_uint<1>>::value)
        and (not std::is_same<input2_T, ap_uint<1>>::value), ret_T>::type
product(input1_T a, input2_T w){
    // 'Normal' product
    #pragma HLS inline off
    return a * w;
}

template<typename input1_T, typename input2_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<input1_T, ap_uint<1>>::value
        and std::is_same<input2_T, ap_uint<1>>::value, ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>>::type
cast(typename CONFIG_T::accum_t x){
  return (ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>) (x - CONFIG_T::n_in / 2) * 2;
}

template<typename input1_T, typename input2_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<(not std::is_same<input1_T, ap_uint<1>>::value) 
        or (not std::is_same<input2_T, ap_uint<1>>::value), res_T>::type
cast(typename CONFIG_T::accum_t x){
  return (res_T) x;
}


template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] + data2[ii];
    }
}


template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void subtract(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] - data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void multiply(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] * data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void average(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] * data2[ii] / (res_T) 2;
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void maximum(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = (data1[ii] > data2[ii]) ? data1[ii] : data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void minimum(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = (data1[ii] < data2[ii]) ? data1[ii] : data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void dot1d(
    input1_T data1[CONFIG_T::n_in], 
	input2_T data2[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out])
{
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    Product: for(int ii=0; ii < CONFIG_T::n_in; ii++){
                #pragma HLS PIPELINE
                int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=product limit=multiplier_limit function
                mult[ii] = product<input1_T, input2_T, res_T>(data1[ii], data2[ii]);
             }

    Accum: for(int ii = 0; ii < CONFIG_T::n_in; ii++){
              #pragma HLS PIPELINE
              acc[0] += mult[ii];
            }
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
              #pragma HLS_UNROLL
              res[ires] = cast<input1_T, input2_T, res_T, CONFIG_T>(acc[ires]);
            }
}


template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate1d(
    input1_T data1[CONFIG_T::n_elem1_0], 
	input2_T data2[CONFIG_T::n_elem2_0],
    res_T res[CONFIG_T::n_elem1_0 + CONFIG_T::n_elem2_0])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
        res[ii] = data1[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem2_0; ii++) {
        res[CONFIG_T::n_elem1_0 + ii] = data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_0(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1; ii++) {
        res[ii] = data1[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1; ii++) {
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + ii] = data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_1(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
        for (int jj=0; jj<CONFIG_T::n_elem1_1; jj++) {
            res[ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + jj] = data1[ii * CONFIG_T::n_elem1_1 + jj];
        }
        for (int jj=0; jj<CONFIG_T::n_elem2_1; jj++) {
            res[ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + CONFIG_T::n_elem1_1 + jj] = data2[ii * CONFIG_T::n_elem2_1 + jj];
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1])
{
    if (CONFIG_T::axis == 1 || CONFIG_T::axis == -1) {
        concatenate2d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate2d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_0(
input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2],
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2; ii++) {
        res[ii] = data1[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2; ii++) {
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + ii] = data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_1(
input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
        for (int jj=0; jj<CONFIG_T::n_elem1_1; jj++) {
            for (int kk=0; kk<CONFIG_T::n_elem1_2; kk++) {
                int res_idx = ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) * CONFIG_T::n_elem1_2
                            + jj * CONFIG_T::n_elem1_2
                            + kk;
                int data_idx = ii * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2
                             + jj * CONFIG_T::n_elem1_2
                             + kk;
                res[res_idx] = data1[data_idx];
            }
        }
        for (int jj=0; jj<CONFIG_T::n_elem2_1; jj++) {
            for (int kk=0; kk<CONFIG_T::n_elem2_2; kk++) {
                int res_idx = ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) * CONFIG_T::n_elem1_2
                            + (jj + CONFIG_T::n_elem1_1) * CONFIG_T::n_elem1_2
                            + kk;
                int data_idx = ii * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2
                             + jj * CONFIG_T::n_elem2_2
                             + kk;
                res[res_idx] = data2[data_idx];
            }
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_2(
input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
        for (int jj=0; jj<CONFIG_T::n_elem1_1; jj++) {
            for (int kk=0; kk<CONFIG_T::n_elem1_2; kk++) {
                int res_idx = ii * CONFIG_T::n_elem1_1 * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2)
                            + jj * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2)
                            + kk;
                int data_idx = ii * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2
                             + jj * CONFIG_T::n_elem1_2
                             + kk;
                res[res_idx] = data1[data_idx];
            }
            for (int kk=0; kk<CONFIG_T::n_elem1_2; kk++) {
                res[ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + CONFIG_T::n_elem1_1 + jj] = data1[ii * CONFIG_T::n_elem2_1 + jj];
                int res_idx = ii * CONFIG_T::n_elem1_1 * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2)
                            + jj * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2)
                            + kk + CONFIG_T::n_elem1_2;
                int data_idx = ii * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2
                             + jj * CONFIG_T::n_elem2_2
                             + kk;
                res[res_idx] = data2[data_idx];
            }
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2],
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2])
{
    if (CONFIG_T::axis == 2 || CONFIG_T::axis == -1) {
        concatenate3d_2<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else if (CONFIG_T::axis == 1) {
        concatenate3d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate3d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

}

#endif
