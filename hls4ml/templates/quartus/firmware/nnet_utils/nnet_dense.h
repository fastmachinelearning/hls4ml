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

#ifndef NNET_DENSE_LARGE_H_
#define NNET_DENSE_LARGE_H_

#include "nnet_common.h"

namespace nnet {


struct dense_config
{
   // Internal data type definitions
   typedef float bias_t;
   typedef float weight_t;
   typedef float accum_t;

   // Layer Sizes
   static const unsigned n_in = 10;
   static const unsigned n_out = 10;

   static const unsigned reuse_factor = 1;
   static const unsigned block_factor = 1; //DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
   static const unsigned multiplier_limit = 1; //DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor)
   static const unsigned multiplier_factor = 1; //min n_in, rf
   static const unsigned multiplier_scale = 1; // M_LIMIT/CONFIG_T::n_out;
   static const unsigned reciprocal = 1; // 2^35 / 25
   static const unsigned rf_pad = 0;
   static const unsigned bf_pad = 0;
   // Resource reuse info
   static const unsigned io_type = io_parallel;
   static const bool store_weights_in_bram = false;
   static const unsigned n_zeros = 0;
   // partitioning arrays cyclically to go with roll factors?
};

template<class data_T, class weight_T, class ret_T>
inline typename std::enable_if<std::is_same<data_T, ac_int<1, false>>::value
        and std::is_same<weight_T, ac_int<1, false>>::value, ac_int<1, false>>::type
product(ac_int<1, false> a, ac_int<1, false> w){
    // specialisation for 1-bit weights and incoming data
    return (ret_T) (a == w);
}

template<class data_T, class weight_T, class ret_T>
inline typename std::enable_if<(not std::is_same<data_T, ac_int<1, false>>::value)
        and std::is_same<weight_T, ac_int<1, false>>::value, ret_T>::type
product(data_T a, ac_int<1, false> w){
    // Specialisation for 1-bit weights, arbitrary data
    return w == 0 ? (ret_T) -a : a;
}

template<class data_T, class weight_T, class ret_T>
inline typename std::enable_if<(not std::is_same<data_T, ac_int<2, false>>::value)
        and std::is_same<weight_T, ac_int<2, true>>::value, ret_T>::type
product(data_T a, ac_int<2, true> w){
    // Specialisation for 2-bit weights, arbitrary data
    if (w == 0) return (ret_T) 0;
    else if(w == -1) return (ret_T) -a;
    else return (ret_T) a; // if(w == 1)
}

template<class data_T, class weight_T, class ret_T>
inline typename std::enable_if<(not std::is_same<data_T, ac_int<1, false>>::value)
        and (not std::is_same<weight_T, ac_int<1, false>>::value), ret_T>::type
product(data_T a, weight_T w){
    // 'Normal' product
    return (ret_T)(a * w);
}

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<data_T, ac_int<1, false>>::value
        and std::is_same<typename CONFIG_T::weight_t, ac_int<1, false>>::value, ac_int<nnet::ceillog2(CONFIG_T::n_in) + 2, true>>::type
cast(typename CONFIG_T::accum_t x){
  return (ac_int<nnet::ceillog2(CONFIG_T::n_in) + 2, true>) (x - CONFIG_T::n_in / 2) * 2;
}

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<(not std::is_same<data_T, ac_int<1, false>>::value), res_T>::type
cast(typename CONFIG_T::accum_t x){
  return (res_T) x;
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_rf_gt(
   data_T    data[CONFIG_T::n_in],
   res_T     res[CONFIG_T::n_out],
   const typename CONFIG_T::weight_t  weights[CONFIG_T::reuse_factor_rounded*CONFIG_T::block_factor_rounded],
   const typename CONFIG_T::bias_t    biases[CONFIG_T::n_out]
   )
{
   assert((CONFIG_T::multiplier_limit % CONFIG_T::n_out == 0 || CONFIG_T::reuse_factor >= CONFIG_T::n_in) && "The current Reuse Factor is not allowed");
   assert((CONFIG_T::reuse_factor > CONFIG_T::n_in) && "This function is correct only for RF > N_IN");
   //#pragma ii CONFIG_T::reuse_factor
   hls_register typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
   Load:
   #pragma unroll
   for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
       acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
   }
   hls_register int out_index[CONFIG_T::reuse_factor][CONFIG_T::block_factor];
   hls_register int d_index[CONFIG_T::reuse_factor][CONFIG_T::block_factor];

   #pragma unroll
   for(int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
       #pragma unroll
       for(int im = 0; im < CONFIG_T::block_factor ; im++) {
         uint32 w_index = ir + CONFIG_T::reuse_factor * im;
         out_index[ir][im] = (w_index / CONFIG_T::multiplier_factor).to_int();
         d_index[ir][im] = w_index % CONFIG_T::n_in;
       }
   }
   Product1:
   #pragma nofusion
   #pragma speculated_iterations 0
   for(int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
      hls_register typename CONFIG_T::accum_t tmp_acc[CONFIG_T::block_factor];
      Product2:
      #pragma unroll
      for(int im = 0; im < CONFIG_T::block_factor ; im++) {
          uint32 w_index = ir + (CONFIG_T::reuse_factor_rounded) * im;
          if (w_index >= CONFIG_T::reuse_factor_rounded*CONFIG_T::block_factor_rounded) continue;
          int data_index = d_index[ir][im];
          tmp_acc[im] = product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[data_index], weights[w_index]);
      }
      hls_register typename CONFIG_T::accum_t mult[CONFIG_T::multiplier_limit];
      ResetMult:
      #pragma unroll
      for (int imult = 0; imult < CONFIG_T::multiplier_limit; imult++) {
          mult[imult] = 0;
      }
      AccumLoop1:
      #pragma unroll
      for(int im = 0; im < CONFIG_T::block_factor ; im++) {
          int o_index = out_index[ir][im];
          if (o_index >= CONFIG_T::n_out) continue; // check out of bounds
          mult[o_index] += tmp_acc[im];
      }
      AccumLoop2:
      #pragma unroll
      for (int im = 0; im < CONFIG_T::multiplier_limit; im++) {
          acc[im] += mult[im];
      }
   }
   Store:
   #pragma unroll
   for(int ires = 0; ires < CONFIG_T::n_out; ires++) {
     res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);//acc[jj];
   }
}
template<class data_T, class res_T, typename CONFIG_T>
void dense_rf_lt(
  data_T    data[CONFIG_T::n_in],
  res_T     res[CONFIG_T::n_out],
  const typename CONFIG_T::weight_t  weights[CONFIG_T::reuse_factor_rounded*CONFIG_T::block_factor_rounded],
  const typename CONFIG_T::bias_t    biases[CONFIG_T::n_out]
  )
{
    assert((CONFIG_T::multiplier_limit % CONFIG_T::n_out == 0 || CONFIG_T::reuse_factor >= CONFIG_T::n_in) && "The current Reuse Factor is not allowed");
    assert((CONFIG_T::multiplier_limit == CONFIG_T::block_factor) && "This function is correct only for RF <= N_IN");

    hls_register typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    InitAccum:
    #pragma unroll
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }
    ReuseLoop:
    #pragma nofusion
    #pragma speculated_iterations 0
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
       hls_register typename CONFIG_T::accum_t mult[CONFIG_T::block_factor];
       MultLoop:
       #pragma unroll
       for (int im = 0, in_index = ir; im < CONFIG_T::block_factor; im++) {
            uint32 w_index = ir + (CONFIG_T::reuse_factor_rounded) * im;
            if (ir + CONFIG_T::reuse_factor * im >= CONFIG_T::n_in*CONFIG_T::n_out) continue;
            mult[im] = product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[in_index], weights[w_index]);
            in_index += CONFIG_T::reuse_factor;
            if (in_index >=  CONFIG_T::n_in) in_index = ir;
       }
       AccumLoop:
       #pragma unroll
       for (int im = 0, out_index = 0, acc_step = 0; im < CONFIG_T::block_factor; im++) {
          acc[out_index] += mult[im];
          if (acc_step + 1 >= CONFIG_T::multiplier_scale) {
            acc_step = 0;
            out_index++;
          } else {
            acc_step++;
          }
       }
    }
    // Cast to "res_t" type
    Result:
    #pragma unroll
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}
template<class data_T, class res_T, typename CONFIG_T>
void dense_resource(
   data_T    data[CONFIG_T::n_in],
   res_T     res[CONFIG_T::n_out],
   const typename CONFIG_T::weight_t  weights[CONFIG_T::reuse_factor_rounded*CONFIG_T::block_factor_rounded],
   const typename CONFIG_T::bias_t    biases[CONFIG_T::n_out]
   )
{
    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_rf_lt<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_rf_gt<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}
}
#endif
