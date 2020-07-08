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

#ifndef NNET_VERIFICATION
#define NNET_VERIFICATION

#include "nnet_common.h"
#include "HLS/hls.h"
#include "HLS/math.h"

namespace nnet {

  template<typename CONFIG_T>
  void softmax_test(float data[CONFIG_T::n_in], float res[CONFIG_T::n_in])
  {
    float m = -INFINITY;
    for (size_t i = 0; i < CONFIG_T::n_in; i++) {
      if (data[i] > m) {
        m = data[i];
      }
    }

    float sum = 0.0;
    for (size_t i = 0; i < CONFIG_T::n_in; i++) {
      sum += expf(data[i] - m);
    }

    float offset = m + logf(sum);
    for (size_t i = 0; i < CONFIG_T::n_in; i++) {
      res[i] = expf(data[i] - offset);
    }
  }
  template<typename CONFIG_T>
  void  relu_test(float data[CONFIG_T::n_in], float res[CONFIG_T::n_in])
  {
      for (int ii=0; ii<CONFIG_T::n_in; ii++) {
          float datareg = data[ii];
          if (datareg > 0) res[ii] = datareg;
          else res[ii] = 0;
      }
  }

  //Standard Dense layer used for testbench verification
  template<typename CONFIG_T>
  void transpose(float arr[CONFIG_T::n_in*CONFIG_T::n_out], float arr_t[CONFIG_T::n_in*CONFIG_T::n_out]) {
      for(int i = 0; i < CONFIG_T::n_in; i++) {
          for(int j = 0; j < CONFIG_T::n_out; j++) {
              arr_t[j * CONFIG_T::n_in + i] = arr[i * CONFIG_T::n_out + j];
          }
      }
  }

  template<typename CONFIG_T>
  void dense_test(
      float data[CONFIG_T::n_in],
      float res[CONFIG_T::n_out],
      float weights[CONFIG_T::n_in*CONFIG_T::n_out],
      float biases[CONFIG_T::n_out])
  {
      float cache;
      float mult[CONFIG_T::n_in*CONFIG_T::n_out];
      float acc[CONFIG_T::n_out];

      //float weights_new[CONFIG_T::n_in*CONFIG_T::n_out];
      //transpose<CONFIG_T>(weights, weights_new);

      // Do the matrix-multiply
      Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
          cache = data[ii];
          Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
              int index = ii*CONFIG_T::n_out+jj;
              mult[index] = cache * weights[index];
          }
      }

      // Initialize accumulator with input biases
      ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
          acc[iacc] = (float) biases[iacc];
      }

      // Accumulate multiplication result
      Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
          Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
              int index = ii*CONFIG_T::n_out+jj;
              acc[jj] += mult[index];
          }
      }
      // Cast to "res_t" type
      Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
          res[ires] = (float) (acc[ires]);
      }
  }
}

#endif
