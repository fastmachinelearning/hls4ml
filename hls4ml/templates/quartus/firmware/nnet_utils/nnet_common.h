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

#ifndef NNET_COMMON_H_
#define NNET_COMMON_H_

#ifndef __INTELFPGA_COMPILER__
#include "ac_int.h"
#include "ac_fixed.h"
#include "math.h"
#else
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#include "HLS/math.h"
#endif

typedef ac_fixed<16,6> table_default_t;

namespace nnet {

// Common type definitions
enum io_type {io_parallel = 0, io_serial};

// Default data types (??) TODO: Deprecate
typedef ac_fixed<16,4>  weight_t_def;
typedef ac_fixed<16,4>  bias_t_def;
typedef ac_fixed<32,10> accum_t_def;

 template<class data_T, int NIN1, int NIN2>
   void merge(
	      data_T data1[NIN1],
	      data_T data2[NIN2],
	      data_T res[NIN1+NIN2])
 {
   #pragma unroll
   for(int ii=0; ii<NIN1; ii++){
     res[ii] = data1[ii];
   }
   #pragma unroll
   for(int ii=0; ii<NIN2; ii++){
     res[NIN1+ii] = data2[ii];
   }
 }

}

#endif
