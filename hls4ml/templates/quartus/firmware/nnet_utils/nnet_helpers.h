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

#ifndef NNET_HELPERS_H
#define NNET_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <map>

#ifndef __INTELFPGA_COMPILER__
#include "stream.h"
template<typename T>
using stream = nnet::stream<T>;
template<typename T>
using stream_in = nnet::stream<T>;
template<typename T>
using stream_out = nnet::stream<T>;
#else
#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
template<typename T>
using stream = ihc::stream<T>;
template<typename T>
using stream_in = ihc::stream_in<T>;
template<typename T>
using stream_out = ihc::stream_out<T>;
#endif

namespace nnet {

template<class srcType, class dstType, size_t SIZE>
void convert_data(srcType *src, dstType *dst) {
  for (size_t i = 0; i < SIZE; i++) {
    dst[i] = dstType(src[i]);
  }
}

template<class srcType, class dstType, size_t SIZE>
void convert_data_back(srcType *src, dstType *dst) {
  for (size_t i = 0; i < SIZE; i++) {
    dst[i] = static_cast<dstType>(src[i].to_double());
  }
}

template<class srcType, class dstType, size_t SIZE>
void convert_data(srcType *src, stream_in<dstType> &dst) {
    for (size_t i = 0; i < SIZE / dstType::size; i++) {
        dstType ctype;
        for (size_t j = 0; j < dstType::size; j++) {
            ctype[j] = typename dstType::value_type(src[i * dstType::size + j]);
        }
        dst.write(ctype);
    }
}

template<class srcType, class dstType, size_t SIZE>
void convert_data_back(stream_out<srcType> &src, dstType *dst) {
    for (size_t i = 0; i < SIZE / srcType::size; i++) {
        srcType ctype = src.read();
        for (size_t j = 0; j < srcType::size; j++) {
            dst[i * srcType::size + j] = dstType(ctype[j].to_double());
        }
    }
}

extern bool trace_enabled;
extern std::map<std::string, void *> *trace_outputs;
extern size_t trace_type_size;

constexpr int ceillog2(int x){
  return (x <= 2) ? 1 : 1 + ceillog2((x+1) / 2);
}

constexpr int floorlog2(int x){
  return (x < 2) ? 0 : 1 + floorlog2(x / 2);
}

constexpr int pow2(int x){
  return x == 0 ? 1 : 2 * pow2(x - 1);
}

}

#endif
