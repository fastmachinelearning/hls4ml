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
#include "hls_stream.h"

namespace nnet {

#ifndef __SYNTHESIS__

#define WEIGHTS_DIR "weights"

template<class T, size_t SIZE>
void load_txt_file(T *w, const char* fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname);
        std::cerr << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        for (size_t i = 0; i < SIZE; i++) {
            iss >> w[i];
        }
    }
}

#endif

template <class dataType, unsigned int nrows>
int read_file_1D(const char * filename, dataType data[nrows])
{
  FILE *fp;
  fp = fopen(filename, "r");
  if (fp == 0) {
    return -1;
  }
  // Read data from file
  float newval;
  for (int ii = 0; ii < nrows; ii++){
    if (fscanf(fp, "%f\n", &newval) != 0){
      data[ii] = newval;
    } else {
      return -2;
    }
  }
  fclose(fp);
  return 0;
}

template <class dataType, unsigned int nrows, unsigned int ncols>
int read_file_2D(const char * filename, dataType data[nrows][ncols])
{
  FILE *fp;
  fp = fopen(filename, "r");
  if (fp == 0) {
    return -1;
  }
  // Read data from file
  float newval;
  for (int ii = 0; ii < nrows; ii++) {
    for (int jj = 0; jj < ncols; jj++){
      if (fscanf(fp, "%f\n", &newval) != 0){
        data[ii][jj] = newval;
      } else {
        return -2;
      }
    }
  }
  fclose(fp);
  return 0;
}

template<class in_T, class out_T, int N_IN>
void change_type(hls::stream<in_T> &in, hls::stream<out_T> &out)
{
    in_T datareg;
    hls::stream<out_T> input_trunc;
    for (int ii=0; ii<N_IN; ii++) {
        out << (out_T) in.read();
    }
}

template<class data_T, int N_IN>
void  hls_stream_debug(hls::stream<data_T> &data, hls::stream<data_T> &res)
{
    data_T datareg;
    for (int ii=0; ii<N_IN; ii++) {
        datareg = data.read();
        std::cout << "[" << ii << "]: " << datareg << std::endl;
        res << datareg;
    }
}

constexpr int ceillog2(int x){
  return (x <= 2) ? 1 : 1 + ceillog2((x+1) / 2);
}

constexpr int pow2(int x){
  return x == 0 ? 1 : 2 * pow2(x - 1);
}

}

#endif
