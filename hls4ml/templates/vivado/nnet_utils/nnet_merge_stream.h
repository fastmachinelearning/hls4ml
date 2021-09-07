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

#ifndef NNET_MERGE_STREAM_H_
#define NNET_MERGE_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    AddLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        AddPack: for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data1[j] + in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void subtract(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    SubtractLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        SubtractPack: for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data1[j] - in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void multiply(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    MultiplyLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        MultiplyPack: for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data1[j] * in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void average(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    AverageLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        AveragePack: for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = (in_data1[j] + in_data2[j]) / (typename res_T::value_type) 2;
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void maximum(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    MaximumLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        MaximumPack: for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = (in_data1[j] > in_data2[j]) ? in_data1[j] : in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void minimum(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    MinimumLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        MinimumPack: for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = (in_data1[j] < in_data2[j]) ? in_data1[j] : in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_0(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    ConcatLoopHeight1: for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        ConcatLoopWidth1: for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #pragma HLS PIPELINE II=1

            input1_T in_data1 = data1.read();
            res_T out_data;
            #pragma HLS DATA_PACK variable=out_data

            ConcatPackInput1: for (int k = 0; k < input1_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data1[k];
            }

            res.write(out_data);
        }
    }
    ConcatLoopHeight2: for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
        ConcatLoopWidth2: for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            #pragma HLS PIPELINE II=1

            input2_T in_data2 = data2.read();
            res_T out_data;
            #pragma HLS DATA_PACK variable=out_data

            ConcatPackInput2: for (int k = 0; k < input2_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data2[k];
            }

            res.write(out_data);
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_1(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    ConcatLoopHeight: for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        ConcatLoopWidth1: for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #pragma HLS PIPELINE II=1

            input1_T in_data1 = data1.read();
            res_T out_data;
            #pragma HLS DATA_PACK variable=out_data

            ConcatPackInput1: for (int k = 0; k < input1_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data1[k];
            }

            res.write(out_data);
        }
        ConcatLoopWidth2: for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            #pragma HLS PIPELINE II=1

            input2_T in_data2 = data2.read();
            res_T out_data;
            #pragma HLS DATA_PACK variable=out_data

            ConcatPackInput2: for (int k = 0; k < input2_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data2[k];
            }

            res.write(out_data);
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_2(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    ConcatLoopHeight: for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        ConcatLoopWidth: for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #pragma HLS PIPELINE II=1

            input1_T in_data1 = data1.read();
            input2_T in_data2 = data2.read();
            res_T out_data;
            #pragma HLS DATA_PACK variable=out_data

            ConcatPackInput1: for (int k = 0; k < input1_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data1[k];
            }
            
            ConcatPackInput2: for (int k = 0; k < input2_T::size; k++) {
                #pragma HLS UNROLL
                out_data[input1_T::size + k] = in_data2[k];
            }

            res.write(out_data);
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
    if (CONFIG_T::axis == 3 || CONFIG_T::axis == -1) {
        concatenate3d_2<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else if (CONFIG_T::axis == 2 || CONFIG_T::axis == -2) {
        concatenate3d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate3d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

}

#endif
