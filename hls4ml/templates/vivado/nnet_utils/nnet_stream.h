
#ifndef NNET_STREAM_H
#define NNET_STREAM_H

#include "hls_stream.h"

template<class data_T, class res_T, int N>
void repack_stream(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    if (data_T::size == res_T::size) {
        for (int i = 0; i < N / data_T::size; i++) {
            #pragma HLS PIPELINE

            data_T in_data = data.read();
            res_T out_data;
            #pragma HLS DATA_PACK variable=out_data

            for (int j = 0; j < data_T::size; j++) {
                #pragma HLS UNROLL
                out_data[j] = in_data[j];
            }

            res.write(out_data);
        }
    } else if (data_T::size > res_T::size) {
        constexpr unsigned pack_diff = data_T::size / res_T::size;
        for (int i = 0; i < N / data_T::size; i++) {
           #pragma HLS PIPELINE

            data_T in_data = data.read();
            res_T out_data;
            #pragma HLS DATA_PACK variable=out_data

            for (int j = 0; j < pack_diff; j++) {
                res_T out_data;
                for (int k = 0; k < res_T::size; k++) {
                    #pragma HLS UNROLL
                    out_data[k] = in_data[j * res_T::size + k];
                }
                res.write(out_data);
            }
        }
    } else { // data_T::size < res_T::size
        res_T out_data;
        constexpr unsigned pack_diff = res_T::size / data_T::size;
        unsigned pack_cnt = 0;
        for (int i = 0; i < N / data_T::size; i++) {
            #pragma HLS PIPELINE

            data_T in_data = data.read();
            for (int j = 0; j < data_T::size; j++) {
                #pragma HLS UNROLL
                out_data[pack_cnt * data_T::size + j] = in_data[j];
            }

            if (pack_cnt == pack_diff - 1) {
                res.write(out_data);
                pack_cnt = 0;
            } else {
                pack_cnt++;
            }
        }
    }
}

#endif