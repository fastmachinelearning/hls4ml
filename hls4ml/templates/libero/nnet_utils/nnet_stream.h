#ifndef NNET_STREAM_H
#define NNET_STREAM_H

#include "nnet_common.h"
#include <hls/streaming.hpp>

namespace nnet {

struct broadcast_config {
    static const unsigned in_height = 1;
    static const unsigned in_width = 1;
    static const unsigned in_chan = 3;
    static const unsigned out_height = 2;
    static const unsigned out_width = 2;
    static const unsigned out_chan = 3;
};

template <class data_T, class res_T, int N>
void clone_stream(hls::FIFO<data_T> &data, hls::FIFO<res_T> &res1, hls::FIFO<res_T> &res2) {
CloneLoop:
    #pragma HLS loop pipeline
    for (int i = 0; i < N / data_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        // PRAGMA_DATA_PACK(out_data1)
        // PRAGMA_DATA_PACK(out_data2)

    ClonePack:
        #pragma HLS loop unroll
        for (int j = 0; j < data_T::size; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
        }

        res1.write(out_data1);
        res2.write(out_data2);
    }
}

template <class data_T, class res_T, int N>
void clone_stream(hls::FIFO<data_T> &data, hls::FIFO<res_T> &res1, hls::FIFO<res_T> &res2, hls::FIFO<res_T> &res3) {
CloneLoop:
    #pragma HLS loop pipeline
    for (int i = 0; i < N / data_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        // PRAGMA_DATA_PACK(out_data1)
        // PRAGMA_DATA_PACK(out_data2)
        // PRAGMA_DATA_PACK(out_data3)

    ClonePack:
        #pragma HLS loop unroll
        for (int j = 0; j < data_T::size; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
        }

        res1.write(out_data1);
        res2.write(out_data2);
        res3.write(out_data3);
    }
}

template <class data_T, class res_T, int N> void repack_stream(hls::FIFO<data_T> &data, hls::FIFO<res_T> &res) {
    if (data_T::size == res_T::size) {
        #pragma HLS loop pipeline
        for (int i = 0; i < N / data_T::size; i++) {

            data_T in_data = data.read();
            res_T out_data;
            // PRAGMA_DATA_PACK(out_data)

            #pragma HLS loop unroll
            for (int j = 0; j < data_T::size; j++) {
                out_data[j] = in_data[j];
            }

            res.write(out_data);
        }
    } else if (data_T::size > res_T::size) {
        constexpr unsigned pack_diff = data_T::size / res_T::size;
        if (N / data_T::size > 1) {
            #pragma HLS loop pipeline
            for (int i = 0; i < N / data_T::size; i++) {

                data_T in_data = data.read();
                res_T out_data;
                // PRAGMA_DATA_PACK(out_data)

                #pragma HLS loop pipeline
                for (int j = 0; j < pack_diff; j++) {

                    res_T out_data;
                    #pragma HLS loop unroll
                    for (int k = 0; k < res_T::size; k++) {
                        out_data[k] = in_data[j * res_T::size + k];
                    }
                    res.write(out_data);
                }
            }
        } else {
            for (int i = 0; i < N / data_T::size; i++) {

                data_T in_data = data.read();
                res_T out_data;
                // PRAGMA_DATA_PACK(out_data)

                #pragma HLS loop pipeline
                for (int j = 0; j < pack_diff; j++) {

                    res_T out_data;
                    #pragma HLS loop unroll
                    for (int k = 0; k < res_T::size; k++) {
                        out_data[k] = in_data[j * res_T::size + k];
                    }
                    res.write(out_data);
                }
            }
        }
    } else { // data_T::size < res_T::size
        res_T out_data;
        constexpr unsigned pack_diff = res_T::size / data_T::size;
        unsigned pack_cnt = 0;
        #pragma HLS loop pipeline
        for (int i = 0; i < N / data_T::size; i++) {

            data_T in_data = data.read();
            #pragma HLS loop unroll
            for (int j = 0; j < data_T::size; j++) {
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

template <class data_T, class res_T, typename CONFIG_T>
void broadcast_stream_1x1xC(hls::FIFO<data_T> &data, hls::FIFO<res_T> &res) {
    assert(CONFIG_T::in_height == 1 && CONFIG_T::in_width == 1 && CONFIG_T::in_chan == CONFIG_T::out_chan);
    int n_dupl = (CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::out_chan) /
                 (CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::in_chan);
BroadcastLoop:
    #pragma HLS loop pipeline
    for (int i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::in_chan / data_T::size; i++) {
        data_T in_data = data.read();
        #pragma HLS loop pipeline
        for (int j = 0; j < n_dupl; j++) {
            res_T out_data;
            // PRAGMA_DATA_PACK(out_data)
            #pragma HLS loop unroll
            for (int k = 0; k < res_T::size; k++) {
                out_data[k] = in_data[k];
            }
            res.write(out_data);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void broadcast_stream_HxWx1(hls::FIFO<data_T> &data, hls::FIFO<res_T> &res) {
    assert(CONFIG_T::in_chan == 1 && CONFIG_T::in_height == CONFIG_T::out_height &&
           CONFIG_T::in_width == CONFIG_T::out_width);
BroadcastLoop:
    #pragma HLS loop pipeline
    for (int i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::in_chan / data_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
        // PRAGMA_DATA_PACK(out_data)
        #pragma HLS loop unroll
        for (int k = 0; k < res_T::size; k++) {
            out_data[k] = in_data[0];
        }
        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void broadcast_stream(hls::FIFO<data_T> &data, hls::FIFO<res_T> &res) {
    if (CONFIG_T::in_height == 1 && CONFIG_T::in_width == 1 && CONFIG_T::in_chan == CONFIG_T::out_chan) {
        broadcast_stream_1x1xC<data_T, res_T, CONFIG_T>(data, res);
    } else if (CONFIG_T::in_chan == 1 && CONFIG_T::in_height == CONFIG_T::out_height &&
               CONFIG_T::in_width == CONFIG_T::out_width) {
        broadcast_stream_HxWx1<data_T, res_T, CONFIG_T>(data, res);
    }
}

template <class data_T, class res_T, typename CONFIG_T> void transpose_2d(hls::FIFO<data_T> &data, hls::FIFO<res_T> &res) {
    #pragma HLS memory partition variable(data_array) type(complete)
    typename data_T::value_type data_array[CONFIG_T::height * CONFIG_T::width];

    #pragma HLS loop pipeline
    for (int i = 0; i < CONFIG_T::height * CONFIG_T::width / data_T::size; i++) {
        data_T in_data = data.read();
        for (int j = 0; j < data_T::size; j++) {
            data_array[i * data_T::size + j] = typename data_T::value_type(in_data[j]);
        }
    }

    #pragma HLS loop pipeline
    for (int i = 0; i < CONFIG_T::height * CONFIG_T::width / res_T::size; i++) {
        res_T out_data;
        // PRAGMA_DATA_PACK(out_data)
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = typename res_T::value_type(data_array[j * data_T::size + i]);
        }
        res.write(out_data);
    }
}
} // namespace nnet

#endif
