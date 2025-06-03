#ifndef NNET_CLONE_H
#define NNET_CLONE_H

#include "nnet_common.h"
#include <cstdio>

namespace nnet {

struct broadcast_config {
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 1;
    static const unsigned n_dupl = 2;
};

template <class data_T, class res_T, int N>
void clone_stream(stream<data_T> &data, stream<res_T> &res1, stream<res_T> &res2) {
CloneLoop:
    #pragma ii 1
    for (int i = 0; i < N / data_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < data_T::size; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
        }

        res1.write(out_data1);
        res2.write(out_data2);
    }
}

template <class data_T, class res_T, int N>
void clone_stream(stream<data_T> &data, stream<res_T> &res1, stream<res_T> &res2, stream<res_T> &res3) {
CloneLoop:
    #pragma ii 1
    for (int i = 0; i < N / data_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;

    ClonePack:
        #pragma unroll
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

template <class data_T, class res_T, int N>
void clone_stream(stream<data_T> &data, stream<res_T> &res1, stream<res_T> &res2, stream<res_T> &res3, stream<res_T> &res4) {
CloneLoop:
    #pragma ii 1
    for (int i = 0; i < N / data_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        res_T out_data4;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < data_T::size; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
            out_data4[j] = in_data[j];
        }

        res1.write(out_data1);
        res2.write(out_data2);
        res3.write(out_data3);
        res4.write(out_data4);
    }
}

template <class data_T, class res_T, int N>
void clone_stream(stream<data_T> &data, stream<res_T> &res1, stream<res_T> &res2, stream<res_T> &res3, stream<res_T> &res4,
                  stream<res_T> &res5) {
CloneLoop:
    #pragma ii 1
    for (int i = 0; i < N / data_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        res_T out_data4;
        res_T out_data5;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < data_T::size; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
            out_data4[j] = in_data[j];
            out_data5[j] = in_data[j];
        }

        res1.write(out_data1);
        res2.write(out_data2);
        res3.write(out_data3);
        res4.write(out_data4);
        res5.write(out_data5);
    }
}

template <class data_T, class res_T, int N>
void clone_stream(stream<data_T> &data, stream<res_T> &res1, stream<res_T> &res2, stream<res_T> &res3, stream<res_T> &res4,
                  stream<res_T> &res5, stream<res_T> &res6) {
CloneLoop:
    #pragma ii 1
    for (int i = 0; i < N / data_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        res_T out_data4;
        res_T out_data5;
        res_T out_data6;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < data_T::size; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
            out_data4[j] = in_data[j];
            out_data5[j] = in_data[j];
            out_data6[j] = in_data[j];
        }

        res1.write(out_data1);
        res2.write(out_data2);
        res3.write(out_data3);
        res4.write(out_data4);
        res5.write(out_data5);
        res6.write(out_data6);
    }
}

template <class data_T, class res_T, int N>
void clone_stream(stream<data_T> &data, stream<res_T> &res1, stream<res_T> &res2, stream<res_T> &res3, stream<res_T> &res4,
                  stream<res_T> &res5, stream<res_T> &res6, stream<res_T> &res7) {
CloneLoop:
    #pragma ii 1
    for (int i = 0; i < N / data_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        res_T out_data4;
        res_T out_data5;
        res_T out_data6;
        res_T out_data7;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < data_T::size; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
            out_data4[j] = in_data[j];
            out_data5[j] = in_data[j];
            out_data6[j] = in_data[j];
            out_data7[j] = in_data[j];
        }

        res1.write(out_data1);
        res2.write(out_data2);
        res3.write(out_data3);
        res4.write(out_data4);
        res5.write(out_data5);
        res6.write(out_data6);
        res7.write(out_data7);
    }
}

template <class data_T, class res_T, int N> void repack_stream(stream<data_T> &data, stream<res_T> &res) {
    if (data_T::size == res_T::size) {
        #pragma ii 1
        for (int i = 0; i < N / data_T::size; i++) {

            data_T in_data = data.read();
            res_T out_data;

            #pragma unroll
            for (int j = 0; j < data_T::size; j++) {
                out_data[j] = in_data[j];
            }

            res.write(out_data);
        }
    } else if (data_T::size > res_T::size) {
        constexpr unsigned pack_diff = data_T::size / res_T::size;

        for (int i = 0; i < N / data_T::size; i++) {

            data_T in_data = data.read();
            res_T out_data;

            #pragma ii 1
            for (int j = 0; j < pack_diff; j++) {

                res_T out_data;

                #pragma unroll
                for (int k = 0; k < res_T::size; k++) {
                    out_data[k] = in_data[j * res_T::size + k];
                }
                res.write(out_data);
            }
        }
    } else { // data_T::size < res_T::size
        res_T out_data;
        constexpr unsigned pack_diff = res_T::size / data_T::size;
        unsigned pack_cnt = 0;
        #pragma ii 1
        for (int i = 0; i < N / data_T::size; i++) {

            data_T in_data = data.read();

            #pragma unroll
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

} // namespace nnet

#endif
