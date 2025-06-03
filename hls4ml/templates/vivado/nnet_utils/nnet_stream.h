
#ifndef NNET_STREAM_H
#define NNET_STREAM_H

#include "hls_stream.h"
#include "nnet_common.h"

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
void clone_stream(hls::stream<data_T> &data, hls::stream<res_T> &res1, hls::stream<res_T> &res2) {
CloneLoop:
    for (int i = 0; i < N / data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        PRAGMA_DATA_PACK(out_data1)
        PRAGMA_DATA_PACK(out_data2)

    ClonePack:
        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
        }

        res1.write(out_data1);
        res2.write(out_data2);
    }
}

template <class data_T, class res_T, int N>
void clone_stream(hls::stream<data_T> &data, hls::stream<res_T> &res1, hls::stream<res_T> &res2, hls::stream<res_T> &res3) {
CloneLoop:
    for (int i = 0; i < N / data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        PRAGMA_DATA_PACK(out_data1)
        PRAGMA_DATA_PACK(out_data2)
        PRAGMA_DATA_PACK(out_data3)

    ClonePack:
        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
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
void clone_stream(hls::stream<data_T> &data, hls::stream<res_T> &res1, hls::stream<res_T> &res2, hls::stream<res_T> &res3,
                  hls::stream<res_T> &res4) {
CloneLoop:
    for (int i = 0; i < N / data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        res_T out_data4;
        PRAGMA_DATA_PACK(out_data1)
        PRAGMA_DATA_PACK(out_data2)
        PRAGMA_DATA_PACK(out_data3)
        PRAGMA_DATA_PACK(out_data4)

    ClonePack:
        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
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
void clone_stream(hls::stream<data_T> &data, hls::stream<res_T> &res1, hls::stream<res_T> &res2, hls::stream<res_T> &res3,
                  hls::stream<res_T> &res4, hls::stream<res_T> &res5) {
CloneLoop:
    for (int i = 0; i < N / data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        res_T out_data4;
        res_T out_data5;
        PRAGMA_DATA_PACK(out_data1)
        PRAGMA_DATA_PACK(out_data2)
        PRAGMA_DATA_PACK(out_data3)
        PRAGMA_DATA_PACK(out_data4)
        PRAGMA_DATA_PACK(out_data5)

    ClonePack:
        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
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
void clone_stream(hls::stream<data_T> &data, hls::stream<res_T> &res1, hls::stream<res_T> &res2, hls::stream<res_T> &res3,
                  hls::stream<res_T> &res4, hls::stream<res_T> &res5, hls::stream<res_T> &res6) {
CloneLoop:
    for (int i = 0; i < N / data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        res_T out_data4;
        res_T out_data5;
        res_T out_data6;
        PRAGMA_DATA_PACK(out_data1)
        PRAGMA_DATA_PACK(out_data2)
        PRAGMA_DATA_PACK(out_data3)
        PRAGMA_DATA_PACK(out_data4)
        PRAGMA_DATA_PACK(out_data5)
        PRAGMA_DATA_PACK(out_data6)

    ClonePack:
        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
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
void clone_stream(hls::stream<data_T> &data, hls::stream<res_T> &res1, hls::stream<res_T> &res2, hls::stream<res_T> &res3,
                  hls::stream<res_T> &res4, hls::stream<res_T> &res5, hls::stream<res_T> &res6, hls::stream<res_T> &res7) {
CloneLoop:
    for (int i = 0; i < N / data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data1;
        res_T out_data2;
        res_T out_data3;
        res_T out_data4;
        res_T out_data5;
        res_T out_data6;
        res_T out_data7;
        PRAGMA_DATA_PACK(out_data1)
        PRAGMA_DATA_PACK(out_data2)
        PRAGMA_DATA_PACK(out_data3)
        PRAGMA_DATA_PACK(out_data4)
        PRAGMA_DATA_PACK(out_data5)
        PRAGMA_DATA_PACK(out_data6)
        PRAGMA_DATA_PACK(out_data7)

    ClonePack:
        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
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

template <class data_T, class res_T, int N> void repack_stream(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    if (data_T::size == res_T::size) {
        for (int i = 0; i < N / data_T::size; i++) {
            #pragma HLS PIPELINE

            data_T in_data = data.read();
            res_T out_data;
            PRAGMA_DATA_PACK(out_data)

            for (int j = 0; j < data_T::size; j++) {
                #pragma HLS UNROLL
                out_data[j] = in_data[j];
            }

            res.write(out_data);
        }
    } else if (data_T::size > res_T::size) {
        constexpr unsigned pack_diff = data_T::size / res_T::size;
        for (int i = 0; i < N / data_T::size; i++) {
            if (N / data_T::size > 1) {
                #pragma HLS PIPELINE
            }

            data_T in_data = data.read();
            res_T out_data;
            PRAGMA_DATA_PACK(out_data)

            for (int j = 0; j < pack_diff; j++) {
                #pragma HLS PIPELINE

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

template <class data_T, class res_T, typename CONFIG_T>
void broadcast_stream_1x1xC(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::in_height == 1 && CONFIG_T::in_width == 1 && CONFIG_T::in_chan == CONFIG_T::out_chan);
    int n_dupl = (CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::out_chan) /
                 (CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::in_chan);
BroadcastLoop:
    for (int i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::in_chan / data_T::size; i++) {
        #pragma HLS PIPELINE
        data_T in_data = data.read();
        for (int j = 0; j < n_dupl; j++) {
            #pragma HLS PIPELINE
            res_T out_data;
            PRAGMA_DATA_PACK(out_data)
            for (int k = 0; k < res_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data[k];
            }
            res.write(out_data);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void broadcast_stream_HxWx1(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::in_chan == 1 && CONFIG_T::in_height == CONFIG_T::out_height &&
           CONFIG_T::in_width == CONFIG_T::out_width);
BroadcastLoop:
    for (int i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::in_chan / data_T::size; i++) {
        #pragma HLS PIPELINE
        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)
        for (int k = 0; k < res_T::size; k++) {
            #pragma HLS UNROLL
            out_data[k] = in_data[0];
        }
        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void broadcast_stream(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    if (CONFIG_T::in_height == 1 && CONFIG_T::in_width == 1 && CONFIG_T::in_chan == CONFIG_T::out_chan) {
        broadcast_stream_1x1xC<data_T, res_T, CONFIG_T>(data, res);
    } else if (CONFIG_T::in_chan == 1 && CONFIG_T::in_height == CONFIG_T::out_height &&
               CONFIG_T::in_width == CONFIG_T::out_width) {
        broadcast_stream_HxWx1<data_T, res_T, CONFIG_T>(data, res);
    }
}

} // namespace nnet

#endif
