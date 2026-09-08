#ifndef NNET_CLONE_H
#define NNET_CLONE_H

#include "nnet_common.h"

namespace nnet {

struct broadcast_config {
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 1;
    static const unsigned n_dupl = 2;
};

template <class data_pipe, class res1_pipe, class res2_pipe, int N> void clone_stream() {
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res1_T = typename ExtractPipeType<res1_pipe>::value_type;
    using res2_T = typename ExtractPipeType<res2_pipe>::value_type;
    constexpr auto datasize = std::tuple_size<data_T>{};
CloneLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < N / datasize; i++) {
        data_T in_data = data_pipe::read();
        res1_T out_data1;
        res2_T out_data2;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < datasize; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
        }

        res1_pipe::write(out_data1);
        res2_pipe::write(out_data2);
    }
}

template <class data_pipe, class res1_pipe, class res2_pipe, class res3_pipe, int N> void clone_stream() {
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res1_T = typename ExtractPipeType<res1_pipe>::value_type;
    using res2_T = typename ExtractPipeType<res2_pipe>::value_type;
    using res3_T = typename ExtractPipeType<res3_pipe>::value_type;
    constexpr auto datasize = std::tuple_size<data_T>{};
CloneLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < N / datasize; i++) {
        data_T in_data = data_pipe::read();
        res1_T out_data1;
        res2_T out_data2;
        res3_T out_data3;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < datasize; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
        }

        res1_pipe::write(out_data1);
        res2_pipe::write(out_data2);
        res3_pipe::write(out_data3);
    }
}

template <class data_pipe, class res1_pipe, class res2_pipe, class res3_pipe, class res4_pipe, int N> void clone_stream() {
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res1_T = typename ExtractPipeType<res1_pipe>::value_type;
    using res2_T = typename ExtractPipeType<res2_pipe>::value_type;
    using res3_T = typename ExtractPipeType<res3_pipe>::value_type;
    using res4_T = typename ExtractPipeType<res4_pipe>::value_type;
    constexpr auto datasize = std::tuple_size<data_T>{};
CloneLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < N / datasize; i++) {
        data_T in_data = data_pipe::read();
        res1_T out_data1;
        res2_T out_data2;
        res3_T out_data3;
        res4_T out_data4;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < datasize; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
            out_data4[j] = in_data[j];
        }

        res1_pipe::write(out_data1);
        res2_pipe::write(out_data2);
        res3_pipe::write(out_data3);
        res4_pipe::write(out_data4);
    }
}

template <class data_pipe, class res1_pipe, class res2_pipe, class res3_pipe, class res4_pipe, class res5_pipe, int N>
void clone_stream() {
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res1_T = typename ExtractPipeType<res1_pipe>::value_type;
    using res2_T = typename ExtractPipeType<res2_pipe>::value_type;
    using res3_T = typename ExtractPipeType<res3_pipe>::value_type;
    using res4_T = typename ExtractPipeType<res4_pipe>::value_type;
    using res5_T = typename ExtractPipeType<res5_pipe>::value_type;
    constexpr auto datasize = std::tuple_size<data_T>{};
CloneLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < N / datasize; i++) {
        data_T in_data = data_pipe::read();
        res1_T out_data1;
        res2_T out_data2;
        res3_T out_data3;
        res4_T out_data4;
        res5_T out_data5;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < datasize; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
            out_data4[j] = in_data[j];
            out_data5[j] = in_data[j];
        }

        res1_pipe::write(out_data1);
        res2_pipe::write(out_data2);
        res3_pipe::write(out_data3);
        res4_pipe::write(out_data4);
        res5_pipe::write(out_data5);
    }
}

template <class data_pipe, class res1_pipe, class res2_pipe, class res3_pipe, class res4_pipe, class res5_pipe,
          class res6_pipe, int N>
void clone_stream() {
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res1_T = typename ExtractPipeType<res1_pipe>::value_type;
    using res2_T = typename ExtractPipeType<res2_pipe>::value_type;
    using res3_T = typename ExtractPipeType<res3_pipe>::value_type;
    using res4_T = typename ExtractPipeType<res4_pipe>::value_type;
    using res5_T = typename ExtractPipeType<res5_pipe>::value_type;
    using res6_T = typename ExtractPipeType<res6_pipe>::value_type;
    constexpr auto datasize = std::tuple_size<data_T>{};
CloneLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < N / datasize; i++) {
        data_T in_data = data_pipe::read();
        res1_T out_data1;
        res2_T out_data2;
        res3_T out_data3;
        res4_T out_data4;
        res5_T out_data5;
        res6_T out_data6;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < datasize; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
            out_data4[j] = in_data[j];
            out_data5[j] = in_data[j];
            out_data6[j] = in_data[j];
        }

        res1_pipe::write(out_data1);
        res2_pipe::write(out_data2);
        res3_pipe::write(out_data3);
        res4_pipe::write(out_data4);
        res5_pipe::write(out_data5);
        res6_pipe::write(out_data6);
    }
}

template <class data_pipe, class res1_pipe, class res2_pipe, class res3_pipe, class res4_pipe, class res5_pipe,
          class res6_pipe, class res7_pipe, int N>
void clone_stream() {
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res1_T = typename ExtractPipeType<res1_pipe>::value_type;
    using res2_T = typename ExtractPipeType<res2_pipe>::value_type;
    using res3_T = typename ExtractPipeType<res3_pipe>::value_type;
    using res4_T = typename ExtractPipeType<res4_pipe>::value_type;
    using res5_T = typename ExtractPipeType<res5_pipe>::value_type;
    using res6_T = typename ExtractPipeType<res6_pipe>::value_type;
    using res7_T = typename ExtractPipeType<res7_pipe>::value_type;
    constexpr auto datasize = std::tuple_size<data_T>{};
CloneLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < N / datasize; i++) {
        data_T in_data = data_pipe::read();
        res1_T out_data1;
        res2_T out_data2;
        res3_T out_data3;
        res4_T out_data4;
        res5_T out_data5;
        res6_T out_data6;
        res7_T out_data7;

    ClonePack:
        #pragma unroll
        for (int j = 0; j < datasize; j++) {
            out_data1[j] = in_data[j];
            out_data2[j] = in_data[j];
            out_data3[j] = in_data[j];
            out_data4[j] = in_data[j];
            out_data5[j] = in_data[j];
            out_data6[j] = in_data[j];
            out_data7[j] = in_data[j];
        }

        res1_pipe::write(out_data1);
        res2_pipe::write(out_data2);
        res3_pipe::write(out_data3);
        res4_pipe::write(out_data4);
        res5_pipe::write(out_data5);
        res6_pipe::write(out_data6);
        res7_pipe::write(out_data7);
    }
}

template <class data_pipe, class res_pipe, int N> void repack_stream() {
    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::value_type;
    constexpr auto datasize = std::tuple_size<data_T>{};
    constexpr auto ressize = std::tuple_size<res_T>{};

    if constexpr (datasize == ressize) {
        [[intel::initiation_interval(1)]] for (int i = 0; i < N / datasize; i++) {

            [[intel::fpga_memory]] auto in_data = data_pipe::read();
            [[intel::fpga_memory]] res_T out_data;

            #pragma unroll
            for (int j = 0; j < datasize; j++) {
                out_data[j] = in_data[j];
            }

            res_pipe::write(out_data);
        }
    } else if constexpr (datasize > ressize) {
        constexpr unsigned pack_diff = datasize / ressize;

        for (int i = 0; i < N / datasize; i++) {

            [[intel::fpga_memory]] auto in_data = data_pipe::read();
            [[intel::fpga_memory]] res_T out_data;

            [[intel::initiation_interval(1)]] for (int j = 0; j < pack_diff; j++) {

                #pragma unroll
                for (int k = 0; k < ressize; k++) {
                    out_data[k] = in_data[j * ressize + k];
                }
                res_pipe::write(out_data);
            }
        }
    } else { // datasize < ressize
        [[intel::fpga_memory]] res_T out_data;
        constexpr unsigned pack_diff = ressize / datasize;
        unsigned pack_cnt = 0;
        [[intel::initiation_interval(1)]] for (int i = 0; i < N / datasize; i++) {

            [[intel::fpga_memory]] auto in_data = data_pipe::read();

            #pragma unroll
            for (int j = 0; j < datasize; j++) {
                out_data[pack_cnt * datasize + j] = in_data[j];
            }

            if (pack_cnt == pack_diff - 1) {
                res_pipe::write(out_data);
                pack_cnt = 0;
            } else {
                pack_cnt++;
            }
        }
    }
}

} // namespace nnet

#endif
