#ifndef NNET_MERGE_STREAM_H_
#define NNET_MERGE_STREAM_H_

namespace nnet {

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void add_stream() {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

AddLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
        [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();

        [[intel::fpga_register]] res_T out_data;

    AddPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(in_data1[j] + in_data2[j]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void subtract_stream() {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

SubtractLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
        [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();

        [[intel::fpga_register]] res_T out_data;

    SubtractPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(in_data1[j] - in_data2[j]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void multiply_stream() {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

MultLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
        [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();

        [[intel::fpga_register]] res_T out_data;

    MultPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(in_data1[j] * in_data2[j]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void average_stream() {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

AvgLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
        [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();

        [[intel::fpga_register]] res_T out_data;

    AvgPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] =
                static_cast<typename res_T::value_type>((in_data1[j] + in_data2[j]) / (typename res_T::value_type)2);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void maximum_stream() {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

MaxLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
        [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();

        [[intel::fpga_register]] res_T out_data;

    MaxPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(out_data[j] = (in_data1[j] > in_data2[j]) ? in_data1[j]
                                                                                                            : in_data2[j]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void minimum_stream() {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

MinLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
        [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();

        [[intel::fpga_register]] res_T out_data;

    MinPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(out_data[j] = (in_data1[j] < in_data2[j]) ? in_data1[j]
                                                                                                            : in_data2[j]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate1d_stream() {
    [[intel::fpga_register]] res_T out_data;

ConcatLoop1:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem1_0 / input1_T::size; i++) {
        [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
    ConcatPack1:
        #pragma unroll
        for (int j = 0; j < input1_T::size; j++) {
            out_data[j + (i * input1_T::size)] = static_cast<typename res_T::value_type>(in_data1[j]);
        }
    }

ConcatLoop2:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem2_0 / input2_T::size; i++) {
        [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();
    ConcatPack2:
        #pragma unroll
        for (int j = 0; j < input2_T::size; j++) {
            out_data[j + (i * input2_T::size) + (CONFIG_T::n_elem1_0)] =
                static_cast<typename res_T::value_type>(in_data2[j]);
        }
    }
    res_pipe::write(out_data);
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate2d_0_stream() {
ConcatLoopHeight1:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {

        [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
        [[intel::fpga_register]] res_T out_data;

    ConcatPackInput1:
        #pragma unroll
        for (int k = 0; k < input1_T::size; k++) {
            out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
        }

        res_pipe::write(out_data);
    }

ConcatLoopHeight2:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
        [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();
        [[intel::fpga_register]] res_T out_data;

    ConcatPackInput2:
        #pragma unroll
        for (int k = 0; k < input2_T::size; k++) {
            out_data[k] = static_cast<typename res_T::value_type>(in_data2[k]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate2d_1_stream() {
ConcatLoopHeight:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
        [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();
        [[intel::fpga_register]] res_T out_data;

    ConcatPackInput1:
        #pragma unroll
        for (int k = 0; k < input1_T::size; k++) {
            out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
        }

    ConcatPackInput2:
        #pragma unroll
        for (int k = 0; k < input2_T::size; k++) {
            out_data[input1_T::size + k] = static_cast<typename res_T::value_type>(in_data2[k]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate2d_stream() {
    if (CONFIG_T::axis == 2 || CONFIG_T::axis == -1) {
        concatenate2d_1_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    } else {
        concatenate2d_0_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_0_stream() {
ConcatLoopHeight1:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth1:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
            [[intel::fpga_register]] res_T out_data;
        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
            }

            res_pipe::write(out_data);
        }
    }

ConcatLoopHeight2:
    for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
    ConcatLoopWidth2:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {

            [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();
            [[intel::fpga_register]] res_T out_data;

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data2[k]);
            }

            res_pipe::write(out_data);
        }
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_1_stream() {
ConcatLoopHeight:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth1:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
            [[intel::fpga_register]] res_T out_data;

        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
            }

            res_pipe::write(out_data);
        }
    ConcatLoopWidth2:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {

            [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();
            [[intel::fpga_register]] res_T out_data;

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data2[k]);
            }

            res_pipe::write(out_data);
        }
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_2_stream() {
ConcatLoopHeight:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            [[intel::fpga_register]] input1_T in_data1 = input1_pipe::read();
            [[intel::fpga_register]] input2_T in_data2 = input2_pipe::read();
            [[intel::fpga_register]] res_T out_data;

        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
            }

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2_T::size; k++) {
                out_data[input1_T::size + k] = static_cast<typename res_T::value_type>(in_data2[k]);
            }

            res_pipe::write(out_data);
        }
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_stream() {
    if (CONFIG_T::axis == 3 || CONFIG_T::axis == -1) {
        concatenate3d_2_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    } else if (CONFIG_T::axis == 2 || CONFIG_T::axis == -2) {
        concatenate3d_1_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    } else {
        concatenate3d_0_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    }
}

} // namespace nnet

#endif
