#ifndef NNET_TRANSPOSE_STREAM_H_
#define NNET_TRANSPOSE_STREAM_H_

namespace nnet {

template <class data_pipe, class res_pipe, typename CONFIG_T> void transpose_2d_stream() {

    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::value_type;

    constexpr auto data_size = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{};
    constexpr auto res_size = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};

    [[intel::fpga_register]] typename data_T::value_type data_array[CONFIG_T::height * CONFIG_T::width];

    for (int i = 0; i < CONFIG_T::height * CONFIG_T::width / data_size; i++) {
        [[intel::fpga_register]] data_T in_data = data_pipe::read();

        #pragma unroll
        for (int j = 0; j < data_size; j++) {
            data_array[i * data_size + j] = typename data_T::value_type(in_data[j]);
        }
    }

    for (int i = 0; i < CONFIG_T::height * CONFIG_T::width / res_size; i++) {
        [[intel::fpga_register]] res_T out_data;

        #pragma unroll
        for (int j = 0; j < res_size; j++) {
            out_data[j] = typename res_T::value_type(data_array[j * data_size + i]);
        }

        res_pipe::write(out_data);
    }
}

} // namespace nnet

#endif
