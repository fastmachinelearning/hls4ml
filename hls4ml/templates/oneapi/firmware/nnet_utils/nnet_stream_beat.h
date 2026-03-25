#ifndef NNET_STREAM_BEAT_H
#define NNET_STREAM_BEAT_H

// These are functions just for streaming in accelerator mode. They convert from using packets
// to not using packets, and visa versa.

namespace nnet {

struct sideband_config {
    static const unsigned n_in = 10;
};

// *************************************************
// Remove sideband and passes it to end via return
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] bool extract_sideband_stream() {

    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool eop = false;

LinearActLoop:
    [[intel::initiation_interval(
        1)]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();

    LinearPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            out_data[j] = in_data.data[j];
        }

        res_pipe::write(out_data);

        eop = in_data.eop;
    }
    return (!eop);
}

// *************************************************
// Recieves sideband via call argument, and makes it sideband
// *************************************************

template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void merge_sideband_stream(bool keep_going, uint32_t count) {
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    constexpr auto num_transfers = CONFIG_T::n_in / std::tuple_size<ResT>{};

LinearActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < num_transfers; i++) {
        auto in_data = data_pipe::read();

    LinearPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
            out_data.data[j] = in_data[j];
        }
        out_data.sop = (i == 0) ? count == 0 : false;
        out_data.eop = (i == num_transfers - 1) ? !keep_going : false;
        res_pipe::write(out_data);
    }
}
} // namespace nnet
#endif
