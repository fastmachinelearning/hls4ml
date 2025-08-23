#ifndef NNET_STREAM_BEAT_H
#define NNET_STREAM_BEAT_H

// These are functions just for streaming in accelerator mode. They convert from using packets
// to not using packets, and visa versa.

struct sideband_config {
    static const unsigned n_in = 10;
};

// *************************************************
// Remove sideband and passes it to end via skip pipe
// *************************************************
template <class data_pipe, class res_pipe, class skip_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void extract_sideband_stream() {

    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool sop = false;
    bool eop = false;

LinearActLoop:
    [[intel::initiation_interval(1)]] while (!eop) {
        for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
            auto in_data = data_pipe::read();

        LinearPackLoop:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
                out_data[j] = in_data.data[j];
            }

            res_pipe::write(out_data);

            if (i == 0) {
                sop = in_data.sop;
            }
            eop = in_data.eop;
        }
        typename ExtractPipeType<skip_pipe>::value_type skip_data; // this is a two-element array, {sop, eop}.
        skip_data[0] = sop;
        skip_data[1] = eop;
        skip_pipe::write(skip_data);
    }
}

// *************************************************
// Recieves sideband via skip pipe, and makees it sideband
// *************************************************

template <class data_pipe, class res_pipe, class skip_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void merge_sideband_stream() {
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    constexpr auto num_transfers = CONFIG_T::n_in / std::tuple_size<ResT>{};

    auto skip_data = skip_pipe::read();

LinearActLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < num_transfers; i++) {
        auto in_data = data_pipe::read();

    LinearPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<ResT>{}; j++) {
            out_data.data[j] = in_data.data[j];
        }
        out_data.sop = (i == 0) ? skip_data[0] : false;
        out_data.eop = (i == num_transfers - 1) ? skip_data[1] : false;
        res_pipe::write(out_data);
    }
}

#endif
