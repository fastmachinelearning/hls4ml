#ifndef NNET_BATCHNORM_STREAM_H_
#define NNET_BATCHNORM_STREAM_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_types.h"

namespace nnet {

// ****************************************************
//       Streaming Batch Normalization
// ****************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void normalize_stream(typename CONFIG_T::scale_t scale, typename CONFIG_T::bias_t bias) {
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = CONFIG_T::n_in / multiplier_limit;
    constexpr auto datasize = std::tuple_size<DataT>{};
    CONFIG_T::template product<typename DataT::value_type, typename CONFIG_T::scale_t::value_type>::limit(multiplier_limit);

    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

BatchNormLoop:
    [[intel::initiation_interval(pipeline)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / datasize; i++) {
            auto in_data = data_pipe::read();

        BatchNormpack:
            #pragma unroll
            for (int j = 0; j < datasize; j++) {
                int norm_index;
                if (CONFIG_T::n_filt == -1)
                    norm_index = i * datasize + j;
                else
                    norm_index = j % CONFIG_T::n_filt;
                out_data.data[j] =
                    CONFIG_T::template product<typename DataT::value_type, typename CONFIG_T::scale_t::value_type>::product(
                        in_data.data[j], scale[norm_index]) +
                    bias[norm_index];
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

// ****************************************************
//       Merged Batch Normalization and Quantized Tanh
// ****************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void normalize_binary_tanh_stream(typename CONFIG_T::threshold_t threshold) {
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    constexpr auto datasize = std::tuple_size<DataT>{};

    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

BinaryNormLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / datasize; i++) {
            auto in_data = data_pipe::read();

        BatchNormPack:
            #pragma unroll
            for (int j = 0; j < datasize; j++) {
                int norm_index;
                if (CONFIG_T::n_filt == -1)
                    norm_index = i * datasize + j;
                else
                    norm_index = j % CONFIG_T::n_filt;

                out_data.data[j] = (in_data.data[j] >= threshold[norm_index]) ? 1 : 0;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T>
void normalize_ternary_tanh_stream(typename CONFIG_T::threshold_hi_t threshold_hi,
                                   typename CONFIG_T::threshold_lo_t threshold_lo) {
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    constexpr auto datasize = std::tuple_size<DataT>{};

    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    bool keep_going = true;

TernaryNormLoop:
    [[intel::initiation_interval(1)]] while (keep_going) {
        for (int i = 0; i < CONFIG_T::n_in / datasize; i++) {
            auto in_data = data_pipe::read();

        BatchNormPack:
            #pragma unroll
            for (int j = 0; j < datasize; j++) {
                int norm_index;
                if (CONFIG_T::n_filt == -1)
                    norm_index = i * datasize + j;
                else
                    norm_index = j % CONFIG_T::n_filt;

                if (in_data.data[j] > threshold_hi[norm_index])
                    out_data.data[j] = 1;
                else if (in_data.data[j] <= threshold_lo[norm_index])
                    out_data.data[j] = -1;
                else
                    out_data.data[j] = 0;
            }

            out_data.sop = in_data.sop;
            out_data.eop = in_data.eop;
            res_pipe::write(out_data);

            keep_going = !in_data.eop;
        }
    }
}

} // namespace nnet

#endif
