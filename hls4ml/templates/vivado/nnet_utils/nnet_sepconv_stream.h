#ifndef NNET_SEPARABLE_CONV_STREAM_H_
#define NNET_SEPARABLE_CONV_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include "nnet_conv_stream.h"

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void depthwise_product(
    data_T    data[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    res_T     res[CONFIG_T::n_chan],
    typename CONFIG_T::weight_t  weights[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]
) {
    #pragma HLS INLINE

    typename CONFIG_T::accum_t mult[CONFIG_T::kernel_size * CONFIG_T::n_chan];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_chan];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    #pragma HLS ARRAY_PARTITION variable=mult complete

    int multiplier_limit  = ceil(float(CONFIG_T::kernel_size * CONFIG_T::n_chan) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
    CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t, typename CONFIG_T::mult_config::accum_t>::limit(multiplier_limit);

    // Do the matrix-multiply
    Product: for(int ii = 0; ii < CONFIG_T::kernel_size * CONFIG_T::n_chan; ii++) {
        #pragma HLS UNROLL
        mult[ii] = CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t, typename CONFIG_T::mult_config::accum_t>::product(data[ii], weights[ii]);
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_chan; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::kernel_size; ii++) {
        Accum2: for(int jj = 0; jj < CONFIG_T::n_chan; jj++) {
            int index = ii * CONFIG_T::n_chan + jj;
            acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_chan; ires++){
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void depthwise_mult_buffer(
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    res_T& res_pack,
    hls::stream<res_T>& res_stream,
    unsigned & outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]
) {
    #pragma HLS INLINE

    typename data_T::value_type data[CONFIG_T::kernel_size * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data complete
    typename res_T::value_type res[CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=res complete

    InitData: for (int id = 0; id < CONFIG_T::kernel_size * CONFIG_T::n_chan; id++) {
        #pragma HLS UNROLL
        data[id] = data_window[id].read();
    }

    #pragma HLS INLINE region
    if (CONFIG_T::strategy == nnet::latency) {
        depthwise_product<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);
    } else {
        assert("Resource strategy for DepthwiseConv2D is not supported." && false);
    }

    CastLoop: for (unsigned jj = 0; jj < CONFIG_T::n_chan; jj++) {
        #pragma HLS UNROLL
        if (res_T::size / CONFIG_T::n_chan == 1) {
            res_pack[jj] = res[jj];
        } else {
            res_pack[outputs_ready * CONFIG_T::n_chan + jj] = res[jj];
        }
    }

    if (res_T::size / CONFIG_T::n_chan == 1) {
        res_stream.write(res_pack);
    } else {
        if (outputs_ready == (res_T::size / CONFIG_T::n_chan) - 1) {
            res_stream.write(res_pack);
            outputs_ready = 0;
        } else {
            outputs_ready++;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void compute_depthwise_output(
    const data_T& in_elem,
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    hls::stream<res_T> &res,
    res_T &res_pack,
    unsigned &outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_chan],
    ap_uint<CONFIG_T::kernel_size> *pixel_idx
) {
    #pragma HLS INLINE

    MultLoop: for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        CopyDataFilt: for (unsigned f = 0; f < CONFIG_T::kernel_size; f++) {
            #pragma HLS UNROLL
            CopyDataChan: for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                #pragma HLS UNROLL
                if (pixel_idx[p][f]) data_window[f * CONFIG_T::n_chan + c].write(in_elem[p * CONFIG_T::n_chan + c]);
            }
        }
        if (pixel_idx[p][CONFIG_T::kernel_size - 1]) {
            depthwise_mult_buffer<data_T, res_T, CONFIG_T>(data_window, res_pack, res, outputs_ready, weights, biases);
        }
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void pointwise_mult_buffer(
    const data_T &data_pack,
    hls::stream<res_T> &res_stream,
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE

    typename data_T::value_type data[CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res complete

    res_T res_pack;
    #pragma HLS DATA_PACK variable=res_pack

    InitData: for (int id = 0; id < CONFIG_T::n_chan; id++) {
        #pragma HLS UNROLL
        data[id] = data_pack[id];
    }

    #pragma HLS INLINE region
    if (CONFIG_T::strategy == nnet::latency) {
        dense_latency<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(data, res, weights, biases);
    } else {
        dense_resource<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(data, res, weights, biases);
    }

    CastLoop: for (unsigned jj = 0; jj < CONFIG_T::n_filt; jj++) {
        #pragma HLS UNROLL
        res_pack[jj] = res[jj];
    }

    res_stream.write(res_pack);
}

}
#endif
