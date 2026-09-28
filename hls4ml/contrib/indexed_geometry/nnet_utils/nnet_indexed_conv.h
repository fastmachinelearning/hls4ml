#ifndef NNET_INDEXED_CONV_H_
#define NNET_INDEXED_CONV_H_

#include "nnet_common.h"
#include "nnet_dense_resource.h"

namespace nnet {

// Configuration struct for indexed_conv_2d.
//
// Represents a Conv2D(filters, kernel_size=(1, n_neighbors),
// padding='valid', activation='relu') applied independently to
// each pixel over the neighbor dimension gathered by
// neighbor_gather_2d.
//
// Input  (flat): [n_pixels * n_neighbors * n_features_in]
// Output (flat): [n_pixels * n_features_out]
//
// Weight layout (flat, row-major): [n_features_out][n_neighbors][n_features_in]
// This is derived from the Keras Conv2D kernel of shape
//   (1, n_neighbors, n_features_in, n_features_out)
// by extracting slice [0], transposing to
//   (n_features_out, n_neighbors, n_features_in),
// and flattening. The Python parser performs this transformation
// (see indexed_conv.py).
//
// reuse_factor controls how many times each multiplier is reused.
// Total MACs per pixel = n_neighbors * n_features_in * n_features_out.
// With reuse_factor=R, the number of multipliers is reduced by R,
// and the initiation interval of the inner MAC loop becomes R.
// Set reuse_factor=1 for fully parallel (latency-optimal) operation.

struct indexed_conv_config {
    static const unsigned n_pixels = 1;
    static const unsigned n_neighbors = 7;
    static const unsigned n_features_in = 1;
    static const unsigned n_features_out = 1;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = (n_neighbors * n_features_in * n_features_out) / reuse_factor;
    // If true, expects the io_stream input to be a single wide packet
    // of n_neighbors*n_features_in per pixel (matching a
    // pack_neighbors=true NeighborGatherLayer output) instead of
    // n_neighbors separate packets of n_features_in. Ignored by
    // io_parallel. Must match the pack_neighbors setting of the
    // preceding NeighborGatherLayer, or shapes will mismatch.
    static const bool pack_neighbors = false;
};

// indexed_conv_2d
//
// For each pixel p and output feature f_out:
//
//   output[p * C_out + f_out] = ReLU(
//       bias[f_out]
//       + sum_{k=0}^{K-1} sum_{f_in=0}^{C_in-1}
//           input  [p * K * C_in  + k * C_in + f_in]
//         * weights[f_out * K * C_in + k * C_in + f_in]
//   )
//
// input_T, output_T, weight_T, and bias_T are independent template
// parameters so that hls4ml can assign per-layer fixed-point precisions.
// bias_T is also used for the accumulator to avoid overflow.
//
// The outer loop over pixels is pipelined (II=1).
// The inner MAC loop is pipelined with II=reuse_factor, with weights
// partitioned cyclically to match the reuse pattern.

template <class input_T, class output_T, class weight_T, class bias_T, typename CONFIG_T>
void indexed_conv_2d(input_T input[CONFIG_T::n_pixels * CONFIG_T::n_neighbors * CONFIG_T::n_features_in],
                     output_T output[CONFIG_T::n_pixels * CONFIG_T::n_features_out],
                     weight_T weights[CONFIG_T::n_features_out * CONFIG_T::n_neighbors * CONFIG_T::n_features_in],
                     bias_T biases[CONFIG_T::n_features_out]) {
    #pragma HLS INLINE

    const unsigned C_in = CONFIG_T::n_features_in;
    const unsigned C_out = CONFIG_T::n_features_out;
    const unsigned K = CONFIG_T::n_neighbors;
    const unsigned P = CONFIG_T::n_pixels;
    const unsigned RF = CONFIG_T::reuse_factor;
    const unsigned N_MACS = K * C_in * C_out;

    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=RF

Loop_Pixels:
    for (unsigned p = 0; p < P; p++) {
        #pragma HLS PIPELINE II=1

        bias_T acc[CONFIG_T::n_features_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    // Initialize accumulators with biases
    Init_Cout:
        for (unsigned f_out = 0; f_out < C_out; f_out++) {
            #pragma HLS UNROLL
            acc[f_out] = static_cast<bias_T>(biases[f_out]);
        }

    // MAC loop — pipelined with II=reuse_factor
    Loop_MAC:
        for (unsigned i = 0; i < N_MACS; i++) {
            #pragma HLS PIPELINE II=RF
            const unsigned f_out = i / (K * C_in);
            const unsigned k = (i / C_in) % K;
            const unsigned f_in = i % C_in;
            const unsigned in_idx = p * (K * C_in) + k * C_in + f_in;
            const unsigned w_idx = f_out * (K * C_in) + k * C_in + f_in;
            acc[f_out] += static_cast<bias_T>(input[in_idx]) * static_cast<bias_T>(weights[w_idx]);
        }

    // ReLU activation
    Write_Out:
        for (unsigned f_out = 0; f_out < C_out; f_out++) {
            #pragma HLS UNROLL
            const unsigned out_idx = p * C_out + f_out;
            output[out_idx] =
                (acc[f_out] > static_cast<bias_T>(0)) ? static_cast<output_T>(acc[f_out]) : static_cast<output_T>(0);
        }
    }
}

// ============================================================
// io_stream variant
// ============================================================
//
// indexed_conv_2d (io_stream overload)

// Processes one neighbor's already-extracted C_in-wide feature vector:
// runs it through dense_resource (a standard hls4ml n_in x n_out Dense
// kernel, weight sub-block for this neighbor) and accumulates the
// result into acc[]. Shared by both the packed and unpacked io_stream
// paths of indexed_conv_2d, so the MAC/accumulation logic exists in
// exactly one place.

template <class scalar_T, class accum_T, typename CONFIG_T>
void indexed_conv_2d_process_neighbor(
    scalar_T in_arr[CONFIG_T::n_features_in], accum_T acc[CONFIG_T::n_features_out],
    typename CONFIG_T::dense_config::weight_t
        weights[CONFIG_T::n_features_out * CONFIG_T::n_neighbors * CONFIG_T::n_features_in],
    unsigned k) {
    #pragma HLS INLINE

    typedef typename CONFIG_T::dense_config dense_config_t;
    const unsigned C_in = CONFIG_T::n_features_in;
    const unsigned C_out = CONFIG_T::n_features_out;

    accum_T partial[CONFIG_T::n_features_out];
    #pragma HLS ARRAY_PARTITION variable=partial complete
    accum_T zero_bias[CONFIG_T::n_features_out];
#pragma HLS ARRAY_PARTITION variable=zero_bias complete
Zero_Bias:
    for (unsigned f_out = 0; f_out < C_out; f_out++) {
        #pragma HLS UNROLL
        zero_bias[f_out] = static_cast<accum_T>(0);
    }

    nnet::dense_resource<scalar_T, accum_T, dense_config_t>(in_arr, partial, &weights[k * C_in * C_out], zero_bias);

Accum_Neighbor:
    for (unsigned f_out = 0; f_out < C_out; f_out++) {
        #pragma HLS UNROLL
        acc[f_out] += partial[f_out];
    }
}

//
// io_stream counterpart of indexed_conv_2d, overloaded on parameter
// type (hls::stream<T>& vs. flat array), following the same convention
// established in nnet_neighbor_gather.h and hls4ml's own Conv1D.
//
// Consumes CONFIG_T::n_neighbors packets of width n_features_in per
// pixel (the layout emitted by neighbor_gather_2d's io_stream overload:
// one packet per neighbor, not a single wide packet) and produces one
// packet of width n_features_out per pixel.
//
// Weights remain a flat array (not a stream), same as io_parallel and
// consistent with hls4ml's own convolution/dense kernels.
//
// reuse_factor is not applied in this version -- full parallelism
// within each per-neighbor MAC step (C_out * C_in multipliers active
// per cycle, reused automatically across the K neighbor-cycles by the
// pipeline itself). Left as a possible follow-up.

template <class data_T, class res_T, class weight_T, class bias_T, typename CONFIG_T>
void indexed_conv_2d(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
                     weight_T weights[CONFIG_T::n_features_out * CONFIG_T::n_neighbors * CONFIG_T::n_features_in],
                     bias_T biases[CONFIG_T::n_features_out]) {
    #pragma HLS INLINE off

    typedef typename CONFIG_T::dense_config dense_config_t;
    typedef typename dense_config_t::accum_t accum_T;

    const unsigned C_in = CONFIG_T::n_features_in;
    const unsigned C_out = CONFIG_T::n_features_out;
    const unsigned K = CONFIG_T::n_neighbors;

    #pragma HLS ARRAY_PARTITION variable=biases complete

    if (CONFIG_T::pack_neighbors) {
        // Local, fully-partitioned copy of the weights, made once
        // (weights don't change per pixel). Kept separate from the
        // function's 'weights' parameter so this partition never
        // interferes with the non-packed path's use of dense_resource
        // (which needs a contiguous pointer-offset view of 'weights').
        const unsigned N_W = C_out * K * C_in;
        weight_T weights_local[C_out * K * C_in];
    #pragma HLS ARRAY_PARTITION variable=weights_local complete dim=0
    Copy_Weights:
        for (unsigned i = 0; i < N_W; i++) {
            #pragma HLS UNROLL
            weights_local[i] = weights[i];
        }

    Loop_Pixels_Packed:
        for (unsigned p = 0; p < CONFIG_T::n_pixels; p++) {
            #pragma HLS PIPELINE II=1
            accum_T acc[CONFIG_T::n_features_out];
        #pragma HLS ARRAY_PARTITION variable=acc complete
        Init_Acc_Packed:
            for (unsigned f_out = 0; f_out < C_out; f_out++) {
                #pragma HLS UNROLL
                acc[f_out] = static_cast<accum_T>(0);
            }

            data_T in_pack_wide = data_stream.read();

            const unsigned N_MACS = K * C_in * C_out;
        Loop_MAC_Packed:
            for (unsigned i = 0; i < N_MACS; i++) {
                #pragma HLS PIPELINE II=1
                const unsigned f_out = i / (K * C_in);
                const unsigned k = (i / C_in) % K;
                const unsigned f_in = i % C_in;
                // weights_local layout: (K, C_out, C_in) -- matches
                // weight_data_stream (see indexed_conv.py)
                const unsigned w_idx = k * (C_in * C_out) + f_out * C_in + f_in;
                acc[f_out] +=
                    static_cast<accum_T>(in_pack_wide[k * C_in + f_in]) * static_cast<accum_T>(weights_local[w_idx]);
            }

            res_T out_pack;
        Write_Out_Packed:
            for (unsigned f_out = 0; f_out < C_out; f_out++) {
                #pragma HLS UNROLL
                accum_T val = acc[f_out] + static_cast<accum_T>(biases[f_out]);
                out_pack[f_out] = (val > static_cast<accum_T>(0)) ? static_cast<typename res_T::value_type>(val)
                                                                  : static_cast<typename res_T::value_type>(0);
            }
            res_stream.write(out_pack);
        }

    } else {

    Loop_Pixels:
        for (unsigned p = 0; p < CONFIG_T::n_pixels; p++) {

            accum_T acc[CONFIG_T::n_features_out];
        #pragma HLS ARRAY_PARTITION variable=acc complete
        Init_Acc:
            for (unsigned f_out = 0; f_out < C_out; f_out++) {
                #pragma HLS UNROLL
                acc[f_out] = static_cast<accum_T>(0);
            }

        Loop_Neighbors:
            for (unsigned k = 0; k < K; k++) {
                data_T in_pack = data_stream.read();

                typename data_T::value_type in_arr[CONFIG_T::n_features_in];
            #pragma HLS ARRAY_PARTITION variable=in_arr complete
            Copy_In:
                for (unsigned f_in = 0; f_in < C_in; f_in++) {
                    #pragma HLS UNROLL
                    in_arr[f_in] = in_pack[f_in];
                }
                indexed_conv_2d_process_neighbor<typename data_T::value_type, accum_T, CONFIG_T>(in_arr, acc, weights, k);
            }

            res_T out_pack;
        Write_Out:
            for (unsigned f_out = 0; f_out < C_out; f_out++) {
                #pragma HLS UNROLL
                accum_T val = acc[f_out] + static_cast<accum_T>(biases[f_out]);
                out_pack[f_out] = (val > static_cast<accum_T>(0)) ? static_cast<typename res_T::value_type>(val)
                                                                  : static_cast<typename res_T::value_type>(0);
            }
            res_stream.write(out_pack);
        }
    }
}

} // namespace nnet

#endif // NNET_INDEXED_CONV_H_
