#ifndef NNET_INDEXED_POOL_H_
#define NNET_INDEXED_POOL_H_

#include "nnet_common.h"

namespace nnet {

// Configuration struct for indexed_maxpool_2d and indexed_avgpool_2d.
//
// Represents a MaxPool2D or AveragePooling2D with
//   pool_size=(1, n_neighbors), padding='valid',
// applied per pixel over the neighbor dimension gathered by
// neighbor_gather_2d.
//
// Input  (flat): [n_pixels * n_neighbors * n_features]
// Output (flat): [n_pixels * n_features]
//
// No learned parameters — pure reduction over the neighbor dimension.
//
// Border slots (index == -1) are zero-masked by neighbor_gather_2d
// before pooling. For MaxPool this means border slots contribute 0
// to the maximum, which is correct when upstream activations are
// non-negative (e.g. after a ReLU). For AvgPool the zeros reduce
// the average proportionally to the number of missing neighbors.

struct indexed_pool_config {
    static const unsigned n_pixels = 1;
    static const unsigned n_neighbors = 7;
    static const unsigned n_features = 1;
    typedef float accum_t;
    // If true, expects the io_stream input to be a single wide packet
    // of n_neighbors*n_features per pixel (matching a
    // pack_neighbors=true NeighborGatherLayer output) instead of
    // n_neighbors separate packets of n_features. Ignored by
    // io_parallel. Must match the pack_neighbors setting of the
    // preceding NeighborGatherLayer, or shapes will mismatch.
    static const bool pack_neighbors = false;
};

// indexed_maxpool_2d
//
// For each pixel p and feature f:
//   output[p * C + f] = max_{k=0}^{K-1} input[p * K * C + k * C + f]

template <class input_T, class output_T, typename CONFIG_T>
void indexed_maxpool_2d(input_T input[CONFIG_T::n_pixels * CONFIG_T::n_neighbors * CONFIG_T::n_features],
                        output_T output[CONFIG_T::n_pixels * CONFIG_T::n_features]) {
    #pragma HLS INLINE

    const unsigned C = CONFIG_T::n_features;
    const unsigned K = CONFIG_T::n_neighbors;
    const unsigned P = CONFIG_T::n_pixels;

Loop_Pixels:
    for (unsigned p = 0; p < P; p++) {
        #pragma HLS UNROLL

    Loop_Features:
        for (unsigned f = 0; f < C; f++) {
            #pragma HLS UNROLL

            input_T max_val = input[p * K * C + 0 * C + f];

        Loop_Neighbors:
            for (unsigned k = 1; k < K; k++) {
                #pragma HLS UNROLL
                input_T val = input[p * K * C + k * C + f];
                if (val > max_val)
                    max_val = val;
            }

            output[p * C + f] = static_cast<output_T>(max_val);
        }
    }
}

// indexed_avgpool_2d
//
// For each pixel p and feature f:
//   output[p * C + f] = (1/K) * sum_{k=0}^{K-1} input[p * K * C + k * C + f]
//
// Border slots zeroed by neighbor_gather_2d reduce the average in
// proportion to the number of missing neighbors, consistent with
// Keras AveragePooling2D over a zero-padded gather tensor.

template <class input_T, class output_T, typename CONFIG_T>
void indexed_avgpool_2d(input_T input[CONFIG_T::n_pixels * CONFIG_T::n_neighbors * CONFIG_T::n_features],
                        output_T output[CONFIG_T::n_pixels * CONFIG_T::n_features]) {
    #pragma HLS INLINE

    typedef typename CONFIG_T::accum_t accum_T;

    const unsigned C = CONFIG_T::n_features;
    const unsigned K = CONFIG_T::n_neighbors;
    const unsigned P = CONFIG_T::n_pixels;

    static const accum_T inv_k = static_cast<accum_T>(1.0f / static_cast<float>(K));

Loop_Pixels:
    for (unsigned p = 0; p < P; p++) {
        #pragma HLS UNROLL

    Loop_Features:
        for (unsigned f = 0; f < C; f++) {
            #pragma HLS UNROLL

            accum_T sum = static_cast<accum_T>(0);

        Loop_Neighbors:
            for (unsigned k = 0; k < K; k++) {
                #pragma HLS UNROLL
                sum += static_cast<accum_T>(input[p * K * C + k * C + f]);
            }

            output[p * C + f] = static_cast<output_T>(sum * inv_k);
        }
    }
}

// ============================================================
// io_stream variant
// ============================================================
//
// indexed_maxpool_2d / indexed_avgpool_2d (io_stream overload)
//
// Same convention as neighbor_gather_2d and indexed_conv_2d: overloaded
// on parameter type (hls::stream<T>& vs. flat array). Consumes
// CONFIG_T::n_neighbors packets of width n_features per pixel (the
// layout emitted by neighbor_gather_2d's io_stream overload -- one
// packet per neighbor) and produces one packet of width n_features per
// pixel, reducing over the K neighbor packets.

// Updates max_val[] with one neighbor's already-extracted C-wide
// feature vector. Shared by both the packed and unpacked io_stream
// paths of indexed_maxpool_2d.
template <class scalar_T, typename CONFIG_T>
void indexed_maxpool_2d_process_neighbor(scalar_T in_arr[CONFIG_T::n_features], scalar_T max_val[CONFIG_T::n_features],
                                         bool is_first_neighbor) {
    #pragma HLS INLINE

Update_Max:
    for (unsigned f = 0; f < CONFIG_T::n_features; f++) {
        #pragma HLS UNROLL
        if (is_first_neighbor) {
            max_val[f] = in_arr[f];
        } else if (in_arr[f] > max_val[f]) {
            max_val[f] = in_arr[f];
        }
    }
}

// Updates sum[] with one neighbor's already-extracted C-wide feature
// vector. Shared by both the packed and unpacked io_stream paths of
// indexed_avgpool_2d.
template <class scalar_T, class accum_T, typename CONFIG_T>
void indexed_avgpool_2d_process_neighbor(scalar_T in_arr[CONFIG_T::n_features], accum_T sum[CONFIG_T::n_features]) {
    #pragma HLS INLINE

Update_Sum:
    for (unsigned f = 0; f < CONFIG_T::n_features; f++) {
        #pragma HLS UNROLL
        sum[f] += static_cast<accum_T>(in_arr[f]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void indexed_maxpool_2d(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream) {
    #pragma HLS INLINE off

    typedef typename data_T::value_type scalar_T;

    const unsigned C = CONFIG_T::n_features;
    const unsigned K = CONFIG_T::n_neighbors;

Loop_Pixels:
    for (unsigned p = 0; p < CONFIG_T::n_pixels; p++) {

        scalar_T max_val[CONFIG_T::n_features];
        #pragma HLS ARRAY_PARTITION variable=max_val complete

        if (CONFIG_T::pack_neighbors) {
            data_T in_pack_wide = data_stream.read();

        Loop_Neighbors_Packed:
            for (unsigned k = 0; k < K; k++) {
                #pragma HLS UNROLL
                scalar_T in_arr[CONFIG_T::n_features];
            #pragma HLS ARRAY_PARTITION variable=in_arr complete
            Copy_In_Packed:
                for (unsigned f = 0; f < C; f++) {
                    #pragma HLS UNROLL
                    in_arr[f] = in_pack_wide[k * C + f];
                }
                indexed_maxpool_2d_process_neighbor<scalar_T, CONFIG_T>(in_arr, max_val, k == 0);
            }
        } else {
        Loop_Neighbors:
            for (unsigned k = 0; k < K; k++) {
                #pragma HLS PIPELINE II=1
                data_T in_pack = data_stream.read();

                scalar_T in_arr[CONFIG_T::n_features];
            #pragma HLS ARRAY_PARTITION variable=in_arr complete
            Copy_In:
                for (unsigned f = 0; f < C; f++) {
                    #pragma HLS UNROLL
                    in_arr[f] = in_pack[f];
                }
                indexed_maxpool_2d_process_neighbor<scalar_T, CONFIG_T>(in_arr, max_val, k == 0);
            }
        }

        res_T out_pack;
    Write_Out:
        for (unsigned f = 0; f < C; f++) {
            #pragma HLS UNROLL
            out_pack[f] = static_cast<typename res_T::value_type>(max_val[f]);
        }
        res_stream.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void indexed_avgpool_2d(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream) {
    #pragma HLS INLINE off

    typedef typename CONFIG_T::accum_t accum_T;
    typedef typename data_T::value_type scalar_T;

    const unsigned C = CONFIG_T::n_features;
    const unsigned K = CONFIG_T::n_neighbors;

    static const accum_T inv_k = static_cast<accum_T>(1.0f / static_cast<float>(K));

Loop_Pixels:
    for (unsigned p = 0; p < CONFIG_T::n_pixels; p++) {

        accum_T sum[CONFIG_T::n_features];
        #pragma HLS ARRAY_PARTITION variable=sum complete

    Init_Sum:
        for (unsigned f = 0; f < C; f++) {
            #pragma HLS UNROLL
            sum[f] = static_cast<accum_T>(0);
        }

        if (CONFIG_T::pack_neighbors) {
            data_T in_pack_wide = data_stream.read();

        Loop_Neighbors_Packed:
            for (unsigned k = 0; k < K; k++) {
                #pragma HLS UNROLL
                scalar_T in_arr[CONFIG_T::n_features];
            #pragma HLS ARRAY_PARTITION variable=in_arr complete
            Copy_In_Packed:
                for (unsigned f = 0; f < C; f++) {
                    #pragma HLS UNROLL
                    in_arr[f] = in_pack_wide[k * C + f];
                }
                indexed_avgpool_2d_process_neighbor<scalar_T, accum_T, CONFIG_T>(in_arr, sum);
            }
        } else {
        Loop_Neighbors:
            for (unsigned k = 0; k < K; k++) {
                #pragma HLS PIPELINE II=1
                data_T in_pack = data_stream.read();

                scalar_T in_arr[CONFIG_T::n_features];
            #pragma HLS ARRAY_PARTITION variable=in_arr complete
            Copy_In:
                for (unsigned f = 0; f < C; f++) {
                    #pragma HLS UNROLL
                    in_arr[f] = in_pack[f];
                }
                indexed_avgpool_2d_process_neighbor<scalar_T, accum_T, CONFIG_T>(in_arr, sum);
            }
        }

        res_T out_pack;
    Write_Out:
        for (unsigned f = 0; f < C; f++) {
            #pragma HLS UNROLL
            out_pack[f] = static_cast<typename res_T::value_type>(sum[f] * inv_k);
        }
        res_stream.write(out_pack);
    }
}

} // namespace nnet

#endif // NNET_INDEXED_POOL_H_
