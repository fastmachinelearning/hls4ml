#ifndef NNET_CONV_TRANSPOSE_H_
#define NNET_CONV_TRANSPOSE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_function_stubs.h"
#include "nnet_mult.h"
#include <cstdlib>

namespace nnet {

// A transposed convolution is computed in blocks of stride_height x stride_width output positions, one block per input
// position. Group g of a block is its position (g / stride_width, g % stride_width). Every kernel element adds to
// exactly one group, so the inputs that all kernel elements of a block read, ordered by group, form a vector of n_buf
// values, and group_of[j] is the group that value j adds to. The weights are the matching n_buf x n_filt matrix,
// transposed for the Resource strategy. Blocks are split into n_partitions parts of n_pixels blocks each; fill_buffer
// collects the inputs of a part and place_block writes its results to their output positions, leaving out the cropped
// ones. Both are generated for each layer with fixed indices. A 1D layer has height 1.

struct conv_transpose_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer sizes
    static const unsigned in_height = 1;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 1;
    static const unsigned out_height = 1;
    static const unsigned out_width = 20;
    static const unsigned n_filt = 1;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 2;
    static const unsigned crop_top = 0;
    static const unsigned crop_left = 0;
    static const unsigned first_block_height = 0;
    static const unsigned first_block_width = 0;
    static const unsigned n_blocks_height = 1;
    static const unsigned n_blocks_width = 10;
    static const unsigned n_buf = 1;

    // Resource reuse info
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = latency;
    static const unsigned n_partitions = 10;
    static const unsigned n_pixels = 1;
    static const unsigned multiplier_limit = 1;
};

template <class data_T, class res_T, typename CONFIG_T>
void conv_transpose_latency_cl(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                               res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    constexpr unsigned mult_n_in = CONFIG_T::n_buf;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;
    constexpr unsigned n_acc = CONFIG_T::stride_height * CONFIG_T::stride_width * CONFIG_T::n_filt;

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    // Limit multipliers to control parallelization; all n_pixels blocks of a part are computed at once
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

PartitionLoop:
    for (unsigned i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind

        // Declared in the loop: each part is computed from scratch, nothing is carried to the next one
        data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
        #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=0

        typename CONFIG_T::accum_t mult[mult_n_in * mult_n_out];
        #pragma HLS ARRAY_PARTITION variable=mult complete

        typename CONFIG_T::accum_t acc[CONFIG_T::n_pixels][n_acc];
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

    PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL

        Product1:
            for (unsigned i_in = 0; i_in < mult_n_in; i_in++) {
                #pragma HLS UNROLL
            Product2:
                for (unsigned i_out = 0; i_out < mult_n_out; i_out++) {
                    #pragma HLS UNROLL
                    mult[i_in * mult_n_out + i_out] =
                        CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                            data_buf[i_pxl][i_in], weights[i_in * mult_n_out + i_out]);
                }
            }

        ResetAccum:
            for (unsigned i_acc = 0; i_acc < n_acc; i_acc++) {
                #pragma HLS UNROLL
                acc[i_pxl][i_acc] = (typename CONFIG_T::accum_t)biases[i_acc % mult_n_out];
            }

        // Each product adds to the group of its input
        Accum1:
            for (unsigned i_in = 0; i_in < mult_n_in; i_in++) {
                #pragma HLS UNROLL
            Accum2:
                for (unsigned i_out = 0; i_out < mult_n_out; i_out++) {
                    #pragma HLS UNROLL
                    acc[i_pxl][CONFIG_T::group_of[i_in] * mult_n_out + i_out] += mult[i_in * mult_n_out + i_out];
                }
            }
        }

        CONFIG_T::template place_block<data_T, res_T, CONFIG_T>::place(acc, res, i_part);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_transpose_resource_cl(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                                res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
                                typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                                typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    constexpr unsigned mult_n_in = CONFIG_T::n_buf;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;
    constexpr unsigned n_groups = CONFIG_T::stride_height * CONFIG_T::stride_width;
    constexpr unsigned n_acc = n_groups * mult_n_out;
    constexpr unsigned block_factor = DIV_ROUNDUP(mult_n_in * mult_n_out, CONFIG_T::reuse_factor);
    constexpr unsigned n_chunks = mult_n_in / CONFIG_T::reuse_factor;

    assert((mult_n_in % CONFIG_T::reuse_factor == 0) && "The reuse factor must divide N_BUF");

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=0

    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    // Multiplier i_blk computes output i_blk / n_chunks from the inputs of chunk i_blk % n_chunks, one input per
    // cycle. It keeps its own sum for each group these inputs add to, so a cycle adds one product to each sum.
    typename CONFIG_T::accum_t partial[CONFIG_T::n_pixels][block_factor][n_groups];
    #pragma HLS ARRAY_PARTITION variable=partial complete dim=0

    typename CONFIG_T::accum_t acc[CONFIG_T::n_pixels][n_acc];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

// All code is in the inner loop, so the two loops form one pipeline of n_partitions * reuse_factor cycles. A part
// collects its inputs in its first cycle and adds up and writes its results in its last.
PartitionLoop:
    for (unsigned i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
    ReuseLoop:
        for (unsigned i_rf = 0; i_rf < CONFIG_T::reuse_factor; i_rf++) {
            #pragma HLS PIPELINE II=1 rewind

            if (i_rf == 0) {
                CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

            PixelInitLoop:
                for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
                    #pragma HLS UNROLL
                InitBlockLoop:
                    for (unsigned i_blk = 0; i_blk < block_factor; i_blk++) {
                        #pragma HLS UNROLL
                    InitGroupLoop:
                        for (unsigned i_g = 0; i_g < n_groups; i_g++) {
                            #pragma HLS UNROLL
                            partial[i_pxl][i_blk][i_g] = 0;
                        }
                    }
                }
            }

        MultLoop:
            for (unsigned i_blk = 0; i_blk < block_factor; i_blk++) {
                #pragma HLS UNROLL
                const unsigned i_chunk = i_blk % n_chunks;
                const unsigned i_in = i_chunk * CONFIG_T::reuse_factor + i_rf;
                // Most chunks lie within one group, so their group is known without looking at i_rf
                const unsigned group = CONFIG_T::chunk_first_group[i_chunk] == CONFIG_T::chunk_last_group[i_chunk]
                                           ? CONFIG_T::chunk_first_group[i_chunk]
                                           : CONFIG_T::group_of[i_in];

            PixelMultLoop:
                for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
                    #pragma HLS UNROLL
                    partial[i_pxl][i_blk][group] += static_cast<typename CONFIG_T::accum_t>(
                        CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                            data_buf[i_pxl][i_in], weights[i_blk * CONFIG_T::reuse_factor + i_rf]));
                }
            }

            if (i_rf == CONFIG_T::reuse_factor - 1) {
            PixelSumLoop:
                for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
                    #pragma HLS UNROLL
                ResetAccum:
                    for (unsigned i_acc = 0; i_acc < n_acc; i_acc++) {
                        #pragma HLS UNROLL
                        acc[i_pxl][i_acc] = (typename CONFIG_T::accum_t)biases[i_acc % mult_n_out];
                    }
                SumBlockLoop:
                    for (unsigned i_blk = 0; i_blk < block_factor; i_blk++) {
                        #pragma HLS UNROLL
                        const unsigned i_chunk = i_blk % n_chunks;
                    SumGroupLoop:
                        for (unsigned i_g = 0; i_g < n_groups; i_g++) {
                            #pragma HLS UNROLL
                            if (i_g >= CONFIG_T::chunk_first_group[i_chunk] && i_g <= CONFIG_T::chunk_last_group[i_chunk]) {
                                acc[i_pxl][i_g * mult_n_out + i_blk / n_chunks] += partial[i_pxl][i_blk][i_g];
                            }
                        }
                    }
                }

                CONFIG_T::template place_block<data_T, res_T, CONFIG_T>::place(acc, res, i_part);
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T> class ConvTransposeLatency {
  public:
    static void conv(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                     res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
                     typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                     typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        #pragma HLS INLINE
        conv_transpose_latency_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
};

template <class data_T, class res_T, typename CONFIG_T> class ConvTransposeResource {
  public:
    static void conv(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                     res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
                     typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                     typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        #pragma HLS INLINE
        conv_transpose_resource_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
};

template <class data_T, class res_T, typename CONFIG_T>
void conv_transpose_cl(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                       res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
                       typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                       typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    #pragma HLS INLINE region
    CONFIG_T::template conv_kernel<data_T, res_T, CONFIG_T>::conv(data, res, weights, biases);
}

} // namespace nnet

#endif
