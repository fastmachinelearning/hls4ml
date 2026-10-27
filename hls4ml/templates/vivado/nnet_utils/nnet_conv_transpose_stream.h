#ifndef NNET_CONV_TRANSPOSE_STREAM_H_
#define NNET_CONV_TRANSPOSE_STREAM_H_

#include "ap_shift_reg.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_conv_transpose.h"
#include "nnet_types.h"

namespace nnet {

// The stream version computes the same blocks as the array version, one block at a time, in two processes. The first
// walks a grid of grid_height x grid_width positions in row order: the input, followed by zeros where blocks read past
// its end. It keeps the last rows in a line buffer, and the position at which a block's last input arrives computes
// that block; inputs before the start of the input row or column are zero. The second receives the blocks in order and
// writes the output pixels in row order, keeping one row of blocks for the output rows after the first one, and leaves
// out the cropped positions.

template <class data_T, typename CONFIG_T>
void read_conv_transpose_grid(
    hls::stream<data_T> &data, unsigned i_gh, unsigned i_gw,
    ap_shift_reg<typename data_T::value_type, CONFIG_T::grid_width> line_buffer[MAX(CONFIG_T::window_height - 1, 1)]
                                                                               [CONFIG_T::n_chan],
    typename data_T::value_type window[CONFIG_T::window_height * CONFIG_T::window_width * CONFIG_T::n_chan],
    typename data_T::value_type data_buf[CONFIG_T::n_buf]) {
    #pragma HLS INLINE

    constexpr unsigned win_h = CONFIG_T::window_height;
    constexpr unsigned win_w = CONFIG_T::window_width;
    constexpr unsigned n_chan = CONFIG_T::n_chan;

    data_T in_elem;
    if (i_gh < CONFIG_T::in_height && i_gw < CONFIG_T::in_width) {
        in_elem = data.read();
    } else {
    ZeroLoop:
        for (unsigned i_ic = 0; i_ic < n_chan; i_ic++) {
            #pragma HLS UNROLL
            in_elem[i_ic] = 0;
        }
    }

    // The new column of the window: the input, and from the line buffer the rows above it
    typename data_T::value_type column[win_h][n_chan];
    #pragma HLS ARRAY_PARTITION variable=column complete dim=0
ColumnLoop:
    for (unsigned i_ic = 0; i_ic < n_chan; i_ic++) {
        #pragma HLS UNROLL
        column[win_h - 1][i_ic] = in_elem[i_ic];
    LineLoop:
        for (unsigned i_row = 1; i_row < win_h; i_row++) {
            #pragma HLS UNROLL
            column[win_h - 1 - i_row][i_ic] = line_buffer[i_row - 1][i_ic].shift(column[win_h - i_row][i_ic]);
        }
    }

    // Above the first input row, the line buffer holds the end of the previous image, and before the first input
    // column, the window holds the end of the previous row: the window gets zeros there instead
WindowRowLoop:
    for (unsigned i_row = 0; i_row < win_h; i_row++) {
        #pragma HLS UNROLL
        const bool above = i_gh < win_h - 1 - i_row;
    WindowColLoop:
        for (unsigned i_col = 0; i_col < win_w; i_col++) {
            #pragma HLS UNROLL
        WindowChanLoop:
            for (unsigned i_ic = 0; i_ic < n_chan; i_ic++) {
                #pragma HLS UNROLL
                const unsigned i_win = (i_row * win_w + i_col) * n_chan + i_ic;
                if (i_col == win_w - 1) {
                    window[i_win] = above ? (typename data_T::value_type)0 : column[i_row][i_ic];
                } else {
                    window[i_win] = i_gw == 0 ? (typename data_T::value_type)0 : window[i_win + n_chan];
                }
            }
        }
    }

// Every kernel element reads the window position of its input
FillLoop:
    for (unsigned i_slot = 0; i_slot < CONFIG_T::n_buf / n_chan; i_slot++) {
        #pragma HLS UNROLL
        const unsigned i_win =
            (win_h - 1 - CONFIG_T::slot_shift_height[i_slot]) * win_w + win_w - 1 - CONFIG_T::slot_shift_width[i_slot];
    FillChanLoop:
        for (unsigned i_ic = 0; i_ic < n_chan; i_ic++) {
            #pragma HLS UNROLL
            data_buf[i_slot * n_chan + i_ic] = window[i_win * n_chan + i_ic];
        }
    }
}

// Whether each kernel element reads inside the input at a grid position, which binary inputs need (see
// conv_transpose_binary).
template <typename CONFIG_T>
void conv_transpose_grid_valid(unsigned i_gh, unsigned i_gw, bool valid[CONFIG_T::n_buf / CONFIG_T::n_chan]) {
    #pragma HLS INLINE
ValidLoop:
    for (unsigned i_slot = 0; i_slot < CONFIG_T::n_buf / CONFIG_T::n_chan; i_slot++) {
        #pragma HLS UNROLL
        // The input this kernel element reads is at (i_gh - shift_height, i_gw - shift_width)
        const unsigned shift_h = CONFIG_T::slot_shift_height[i_slot];
        const unsigned shift_w = CONFIG_T::slot_shift_width[i_slot];
        valid[i_slot] = i_gh >= shift_h && i_gh - shift_h < CONFIG_T::in_height && i_gw >= shift_w &&
                        i_gw - shift_w < CONFIG_T::in_width;
    }
}

template <typename CONFIG_T> bool is_conv_transpose_block(unsigned i_gh, unsigned i_gw) {
    #pragma HLS INLINE
    return i_gh >= CONFIG_T::first_block_height && i_gh < CONFIG_T::first_block_height + CONFIG_T::n_blocks_height &&
           i_gw >= CONFIG_T::first_block_width && i_gw < CONFIG_T::first_block_width + CONFIG_T::n_blocks_width;
}

template <class data_T, class block_T, typename CONFIG_T>
void conv_transpose_blocks_latency(hls::stream<data_T> &data, hls::stream<block_T> &blocks,
                                   typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                                   typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    constexpr unsigned mult_n_in = CONFIG_T::n_buf;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;
    constexpr unsigned n_acc = CONFIG_T::stride_height * CONFIG_T::stride_width * CONFIG_T::n_filt;

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    // Limit multipliers to control parallelization
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::mult_config::multiplier_limit

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::grid_width> line_buffer[MAX(CONFIG_T::window_height - 1, 1)]
                                                                                      [CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2

    static typename data_T::value_type window[CONFIG_T::window_height * CONFIG_T::window_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=window complete

GridHeight:
    for (unsigned i_gh = 0; i_gh < CONFIG_T::grid_height; i_gh++) {
    GridWidth:
        for (unsigned i_gw = 0; i_gw < CONFIG_T::grid_width; i_gw++) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind

            typename data_T::value_type data_buf[mult_n_in];
            #pragma HLS ARRAY_PARTITION variable=data_buf complete
            bool valid[mult_n_in / CONFIG_T::n_chan];
            #pragma HLS ARRAY_PARTITION variable=valid complete

            typename CONFIG_T::accum_t mult[mult_n_in * mult_n_out];
            #pragma HLS ARRAY_PARTITION variable=mult complete

            typename CONFIG_T::accum_t acc[n_acc];
            #pragma HLS ARRAY_PARTITION variable=acc complete

            read_conv_transpose_grid<data_T, CONFIG_T>(data, i_gh, i_gw, line_buffer, window, data_buf);
            conv_transpose_grid_valid<CONFIG_T>(i_gh, i_gw, valid);

            if (is_conv_transpose_block<CONFIG_T>(i_gh, i_gw)) {
            Product1:
                for (unsigned i_in = 0; i_in < mult_n_in; i_in++) {
                    #pragma HLS UNROLL
                Product2:
                    for (unsigned i_out = 0; i_out < mult_n_out; i_out++) {
                        #pragma HLS UNROLL
                        mult[i_in * mult_n_out + i_out] = conv_transpose_product<typename data_T::value_type, CONFIG_T>(
                            data_buf[i_in], weights[i_in * mult_n_out + i_out], valid[i_in / CONFIG_T::n_chan]);
                    }
                }

            ResetAccum:
                for (unsigned i_acc = 0; i_acc < n_acc; i_acc++) {
                    #pragma HLS UNROLL
                    acc[i_acc] = conv_transpose_start<typename data_T::value_type, CONFIG_T>(biases[i_acc % mult_n_out]);
                }

            // Each product adds to the group of its input
            Accum1:
                for (unsigned i_in = 0; i_in < mult_n_in; i_in++) {
                    #pragma HLS UNROLL
                Accum2:
                    for (unsigned i_out = 0; i_out < mult_n_out; i_out++) {
                        #pragma HLS UNROLL
                        acc[CONFIG_T::group_of[i_in] * mult_n_out + i_out] += mult[i_in * mult_n_out + i_out];
                    }
                }
                conv_transpose_count<typename data_T::value_type, CONFIG_T>(acc, valid, biases);

                block_T block;
            CastLoop:
                for (unsigned i_acc = 0; i_acc < n_acc; i_acc++) {
                    #pragma HLS UNROLL
                    block[i_acc] = (typename block_T::value_type)acc[i_acc];
                }
                blocks.write(block);
            }
        }
    }
}

template <class data_T, class block_T, typename CONFIG_T>
void conv_transpose_blocks_resource(hls::stream<data_T> &data, hls::stream<block_T> &blocks,
                                    typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                                    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    constexpr unsigned mult_n_in = CONFIG_T::n_buf;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;
    constexpr unsigned n_groups = CONFIG_T::stride_height * CONFIG_T::stride_width;
    constexpr unsigned n_acc = n_groups * mult_n_out;
    constexpr unsigned block_factor = DIV_ROUNDUP(mult_n_in * mult_n_out, CONFIG_T::reuse_factor);
    constexpr unsigned n_chunks = mult_n_in / CONFIG_T::reuse_factor;

    assert((mult_n_in % CONFIG_T::reuse_factor == 0) && "The reuse factor must divide N_BUF");

    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::grid_width> line_buffer[MAX(CONFIG_T::window_height - 1, 1)]
                                                                                      [CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2

    static typename data_T::value_type window[CONFIG_T::window_height * CONFIG_T::window_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=window complete

    typename data_T::value_type data_buf[mult_n_in];
    #pragma HLS ARRAY_PARTITION variable=data_buf complete

    // As in the array version, multiplier i_blk keeps its own sum for each group its inputs add to
    typename CONFIG_T::accum_t partial[block_factor][n_groups];
    #pragma HLS ARRAY_PARTITION variable=partial complete dim=0

    typename CONFIG_T::accum_t acc[n_acc];
    #pragma HLS ARRAY_PARTITION variable=acc complete

// All code is in the inner loop, so the loops form one pipeline of reuse_factor cycles per position, as the Latency
// version with its interval of reuse_factor. A position reads its input in its first cycle, and a block position
// writes its block in its last.
GridHeight:
    for (unsigned i_gh = 0; i_gh < CONFIG_T::grid_height; i_gh++) {
    GridWidth:
        for (unsigned i_gw = 0; i_gw < CONFIG_T::grid_width; i_gw++) {
        ReuseLoop:
            for (unsigned i_rf = 0; i_rf < CONFIG_T::reuse_factor; i_rf++) {
                #pragma HLS PIPELINE II=1 rewind

                const bool is_block = is_conv_transpose_block<CONFIG_T>(i_gh, i_gw);
                bool valid[mult_n_in / CONFIG_T::n_chan];
                #pragma HLS ARRAY_PARTITION variable=valid complete
                conv_transpose_grid_valid<CONFIG_T>(i_gh, i_gw, valid);

                if (i_rf == 0) {
                    read_conv_transpose_grid<data_T, CONFIG_T>(data, i_gh, i_gw, line_buffer, window, data_buf);

                InitBlockLoop:
                    for (unsigned i_blk = 0; i_blk < block_factor; i_blk++) {
                        #pragma HLS UNROLL
                    InitGroupLoop:
                        for (unsigned i_g = 0; i_g < n_groups; i_g++) {
                            #pragma HLS UNROLL
                            partial[i_blk][i_g] = 0;
                        }
                    }
                }

                if (is_block) {
                MultLoop:
                    for (unsigned i_blk = 0; i_blk < block_factor; i_blk++) {
                        #pragma HLS UNROLL
                        const unsigned i_chunk = i_blk % n_chunks;
                        const unsigned i_in = i_chunk * CONFIG_T::reuse_factor + i_rf;
                        const unsigned group = CONFIG_T::chunk_first_group[i_chunk] == CONFIG_T::chunk_last_group[i_chunk]
                                                   ? CONFIG_T::chunk_first_group[i_chunk]
                                                   : CONFIG_T::group_of[i_in];
                        partial[i_blk][group] += conv_transpose_product<typename data_T::value_type, CONFIG_T>(
                            data_buf[i_in], weights[i_blk * CONFIG_T::reuse_factor + i_rf], valid[i_in / CONFIG_T::n_chan]);
                    }
                }

                if (is_block && i_rf == CONFIG_T::reuse_factor - 1) {
                ResetAccum:
                    for (unsigned i_acc = 0; i_acc < n_acc; i_acc++) {
                        #pragma HLS UNROLL
                        acc[i_acc] = conv_transpose_start<typename data_T::value_type, CONFIG_T>(biases[i_acc % mult_n_out]);
                    }
                SumBlockLoop:
                    for (unsigned i_blk = 0; i_blk < block_factor; i_blk++) {
                        #pragma HLS UNROLL
                        const unsigned i_chunk = i_blk % n_chunks;
                    SumGroupLoop:
                        for (unsigned i_g = 0; i_g < n_groups; i_g++) {
                            #pragma HLS UNROLL
                            if (i_g >= CONFIG_T::chunk_first_group[i_chunk] && i_g <= CONFIG_T::chunk_last_group[i_chunk]) {
                                acc[i_g * mult_n_out + i_blk / n_chunks] += partial[i_blk][i_g];
                            }
                        }
                    }
                    conv_transpose_count<typename data_T::value_type, CONFIG_T>(acc, valid, biases);

                    block_T block;
                CastLoop:
                    for (unsigned i_acc = 0; i_acc < n_acc; i_acc++) {
                        #pragma HLS UNROLL
                        block[i_acc] = (typename block_T::value_type)acc[i_acc];
                    }
                    blocks.write(block);
                }
            }
        }
    }
}

template <class block_T, class res_T, typename CONFIG_T>
void place_conv_transpose_blocks(hls::stream<block_T> &blocks, hls::stream<res_T> &res) {
    constexpr unsigned n_filt = CONFIG_T::n_filt;
    constexpr unsigned row_size = CONFIG_T::stride_width * CONFIG_T::n_filt;
    constexpr unsigned n_later = (CONFIG_T::stride_height - 1) * row_size;

    // The output rows after the first one of a block row read the rest of each block from here
    typename res_T::value_type row_blocks[CONFIG_T::n_blocks_width][MAX(n_later, 1)];
    #pragma HLS ARRAY_PARTITION variable=row_blocks complete dim=2

    typename res_T::value_type block[block_T::size];
    #pragma HLS ARRAY_PARTITION variable=block complete

    // Group (i_grh, i_grw) of block i_bw is the output position being written. The counters are back at their first
    // values after the last pixel, so the loops can start again without a pause.
    static unsigned i_grh = CONFIG_T::first_group_height;
    static unsigned i_grw = CONFIG_T::first_group_width;
    static unsigned i_bw = 0;

OutputHeight:
    for (unsigned i_oh = 0; i_oh < CONFIG_T::out_height; i_oh++) {
    OutputWidth:
        for (unsigned i_ow = 0; i_ow < CONFIG_T::out_width; i_ow++) {
            #pragma HLS PIPELINE II=1 rewind

            if (i_ow == 0 || i_grw == 0) {
                if (i_oh == 0 || i_grh == 0) {
                    block_T new_block = blocks.read();
                ReadBlockLoop:
                    for (unsigned i = 0; i < block_T::size; i++) {
                        #pragma HLS UNROLL
                        block[i] = new_block[i];
                    }
                StoreBlockLoop:
                    for (unsigned i = 0; i < n_later; i++) {
                        #pragma HLS UNROLL
                        row_blocks[i_bw][i] = new_block[row_size + i];
                    }
                } else {
                LoadBlockLoop:
                    for (unsigned i = 0; i < n_later; i++) {
                        #pragma HLS UNROLL
                        block[row_size + i] = row_blocks[i_bw][i];
                    }
                }
            }

            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        CastLoop:
            for (unsigned i_ff = 0; i_ff < n_filt; i_ff++) {
                #pragma HLS UNROLL
                res_pack[i_ff] = block[(i_grh * CONFIG_T::stride_width + i_grw) * n_filt + i_ff];
            }
            res.write(res_pack);

            if (i_ow == CONFIG_T::out_width - 1) {
                i_grw = CONFIG_T::first_group_width;
                i_bw = 0;
                if (i_oh == CONFIG_T::out_height - 1) {
                    i_grh = CONFIG_T::first_group_height;
                } else {
                    i_grh = i_grh == CONFIG_T::stride_height - 1 ? 0 : i_grh + 1;
                }
            } else if (i_grw == CONFIG_T::stride_width - 1) {
                i_grw = 0;
                i_bw++;
            } else {
                i_grw++;
            }
        }
    }
}

template <class data_T, class block_T, typename CONFIG_T> class ConvTransposeStreamLatency {
  public:
    static void compute(hls::stream<data_T> &data, hls::stream<block_T> &blocks,
                        typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                        typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        #pragma HLS INLINE
        conv_transpose_blocks_latency<data_T, block_T, CONFIG_T>(data, blocks, weights, biases);
    }
};

template <class data_T, class block_T, typename CONFIG_T> class ConvTransposeStreamResource {
  public:
    static void compute(hls::stream<data_T> &data, hls::stream<block_T> &blocks,
                        typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                        typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        #pragma HLS INLINE
        conv_transpose_blocks_resource<data_T, block_T, CONFIG_T>(data, blocks, weights, biases);
    }
};

template <class data_T, class res_T, typename CONFIG_T>
void conv_transpose_cl(hls::stream<data_T> &data, hls::stream<res_T> &res,
                       typename CONFIG_T::weight_t weights[CONFIG_T::n_buf * CONFIG_T::n_filt],
                       typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    #pragma HLS DATAFLOW

    typedef nnet::array<typename res_T::value_type, CONFIG_T::stride_height * CONFIG_T::stride_width * CONFIG_T::n_filt>
        block_t;

    hls::stream<block_t> blocks("conv_transpose_blocks");
    #pragma HLS STREAM variable=blocks depth=CONFIG_T::block_fifo_depth

    CONFIG_T::template conv_kernel<data_T, block_t, CONFIG_T>::compute(data, blocks, weights, biases);
    place_conv_transpose_blocks<block_t, res_T, CONFIG_T>(blocks, res);
}

} // namespace nnet

#endif
