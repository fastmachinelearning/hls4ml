#ifndef NNET_CONV_STREAM_H_
#define NNET_CONV_STREAM_H_

#include "ap_shift_reg.h"
#include "nnet_common.h"
#include "hls_stream.h"

namespace nnet {

template<unsigned K, unsigned S, unsigned W>
unsigned scale_index(const unsigned idx) {
    #pragma HLS INLINE

    if (idx < K - S) {
        return idx;
    }

    constexpr unsigned nW = ((W - K) / S) * S + K; // Nearest W without unused pixels on the right
    constexpr unsigned sW = (DIV_ROUNDUP(K, S) - 1) * S + K; // Scaled W that behaves like original W
    if (idx >= nW) {
        return sW;
    }

    const unsigned r = nW - idx;
    if (r <= K - S) {
        return sW - r;
    }

    return K - S + (idx - (K - S)) % S;
}

template<class data_T, class res_T, typename CONFIG_T>
void mult_buffer(
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    res_T& res_pack,
    hls::stream<res_T>& res_stream,
    unsigned & outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE

    typename data_T::value_type data[CONFIG_T::kernel_size * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data complete
    typename res_T::value_type res[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res complete

    InitData: for (int id = 0; id < CONFIG_T::kernel_size * CONFIG_T::n_chan; id++) {
        #pragma HLS UNROLL
        data[id] = data_window[id].read();
    }

    #pragma HLS INLINE region
    if (CONFIG_T::strategy == nnet::latency) {
        dense_latency<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(data, res, weights, biases);
    } else {
        dense_resource<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(data, res, weights, biases);
    }

    CastLoop: for (unsigned jj = 0; jj < CONFIG_T::n_filt; jj++) {
        #pragma HLS UNROLL
        if (res_T::size / CONFIG_T::n_filt == 1) {
            res_pack[jj] = res[jj];
        } else {
            res_pack[outputs_ready * CONFIG_T::n_filt + jj] = res[jj];
        }
    }

    if (res_T::size / CONFIG_T::n_filt == 1) {
        res_stream.write(res_pack);
    } else {
        if (outputs_ready == (res_T::size / CONFIG_T::n_filt) - 1) {
            res_stream.write(res_pack);
            outputs_ready = 0;
        } else {
            outputs_ready++;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void compute_output(
    const data_T& in_elem,
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    hls::stream<res_T> &res,
    res_T &res_pack,
    unsigned &outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt],
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
            mult_buffer<data_T, res_T, CONFIG_T>(data_window, res_pack, res, outputs_ready, weights, biases);
        }
    }
}



// *************************************************
//       Line Buffer Implementation (Phil's)
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void kernel_shift(
    typename data_T::value_type shift_buffer[CONFIG_T::filt_height][CONFIG_T::n_chan],
    typename res_T::value_type kernel_window[CONFIG_T::filt_width * CONFIG_T::filt_height * CONFIG_T::n_chan]) {
  #pragma HLS inline
      
  // Shift kernel_window by one step to the left (manual shift operation)
  static const int filt_width = CONFIG_T::filt_width - 1;
  for (int i0 = 0; i0 < filt_width; i0++) {
    #pragma HLS PIPELINE II = 1
    for (unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) {
      for (unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) {
        // Shift every element in kernel_window to the left
        kernel_window[i1 * CONFIG_T::filt_width * CONFIG_T::n_chan + i0 * CONFIG_T::n_chan + i2] = kernel_window[i1 * CONFIG_T::filt_width * CONFIG_T::n_chan + (i0 + 1) * CONFIG_T::n_chan + i2];
      }
    }
  }

  // Insert shift_buffer column into right-most column of kernel
  static const int lastheight = (CONFIG_T::filt_width - 1) * CONFIG_T::n_chan;
  for (int i1 = 0; i1 < CONFIG_T::filt_height; i1++) {
    #pragma HLS UNROLL
    for (int i2 = 0; i2 < CONFIG_T::n_chan; i2++) {
      kernel_window[lastheight + i1 * CONFIG_T::filt_width * CONFIG_T::n_chan + i2] = shift_buffer[i1][i2];
    }
  }
}

template <class data_T, class res_T, typename CONFIG_T>
void shift_line_buffer(typename data_T::value_type data[CONFIG_T::n_chan],
                  ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[CONFIG_T::filt_height - 1][CONFIG_T::n_chan],
                  typename data_T::value_type kernel_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan]) {
  
  #pragma HLS PIPELINE

  // Temporary buffer for popped (shifted) elements
  typename data_T::value_type shift_buffer[CONFIG_T::filt_height][CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable = shift_buffer complete dim = 0

  for (int c = 0; c < CONFIG_T::n_chan; c++) {
    #pragma HLS UNROLL

    // Insert pixel(s) at end of shift buffer
    shift_buffer[CONFIG_T::filt_height - 1][c] = data[c];

    // Shift the shift buffer into the line buffer
    for (unsigned i1 = 1; i1 < CONFIG_T::filt_height; i1++) {
      #pragma HLS UNROLL
      typename data_T::value_type pop_elem = line_buffer[i1 - 1][c].shift(shift_buffer[CONFIG_T::filt_height - i1][c]); // Shift the line buffer, return the popped pixel
      shift_buffer[CONFIG_T::filt_height - i1 - 1][c] = pop_elem; // Popped element placed back into shift_buffer, one row up.
    }
  }
  kernel_shift<data_T, res_T, CONFIG_T>(shift_buffer, kernel_window);
}

template<class data_T, class res_T, typename CONFIG_T>
void compute_output(
    const data_T& in_elem,
    ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[CONFIG_T::filt_height - 1][CONFIG_T::n_chan],
    hls::stream<res_T> &res_stream,
    res_T &res_pack,
    unsigned &outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    #pragma HLS INLINE

    // Thresholds
    const static int lShiftX = CONFIG_T::filt_width - 1;
    const static int lShiftY = CONFIG_T::filt_height - 1;

    // Pixel Pointers
    static int pX = 0;
    static int pY = 0;

    typename data_T::value_type data_in[CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=data_in complete

    static typename data_T::value_type kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=kernel_data complete

    typename res_T::value_type res_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=res_out complete dim = 0

    // Read data: Stream -> array
    // TODO: data_T::size / CONFIG_T::n_chan != 1 case
    InitData: for (int i1 = 0; i1 < CONFIG_T::n_chan; i1++) {
        #pragma HLS UNROLL
        data_in[i1] = in_elem[i1];
    }

    // Add pixel to buffer
    nnet::shift_line_buffer<data_T, res_T, CONFIG_T>(data_in, line_buffer, kernel_data);

    // Check to see if we have a full kernel
    if ((pX - lShiftX) % CONFIG_T::stride_width == 0 && (pY - lShiftY) % CONFIG_T::stride_height == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
        
        // Dense multiply
        #pragma HLS INLINE region
        if (CONFIG_T::strategy == nnet::latency) {
            dense_latency<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(kernel_data, res_out, weights, biases);
        } else {
            dense_resource<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(kernel_data, res_out, weights, biases);
        }

        // Pack output
        CastLoop: for (unsigned jj = 0; jj < CONFIG_T::n_filt; jj++) {
        #pragma HLS UNROLL
            if (res_T::size / CONFIG_T::n_filt == 1) {
                res_pack[jj] = res_out[jj];
            } else {
                res_pack[outputs_ready * CONFIG_T::n_filt + jj] = res_out[jj];
            }
        }

        // Write output to stream when output ready
        if (res_T::size / CONFIG_T::n_filt == 1) {
            res_stream.write(res_pack);
        } else {
            if (outputs_ready == (res_T::size / CONFIG_T::n_filt) - 1) {
                res_stream.write(res_pack);
                outputs_ready = 0;
            } else {
                outputs_ready++;
            }
        }
    }

    // Pointer Housekeeping
    if (pX + 1 == CONFIG_T::in_width)  // Includes padding, end of line (padded)
    {
        pX = 0;
        if (pY + 1 == CONFIG_T::in_height) {  // Reached bottom of image
            pY = 0;
        } else {
            pY = pY + 1;
        }
    } else {
        pX = pX + 1;
    }

}

}
#endif
