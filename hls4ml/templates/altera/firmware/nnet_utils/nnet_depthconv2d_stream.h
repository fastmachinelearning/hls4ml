#ifndef NNET_DEPTHCONV2D_STREAM_H_
#define NNET_DEPTHCONV2D_STREAM_H_

#include "nnet_conv2d_stream.h"
#include "nnet_depthconv_stream.h"
#include "nnet_types.h"

namespace nnet {

/*
 * Same as compute_output_buffer_2d, but the full kernel window is passed to depthwise_product instead of dense_resource
 */
template <class data_T, class data_window_T, class res_pipe, typename CONFIG_T>
void compute_depthwise_output_buffer_2d(
    const data_T &in_elem,
    nnet::shift_reg<typename data_T::value_type, CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right>
        line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan],
    data_window_T &kernel_window, const typename CONFIG_T::weight_t &weights, const typename CONFIG_T::bias_t &biases,
    int &pX, int &pY, int &sX, int &sY) {

    using res_T = typename ExtractPipeType<res_pipe>::value_type;

    // Thresholds
    constexpr int lShiftX = CONFIG_T::filt_width - 1;
    constexpr int lShiftY = CONFIG_T::filt_height - 1;

    // Step 1 - Shift line buffer
    [[intel::fpga_register]] typename data_T::value_type shift_buffer[CONFIG_T::filt_height][CONFIG_T::n_chan];
    nnet::shift_line_buffer_2d<data_T, CONFIG_T>(in_elem, line_buffer, shift_buffer);

    // Step 2 - Kernel shift
    nnet::kernel_shift_2d<data_T, data_window_T, CONFIG_T>(shift_buffer, kernel_window);

    // Check to see if we have a full kernel
    if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > (lShiftY - 1) && pX > (lShiftX - 1)) {
        // Step 3 - Depthwise multiplication
        [[intel::fpga_register]] res_T res_out;
        depthwise_product<data_window_T, res_T, CONFIG_T>(kernel_window, res_out, weights, biases);
        res_pipe::write(res_out);
    }

    // Reached end of image
    if ((pX + 1) == (CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right) &&
        (pY + 1) == (CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom)) {
        pX = 0;
        sX = 0;
        pY = 0;
        sY = 0;
        // Reached end of row
    } else if ((pX + 1) == (CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right)) {
        pX = 0;
        sX = 0;
        pY++;
        sY = ((sY - lShiftY) == 0) ? (sY - CONFIG_T::stride_height + 1) : (sY + 1);
        // Same row, same column, therefore, move to the right
    } else {
        pX++;
        sX = ((sX - lShiftX) == 0) ? (sX - CONFIG_T::stride_width + 1) : (sX + 1);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void depthwise_conv_2d_cl_stream() {

    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_element_T = typename data_arr_T::value_type;
    using data_window_T = array<data_element_T, CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan>;

    // Line buffer and kernel window
    [[intel::fpga_register]] nnet::shift_reg<data_element_T, CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right>
        line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan];
    [[intel::fpga_register]] data_window_T kernel_window;

    // An array of length CONFIG_T::n_chan, with elements set to zero (padding for each channel)
    constexpr auto padds = zero_array<data_arr_T>();

    // X position pixel
    int pX = 0;
    // Y position pixel
    int pY = 0;
    // X strides
    int sX = 0;
    // Y strides
    int sY = 0;

// Padding above input image
PaddingTopHeight:
    [[intel::loop_coalesce(2)]] for (int row = 0; row < CONFIG_T::pad_top; row++) {
    PaddingTopWidth:
        for (int col = 0; col < CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right; col++) {
            compute_depthwise_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                padds, line_buffer, kernel_window, CONFIG_T::weights, CONFIG_T::biases, pX, pY, sX, sY);
        }
    }

ReadInputHeight:
    [[intel::loop_coalesce(2)]] for (int row = 0; row < CONFIG_T::in_height; row++) {
    // Input image left-side padding
    PaddingLeftWidth:
        for (int col = 0; col < CONFIG_T::pad_left; col++) {
            compute_depthwise_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                padds, line_buffer, kernel_window, CONFIG_T::weights, CONFIG_T::biases, pX, pY, sX, sY);
        }

    // Read input image
    ReadInputWidth:
        for (int col = 0; col < CONFIG_T::in_width; col++) {
            compute_depthwise_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                data_pipe::read(), line_buffer, kernel_window, CONFIG_T::weights, CONFIG_T::biases, pX, pY, sX, sY);
        }

    // Input image right-side padding
    PaddingRightWidth:
        for (int col = 0; col < CONFIG_T::pad_right; col++) {
            compute_depthwise_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                padds, line_buffer, kernel_window, CONFIG_T::weights, CONFIG_T::biases, pX, pY, sX, sY);
        }
    }

// Padding below input image
PaddingBottomHeight:
    [[intel::loop_coalesce(2)]] for (int row = 0; row < CONFIG_T::pad_bottom; row++) {
    PaddingBottomWidth:
        for (int col = 0; col < CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right; col++) {
            compute_depthwise_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                padds, line_buffer, kernel_window, CONFIG_T::weights, CONFIG_T::biases, pX, pY, sX, sY);
        }
    }
}

} // namespace nnet

#endif
