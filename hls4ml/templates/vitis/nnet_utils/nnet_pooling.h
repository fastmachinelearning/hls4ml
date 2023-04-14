#ifndef NNET_POOLING_H_
#define NNET_POOLING_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include <iostream>

namespace nnet {

// Return the maximum value from an array
template <typename T, int N> T max(T x[N]) {
    T y = x[0];
    for (int i = 1; i < N; i++) {
        y = x[i] > y ? x[i] : y;
    }
    return y;
}

template <int W, int N> ap_int<W> avg(ap_int<W> (&x)[N]) {
    // Use a wider accumulator than the input to avoid overflow
    ap_int<W + ceillog2(N)> tmp = 0;
    for (int i = 0; i < N; i++) {
        tmp += x[i];
    }
    tmp /= N;
    // Now cast back to original type
    ap_int<W> y = tmp;
    return tmp;
}

template <int W, int I, int N> ap_fixed<W, I> avg(ap_fixed<W, I> (&x)[N]) {
    // Use a wider accumulator than the input to avoid overflow
    ap_fixed<W + ceillog2(N), I + ceillog2(N)> tmp = 0;
    for (int i = 0; i < N; i++) {
        tmp += x[i];
    }
    tmp /= N;
    // Now cast back to original type
    ap_fixed<W, I> y = tmp;
    return y;
}

// Return the mean value of an array
template <typename T, int N> T avg(T (&x)[N]) {
    T y = 0;
    for (int i = 0; i < N; i++) {
        y += x[i];
    }
    y /= N;
    return y;
}

// Enumeration for pooling operation (max, avg, l2norm pooling)
enum Pool_Op { Max, Average }; // L2Norm };
template <typename T, int N, Pool_Op op> T pool_op(T (&x)[N]) {
    switch (op) {
    case Max:
        return max<T, N>(x);
    case Average:
        return avg(x);
        // case L2Norm: return l2norm<T, N>(x);
    }
}

template <typename T, Pool_Op op> T pad_val() {
    /*---
     *- In Tensorflow, pooling ignores the value in the padded cells
     *- For Avg pooling, return 0 (the divisior is modified to the
     *- area overlapping the unpadded image.
     *- For max pooling, return the most negative value for the type.
     *- TODO this is not really generic, it assumes fixed point or integer T
    ---*/
    switch (op) {
    case Max: {
        T x = 0;
        x[x.width - 1] = 1;
        return x;
        break;
    }
    case Average:
        return 0;
    }
}

struct pooling1d_config {
    // IO size
    static const unsigned n_in = 10;
    static const unsigned pool_width = 2;
    static const unsigned stride_width = 2;
    static const unsigned n_out = (n_in - pool_width) / stride_width + 1;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    // Pooling function
    static const Pool_Op pool_op = Max;
};

template <typename CONFIG_T> constexpr int pool_op_limit_1d() {
    return CONFIG_T::n_in * CONFIG_T::n_filt / CONFIG_T::reuse_factor;
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling1d_cl(data_T data[CONFIG_T::n_in * CONFIG_T::n_filt], res_T res[CONFIG_T::n_out * CONFIG_T::n_filt]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // TODO partition the arrays according to the reuse factor
    const int limit = pool_op_limit_1d<CONFIG_T>();
    #pragma HLS ALLOCATION function instances=pool_op<data_T, CONFIG_T::pool_width, CONFIG_T::pool_op> limit=limit
    // Add any necessary padding
    unsigned padded_width = CONFIG_T::n_in + CONFIG_T::pad_left + CONFIG_T::pad_right;
    if (CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0) {
        padded_width -= padded_width - (padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width);
    }

    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        // Loop over input image x in steps of stride
        for (int ii = 0; ii < padded_width; ii += CONFIG_T::stride_width) {
            data_T pool[CONFIG_T::pool_width];
            // Keep track of number of pixels in image vs padding region
            unsigned img_overlap = 0;
            // Loop over pool window x
            for (int jj = 0; jj < CONFIG_T::stride_width; jj++) {
                if (ii + jj < CONFIG_T::pad_left || ii + jj >= (padded_width - CONFIG_T::pad_right)) {
                    // Add padding
                    pool[jj] = pad_val<data_T, CONFIG_T::pool_op>();
                } else {
                    pool[jj] = data[(ii + jj - CONFIG_T::pad_left) * CONFIG_T::n_filt + ff];
                    img_overlap++;
                }
            }
            // do the pooling
            // TODO in the case of average pooling, need to reduce width to area of pool window
            // not overlapping padding region
            res[(ii / CONFIG_T::stride_width) * CONFIG_T::n_filt + ff] =
                pool_op<data_T, CONFIG_T::pool_width, CONFIG_T::pool_op>(pool);
            // If the pool op is Average, the zero-padding needs to be removed from the results
            if (CONFIG_T::pool_op == Average) {
                data_T rescale = static_cast<data_T>(CONFIG_T::pool_width) / img_overlap;
                res[(ii / CONFIG_T::stride_width) * CONFIG_T::n_filt + ff] *= rescale;
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling1d_cl(data_T data[CONFIG_T::n_in * CONFIG_T::n_filt], res_T res[CONFIG_T::n_filt]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);

    // TODO partition the arrays according to the reuse factor
    const int limit = pool_op_limit_1d<CONFIG_T>();
    #pragma HLS ALLOCATION function instances=pool_op<data_T, CONFIG_T::pool_width, CONFIG_T::pool_op> limit=limit

    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        data_T pool[CONFIG_T::n_in];
        for (int jj = 0; jj < CONFIG_T::n_in; jj++) {
            pool[jj] = data[jj * CONFIG_T::n_filt + ff];
        }
        // do the pooling
        res[ff] = pool_op<data_T, CONFIG_T::n_in, CONFIG_T::pool_op>(pool);
    }
}

struct pooling2d_config {
    // IO size
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_filt = 4;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned out_height = (in_height - pool_height) / stride_height + 1;
    static const unsigned out_width = (in_width - pool_width) / stride_width + 1;
    // Padding
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    // Pooling function
    static const Pool_Op pool_op = Max;
    // Reuse factor
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef float accum_t;
};

template <typename CONFIG_T> constexpr int pool_op_limit() {
    return DIV_ROUNDUP((CONFIG_T::out_height * CONFIG_T::out_width) * CONFIG_T::n_filt, CONFIG_T::reuse_factor);
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_cl(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt],
                  res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // TODO partition the arrays according to the reuse factor
    const int limit = pool_op_limit<CONFIG_T>();
    #pragma HLS ALLOCATION function instances=pool_op<data_T, CONFIG_T::pool_height*CONFIG_T::pool_width, \
        CONFIG_T::pool_op> limit=limit
    // Add any necessary padding
    unsigned padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
    unsigned padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;
    if (CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0) {
        padded_height -= padded_height - (padded_height / CONFIG_T::stride_height * CONFIG_T::stride_height);
        padded_width -= padded_width - (padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width);
    }

    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        // Loop over input image y in steps of stride
        for (int ii = 0; ii < padded_height; ii += CONFIG_T::stride_height) {
            // Loop over input image x in steps of stride
            for (int jj = 0; jj < padded_width; jj += CONFIG_T::stride_width) {
                data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
                // Keep track of number of pixels in image vs padding region
                unsigned img_overlap = 0;
                // Loop over pool window y
                for (int kk = 0; kk < CONFIG_T::stride_height; kk++) {
                    // Loop over pool window x
                    for (int ll = 0; ll < CONFIG_T::stride_width; ll++) {
                        if (ii + kk < CONFIG_T::pad_top || ii + kk >= (padded_height - CONFIG_T::pad_bottom) ||
                            jj + ll < CONFIG_T::pad_left || jj + ll >= (padded_width - CONFIG_T::pad_right)) {
                            // Add padding
                            pool[kk * CONFIG_T::stride_width + ll] = pad_val<data_T, CONFIG_T::pool_op>();
                        } else {
                            pool[kk * CONFIG_T::stride_width + ll] =
                                data[(ii + kk - CONFIG_T::pad_top) * CONFIG_T::in_width * CONFIG_T::n_filt +
                                     (jj + ll - CONFIG_T::pad_left) * CONFIG_T::n_filt + ff];
                            img_overlap++;
                        }
                    }
                }
                // do the pooling
                // TODO in the case of average pooling, need to reduce height * width to area of pool window
                // not overlapping padding region
                res[(ii / CONFIG_T::stride_height) * CONFIG_T::out_width * CONFIG_T::n_filt +
                    (jj / CONFIG_T::stride_width) * CONFIG_T::n_filt + ff] =
                    pool_op<data_T, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T::pool_op>(pool);
                // If the pool op is Average, the zero-padding needs to be removed from the results
                if (CONFIG_T::pool_op == Average) {
                    data_T rescale =
                        static_cast<data_T>(CONFIG_T::pool_height) * static_cast<data_T>(CONFIG_T::pool_width) / img_overlap;
                    res[(ii / CONFIG_T::stride_height) * CONFIG_T::out_width * CONFIG_T::n_filt +
                        (jj / CONFIG_T::stride_width) * CONFIG_T::n_filt + ff] *= rescale;
                }
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_cf(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt],
                  res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // TODO partition the arrays according to the reuse factor
    const int limit = pool_op_limit<CONFIG_T>();
    #pragma HLS ALLOCATION function instances=pool_op<data_T, CONFIG_T::pool_height*CONFIG_T::pool_width, \
        CONFIG_T::pool_op> limit=limit
    // Add any necessary padding
    unsigned padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
    unsigned padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;
    if (CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0) {
        padded_height -= padded_height - (padded_height / CONFIG_T::stride_height * CONFIG_T::stride_height);
        padded_width -= padded_width - (padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width);
    }

    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        // Loop over input image y in steps of stride
        for (int ii = 0; ii < padded_height; ii += CONFIG_T::stride_height) {
            // Loop over input image x in steps of stride
            for (int jj = 0; jj < padded_width; jj += CONFIG_T::stride_width) {
                data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
                // Keep track of number of pixels in image vs padding region
                unsigned img_overlap = 0;
                // Loop over pool window y
                for (int kk = 0; kk < CONFIG_T::stride_height; kk++) {
                    // Loop over pool window x
                    for (int ll = 0; ll < CONFIG_T::stride_width; ll++) {
                        if (ii + kk < CONFIG_T::pad_top || ii + kk >= (padded_height - CONFIG_T::pad_bottom) ||
                            jj + ll < CONFIG_T::pad_left || jj + ll >= (padded_width - CONFIG_T::pad_right)) {
                            // Add padding
                            pool[kk * CONFIG_T::stride_width + ll] = pad_val<data_T, CONFIG_T::pool_op>();
                        } else {
                            pool[kk * CONFIG_T::stride_width + ll] =
                                data[(ii + kk) * CONFIG_T::in_width + ff * CONFIG_T::in_width * CONFIG_T::in_height + ll +
                                     jj];
                            img_overlap++;
                        }
                    }
                }
                // do the pooling
                // TODO in the case of average pooling, need to reduce height * width to area of pool window
                // not overlapping padding region
                res[(ii / CONFIG_T::stride_height) * CONFIG_T::out_width + (jj / CONFIG_T::stride_width) +
                    ff * CONFIG_T::out_height * CONFIG_T::out_width] =
                    pool_op<data_T, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T::pool_op>(pool);
                // If the pool op is Average, the zero-padding needs to be removed from the results
                if (CONFIG_T::pool_op == Average) {
                    data_T rescale =
                        static_cast<data_T>(CONFIG_T::pool_height) * static_cast<data_T>(CONFIG_T::pool_width) / img_overlap;
                    res[(ii / CONFIG_T::stride_height) * CONFIG_T::out_width + (jj / CONFIG_T::stride_width) +
                        ff * CONFIG_T::out_height * CONFIG_T::out_width] *= rescale;
                }
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling2d_cl(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt],
                         res_T res[CONFIG_T::n_filt]) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height);

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    const int limit = pool_op_limit<CONFIG_T>();
    #pragma HLS ALLOCATION function instances=pool_op<data_T, CONFIG_T::pool_width * CONFIG_T::pool_height, \
        CONFIG_T::pool_op> limit=limit

FiltLoop:
    for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
        data_T pool[CONFIG_T::in_height * CONFIG_T::in_width];

    InputLoop:
        for (int i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width; i++) {
            pool[i] = data[i * CONFIG_T::n_filt + filt];
        }

        res[filt] = static_cast<res_T>(pool_op<data_T, CONFIG_T::in_height * CONFIG_T::in_width, CONFIG_T::pool_op>(pool));
    }
}

} // namespace nnet

#endif
