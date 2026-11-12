#ifndef NNET_PADDING_H_
#define NNET_PADDING_H_

#include <math.h>

namespace nnet {

// Binary data stores -1 as 0. When it is padded, it is copied as -1 and +1 into a type that also holds the padded
// zeros.
template <class res_T, bool binary_input, class data_T> inline res_T pad_copy(data_T x) {
    return binary_input ? (x == 0 ? res_T(-1) : res_T(1)) : res_T(x);
}

struct padding1d_config {
    static const unsigned n_chan = 10;
    static const unsigned in_width = 10;
    static const unsigned out_width = 10;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool binary_input = false;
};

template <class data_T, class res_T, typename CONFIG_T>
void zeropad1d_cf(data_T data[CONFIG_T::n_chan * CONFIG_T::in_width], res_T res[CONFIG_T::n_chan * CONFIG_T::out_width]) {
    //#pragma HLS PIPELINE

    for (int j = 0; j < CONFIG_T::n_chan; j++) {
        for (int i = 0; i < CONFIG_T::pad_left; i++) {
            *(res++) = 0;
        }

        for (int i = 0; i < CONFIG_T::in_width; i++) {
            *(res++) = pad_copy<res_T, CONFIG_T::binary_input>(*(data++));
        }

        for (int i = 0; i < CONFIG_T::pad_right; i++) {
            *(res++) = 0;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void zeropad1d_cl(data_T data[CONFIG_T::n_chan * CONFIG_T::in_width], res_T res[CONFIG_T::n_chan * CONFIG_T::out_width]) {
    //#pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::pad_left; i++) {
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            *(res++) = 0;
        }
    }

    for (int i = 0; i < CONFIG_T::in_width; i++) {
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            *(res++) = pad_copy<res_T, CONFIG_T::binary_input>(*(data++));
        }
    }

    for (int i = 0; i < CONFIG_T::pad_right; i++) {
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            *(res++) = 0;
        }
    }
}

struct padding2d_config {
    static const unsigned n_chan = 10;
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned out_height = 10;
    static const unsigned out_width = 10;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool binary_input = false;
};

template <class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cf(data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
                  res_T res[CONFIG_T::n_chan * CONFIG_T::out_height * CONFIG_T::out_width]) {
    //#pragma HLS PIPELINE

    for (int k = 0; k < CONFIG_T::n_chan; k++) {

        for (int i = 0; i < CONFIG_T::pad_top; i++) {
            for (int j = 0; j < CONFIG_T::out_width; j++) {
                *(res++) = 0;
            }
        }

        for (int i = 0; i < CONFIG_T::in_height; i++) {
            for (int j = 0; j < CONFIG_T::pad_left; j++) {
                *(res++) = 0;
            }
            for (int j = 0; j < CONFIG_T::in_width; j++) {
                *(res++) = pad_copy<res_T, CONFIG_T::binary_input>(*(data++));
            }
            for (int j = 0; j < CONFIG_T::pad_right; j++) {
                *(res++) = 0;
            }
        }

        for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
            for (int j = 0; j < CONFIG_T::out_width; j++) {
                *(res++) = 0;
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl(data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
                  res_T res[CONFIG_T::n_chan * CONFIG_T::out_height * CONFIG_T::out_width]) {
    //#pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::pad_top; i++) {
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = 0;
            }
        }
    }

    for (int i = 0; i < CONFIG_T::in_height; i++) {
        for (int j = 0; j < CONFIG_T::pad_left; j++) {
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = 0;
            }
        }
        for (int j = 0; j < CONFIG_T::in_width; j++) {
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = pad_copy<res_T, CONFIG_T::binary_input>(*(data++));
            }
        }
        for (int j = 0; j < CONFIG_T::pad_right; j++) {
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = 0;
            }
        }
    }

    for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = 0;
            }
        }
    }
}

} // namespace nnet

#endif
