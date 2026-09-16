#ifndef NNET_HELPERS_H
#define NNET_HELPERS_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace nnet {

template <typename data_T> constexpr data_T minval() {
    if constexpr (std::numeric_limits<data_T>::is_specialized) {
        return std::numeric_limits<data_T>::lowest();
    } else {
        if (!data_T::sign)
            return data_T(0);
        else
            return data_T(-(1LL << (data_T::i_width - 1)));
    }
}

constexpr unsigned ceil_log2(unsigned x) {
    if (x == 0)
        return 0;
    unsigned res = 0;
    x -= 1;
    while (x > 0) {
        x >>= 1;
        res++;
    }
    return res;
}

constexpr unsigned floor_log2(unsigned x) {
    if (x == 0)
        return 0;
    unsigned res = 0;
    while (x > 0) {
        x >>= 1;
        res++;
    }
    return res - 1;
}

// This func takes the input size and rounds to nearest power of 2
constexpr unsigned numbanks_round(unsigned x) {
    unsigned p = floor_log2(x);
    return 1 << p;
}

template <class srcType, class dest_pipe, size_t SIZE> void convert_data(sycl::queue &q, srcType *src) {
    constexpr auto dstTypeSize = std::tuple_size<typename ExtractPipeType<dest_pipe>::value_type>{};
    for (size_t i = 0; i < SIZE / dstTypeSize; i++) {
        typename ExtractPipeType<dest_pipe>::value_type ctype;
        for (size_t j = 0; j < dstTypeSize; j++) {
            ctype[j] = src[i * dstTypeSize + j];
        }
        dest_pipe::write(q, ctype);
    }
}

template <class src_pipe, class dstType, size_t SIZE> void convert_data_back(sycl::queue &q, dstType *dst) {
    constexpr auto srcTypeSize = std::tuple_size<typename ExtractPipeType<src_pipe>::value_type>{};
    for (size_t i = 0; i < SIZE / srcTypeSize; i++) {
        auto ctype = src_pipe::read(q);
        for (size_t j = 0; j < srcTypeSize; j++) {
            dst[i * srcTypeSize + j] = ctype[j].to_double();
        }
    }
}

extern bool trace_enabled;
extern std::map<std::string, void *> *trace_outputs;
extern size_t trace_type_size;

// constexpr int ceillog2(int x) { return (x <= 2) ? 1 : 1 + ceillog2((x + 1) / 2); }
// replace with template metaprogramming
template <int n> struct ceillog2 {
    enum { val = 1 + ceillog2<((n + 1) / 2)>::val };
};

template <> struct ceillog2<2> {
    enum { val = 1 };
};

template <> struct ceillog2<1> {
    enum { val = 0 };
};

// constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }
// replace with template metaprogramming
template <int n> struct floorlog2 {
    enum { val = 1 + floorlog2<(n / 2)>::val };
};

template <> struct floorlog2<1> {
    enum { val = 0 };
};

template <> struct floorlog2<0> {
    enum { val = 0 };
};

// constexpr int pow2(int x) { return x == 0 ? 1 : 2 * pow2(x - 1); }
// replace with template metaprogramming
template <int n> struct pow2 {
    enum { val = 2 * pow2<(n - 1)>::val };
};

template <> struct pow2<0> {
    enum { val = 1 };
};

template <class data_T, class save_T> void save_output_array(data_T *data, save_T *ptr, size_t layer_size) {
    for (int i = 0; i < layer_size; i++) {
        ptr[i] = static_cast<save_T>(data[i].to_double());
    }
}

// We don't want to include save_T in this function because it will be inserted into myproject.cpp
// so a workaround with element size is used
template <class data_T> void save_layer_output(data_T *data, const char *layer_name, size_t layer_size) {
    if (!trace_enabled)
        return;

    if (trace_outputs) {
        if (trace_outputs->count(layer_name) > 0) {
            if (trace_type_size == 4) {
                save_output_array(data, (float *)(*trace_outputs)[layer_name], layer_size);
            } else if (trace_type_size == 8) {
                save_output_array(data, (double *)(*trace_outputs)[layer_name], layer_size);
            } else {
                std::cout << "Unknown trace type!" << std::endl;
            }
        } else {
            std::cout << "Layer name: " << layer_name << " not found in debug storage!" << std::endl;
        }
    } else {
        std::ostringstream filename;
        filename << "./tb_data/" << layer_name << "_output.log"; // TODO if run as a shared lib, path should be ../tb_data
        std::fstream out;
        out.open(filename.str(), std::ios::app);
        assert(out.is_open());
        for (int i = 0; i < layer_size; i++) {
            out << data[i] << " "; // We don't care about precision in text files
        }
        out << std::endl;
        out.close();
    }
}

} // namespace nnet

#endif
