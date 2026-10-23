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

// Called on the host. The trace pipe is always drained so the kernel does not stall when tracing is disabled.
template <class src_pipe, class dstType, size_t SIZE> void collect_trace_data(sycl::queue &q, const char *layer_name) {
    dstType *dst = nullptr;
    if (trace_enabled && trace_outputs) {
        if (trace_outputs->count(layer_name) > 0) {
            dst = (dstType *)(*trace_outputs)[layer_name];
        } else {
            std::cout << "Layer name: " << layer_name << " not found in debug storage!" << std::endl;
        }
    }

    constexpr auto srcTypeSize = std::tuple_size<typename ExtractPipeType<src_pipe>::value_type>{};
    for (size_t i = 0; i < SIZE / srcTypeSize; i++) {
        auto ctype = src_pipe::read(q);
        if (dst) {
            for (size_t j = 0; j < srcTypeSize; j++) {
                dst[i * srcTypeSize + j] = ctype[j].to_double();
            }
        }
    }
}

} // namespace nnet

#endif
