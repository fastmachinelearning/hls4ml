#ifndef NNET_RELOADABLE_H_
#define NNET_RELOADABLE_H_

#include <sycl/sycl.hpp>
#include <utility>

namespace nnet {

constexpr int mode_infer = 0;
constexpr int mode_load_weights = 1;

// Without host_access_none the compiler adds host access ports, which share an arbitrated read-write port
// with the kernel and force the memory into true dual-port mode
using reloadable_weight_props = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::oneapi::experimental::device_image_scope, sycl::ext::oneapi::experimental::host_access_none));

// Must be a plain array: wrapping it in a struct makes the whole struct a single memory word
template <class T, std::size_t N>
using reloadable_weight_memory = sycl::ext::oneapi::experimental::device_global<T[N], reloadable_weight_props>;

// Stands in for the nnet::array of constant weights. Reads come from the set selected at run time, and load()
// fills that set from load_pipe, one weight per word in its low bits. A device_global can only be accessed from
// one kernel or task function, so reads and loads of a given weight must happen in the same function.
template <class T, std::size_t N, class load_pipe, auto &...sets> struct reloadable_weights {
    using value_type = T;
    int set;

    T operator[](std::size_t i) const {
        if constexpr (sizeof...(sets) == 1) {
            return first(i);
        } else {
            return select(i, std::make_index_sequence<sizeof...(sets)>{});
        }
    }

    void load() const {
        for (std::size_t i = 0; i < N; i++) {
            auto word = load_pipe::read();
            T val;
            val.set_slc(0, word.template slc<T::width>(0));
            if constexpr (sizeof...(sets) == 1) {
                ((sets[i] = val), ...);
            } else {
                store(i, val, std::make_index_sequence<sizeof...(sets)>{});
            }
        }
    }

  private:
    T first(std::size_t i) const {
        T val;
        ((val = sets[i]), ...);
        return val;
    }

    template <std::size_t... I> T select(std::size_t i, std::index_sequence<I...>) const {
        T val = 0;
        ((set == I ? (void)(val = sets[i]) : void()), ...);
        return val;
    }

    template <std::size_t... I> void store(std::size_t i, T val, std::index_sequence<I...>) const {
        ((set == I ? (void)(sets[i] = val) : void()), ...);
    }
};

// Task function for a layer with reloadable weights in io_stream: either loads the weights or runs the layer
template <auto layer_func, class... weights_T> void reloadable_task(int mode, weights_T... weights) {
    if (mode == mode_load_weights) {
        (weights.load(), ...);
    } else {
        layer_func(weights...);
    }
}

// In io_stream, the kernel passes the weights from the load interface on to the task that owns them
template <class src_pipe, class dst_pipe, std::size_t N> void forward_weights() {
    for (std::size_t i = 0; i < N; i++) {
        dst_pipe::write(src_pipe::read());
    }
}

// Host side: convert values to the weight type and write their bits to the load interface
template <class T, std::size_t N, class load_pipe, class src_T> void write_weights(sycl::queue &q, const src_T *src) {
    for (std::size_t i = 0; i < N; i++) {
        T val = src[i];
        typename ExtractPipeType<load_pipe>::value_type word = 0;
        word.set_slc(0, val.template slc<T::width>(0));
        load_pipe::write(q, word);
    }
}

} // namespace nnet

#endif
