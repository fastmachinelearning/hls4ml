#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

#include <assert.h>
#include <cstddef>
#include <cstdio>

namespace nnet {

// Fixed-size array
template <typename T, unsigned N> struct array {
    typedef T value_type;
    static const unsigned size = N;

    T data[N];

    array() {}

    array(T x) {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            data[i] = x;
        }
    }

    T &operator[](size_t pos) { return data[pos]; }

    const T &operator[](size_t pos) const { return data[pos]; }

    array &operator=(const array &other) {
        if (&other == this)
            return *this;

        assert(N == other.size && "Array sizes must match.");

        #pragma unroll
        for (unsigned i = 0; i < N; i++) {
            data[i] = other[i];
        }
        return *this;
    }
};

/*
 * HLS Shift Register Implementation
 * To verify a shift register is used in hardware, go to report.html > Area Analysis of System
 * Unrolling the shift loop minimizes resource usage and latency at the same time
 * The shift loop should be either fully unrolled or not unrolled at all
 * Unrolling with a specific unroll factor or pipelining with certain ii's, can cause an irregular access pattern, which
 * wouldn't allow shift register usage in RTL
 */
template <typename T, int N> struct shift_reg {
  private:
    T data[N];

  public:
    // Default constructor
    shift_reg() {}

    // Shift queue, insert new element and return element from the front
    T shift(T inp) {
        T out = data[N - 1];

        #pragma unroll
        for (int i = N - 1; i > 0; i--) {
            data[i] = data[i - 1];
        }
        data[0] = inp;

        return out;
    }

    T read(int pos) { return data[pos]; }
};

} // namespace nnet

#endif
