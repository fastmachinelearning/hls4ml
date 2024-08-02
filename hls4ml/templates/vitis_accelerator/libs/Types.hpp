#pragma once

#include <cstdint>

template<class T, class U>
struct Batch {
    const T* dataIn;
    U* dataOut;
};

enum class FPGAType : uint8_t {
    DDR = 0,
    HBM = 1
};
