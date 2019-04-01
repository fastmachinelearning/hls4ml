#ifndef TEST_GALAPAGOS_H_
#define TEST_GALAPAGOS_H_

#include <cstddef>
#include <cstring>
#include "galapagos_packet.h"

#ifdef CPU
#include "galapagos_stream.hpp"
void test_galapagos(galapagos::stream * in, galapagos::stream * out);
#else
void test_galapagos(hls::stream<galapagos_packet> * in, hls::stream<galapagos_packet> * out);
#endif

#endif
