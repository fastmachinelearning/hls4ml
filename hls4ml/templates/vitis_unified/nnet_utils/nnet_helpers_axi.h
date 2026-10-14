#ifndef NNET_HELPERS_AXI_H
#define NNET_HELPERS_AXI_H

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include <iostream>
#include <vector>

namespace nnet {

#ifndef __SYNTHESIS__

template <class pack_T, class src_T, size_t SIZE> void convert_data_axis(src_T *src, hls::stream<pack_T> &dst) {
    for (size_t i = 0; i < SIZE; i++) {
        pack_T pack;
        pack.data = src[i];
        pack.keep = -1;
        pack.last = (i == SIZE - 1) ? 1 : 0;
        dst.write(pack);
    }
}

template <class pack_T, class src_T, size_t SIZE> void convert_data_axis(std::vector<src_T> &src, hls::stream<pack_T> &dst) {
    for (size_t i = 0; i < SIZE; i++) {
        pack_T pack;
        pack.data = src[i];
        pack.keep = -1;
        pack.last = (i == SIZE - 1) ? 1 : 0;
        dst.write(pack);
    }
}

template <class pack_T, class dst_T, size_t SIZE> void convert_data_axis(hls::stream<pack_T> &src, dst_T *dst) {
    for (size_t i = 0; i < SIZE; i++) {
        pack_T pack = src.read();
        dst[i] = dst_T(pack.data);
    }
}

#endif

template <class res_T, size_t SIZE>
void print_result_axis(hls::stream<res_T> &result, std::ostream &out, bool keep = false) {
    for (int i = 0; i < SIZE; i++) {
        res_T res_pack = result.read();
        out << res_pack.data << " ";
        if (keep)
            result.write(res_pack);
    }
    out << std::endl;
}

template <class data_T, size_t SIZE> void fill_zero_axi(hls::stream<data_T> &data, bool reqLast) {
    for (int i = 0; i < SIZE; i++) {
        data_T data_pack;
        data_pack.data = 0;
        data_pack.keep = -1;
        data_pack.last = reqLast && (i == (SIZE - 1)) ? 1 : 0;
        data.write(data_pack);
    }
}

} // namespace nnet

#endif
