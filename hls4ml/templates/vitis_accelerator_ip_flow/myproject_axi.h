#ifndef MYPROJECT_AXI_H_
#define MYPROJECT_AXI_H_

#include <iostream>
// hls-fpga-machine-learning insert include

// hls-fpga-machine-learning insert definitions

void myproject_axi(hls::stream<dma_data_packet> &in, hls::stream<dma_data_packet> &out);
#endif
