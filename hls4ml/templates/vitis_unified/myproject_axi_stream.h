#ifndef MYPROJECT_AXI_H_
#define MYPROJECT_AXI_H_

#include <iostream>
// hls-fpga-machine-learning insert include

// hls-fpga-machine-learning insert definitions

void MY_PROJECT_TOP_FUNC(hls::stream<dma_data_packet>& axi_input_stream,
                         hls::stream<dma_data_packet>& axi_output_stream);
#endif
