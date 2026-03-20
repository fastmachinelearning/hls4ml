// hls-fpga-machine-learning insert include

void load_input(hls::stream<dma_data_packet> &axi_input_stream, hls::stream<INPUT_LAYER_TYPE> &model_input_stream,
                int batch_size) {
load_input_loop:
    // send the data to the stream
    for (int q = 0; q < batch_size; q++) {
        for (unsigned chunk_idx = 0; chunk_idx < N_IN / INPUT_LAYER_TYPE::size; ++chunk_idx) {
            INPUT_LAYER_TYPE input_chunk;
            for (unsigned elem_idx = 0; elem_idx < INPUT_LAYER_TYPE::size; elem_idx++) {
                dma_data_packet axi_packet;
                axi_input_stream.read(axi_packet);
                input_chunk[elem_idx] = axi_packet.data;
            }
            model_input_stream.write(input_chunk);
        }
    }
}

void store_result(hls::stream<OUTPUT_LAYER_TYPE> &model_output_stream, hls::stream<dma_data_packet> &axi_output_stream,
                  int batch_size) {
store_result_loop:
    // send back the data
    for (int q = 0; q < batch_size; q++) {
        for (unsigned chunk_idx = 0; chunk_idx < N_OUT / OUTPUT_LAYER_TYPE::size; ++chunk_idx) {
            OUTPUT_LAYER_TYPE output_chunk = model_output_stream.read();
            for (unsigned elem_idx = 0; elem_idx < OUTPUT_LAYER_TYPE::size; elem_idx++) {
                dma_data_packet axi_packet;
                axi_packet.keep = -1;
                axi_packet.data = (OUTPUT_GMEM_TYPE)(output_chunk[elem_idx]);
                axi_packet.last = (q == (batch_size - 1)) && (((chunk_idx + 1) * (elem_idx + 1)) == N_OUT);
                axi_output_stream.write(axi_packet);
            }
        }
    }
}

void compute( // hls-fpga-machine-learning insert stream parameter,
    int batch_size) {
    for (int q = 0; q < batch_size; q++) {
        MY_PROJECT(model_input_stream, model_output_stream);
    }
}

void MY_PROJECT_TOP_FUNC(hls::stream<dma_data_packet> &axi_input_stream, hls::stream<dma_data_packet> &axi_output_stream,
                         int batch_size) {

    // hls-fpga-machine-learning insert interface

    // hls-fpga-machine-learning insert stream decl

    #pragma HLS DATAFLOW

    load_input(axi_input_stream, model_input_stream, batch_size);
    compute(model_input_stream, model_output_stream, batch_size);
    store_result(model_output_stream, axi_output_stream, batch_size);
}
