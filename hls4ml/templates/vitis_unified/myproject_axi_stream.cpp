// hls-fpga-machine-learning insert include

void load_input(hls::stream<dma_data_packet> &axi_input_stream, hls::stream<INPUT_LAYER_TYPE> &model_input_stream,
                hls::stream<bool> &model_tlast_stream) {
load_input_loop:
    dma_data_packet axi_packet;
    for (unsigned chunk_idx = 0; chunk_idx < N_IN / INPUT_LAYER_TYPE::size; ++chunk_idx) {
        INPUT_LAYER_TYPE input_chunk;
        for (unsigned elem_idx = 0; elem_idx < INPUT_LAYER_TYPE::size; elem_idx++) {
            axi_input_stream.read(axi_packet);
            input_chunk[elem_idx] = axi_packet.data;
            if (((chunk_idx + 1) * (elem_idx + 1)) == N_IN) {
                model_tlast_stream.write(axi_packet.last);
            }
        }
        model_input_stream.write(input_chunk);
    }
}

void store_result(hls::stream<OUTPUT_LAYER_TYPE> &model_output_stream, hls::stream<dma_data_packet> &axi_output_stream,
                  hls::stream<bool> &model_tlast_stream) {
store_result_loop:
    dma_data_packet axi_packet;
    axi_packet.keep = -1;
    for (unsigned chunk_idx = 0; chunk_idx < N_OUT / OUTPUT_LAYER_TYPE::size; ++chunk_idx) {
        OUTPUT_LAYER_TYPE output_chunk = model_output_stream.read();
        for (unsigned elem_idx = 0; elem_idx < OUTPUT_LAYER_TYPE::size; elem_idx++) {
            axi_packet.data = (OUTPUT_GMEM_TYPE)(output_chunk[elem_idx]);
            axi_packet.last = 0;
            if ((chunk_idx + 1) * (elem_idx + 1) == N_OUT) {
                axi_packet.last = model_tlast_stream.read();
            }
            axi_output_stream.write(axi_packet);
        }
    }
}

void MY_PROJECT_TOP_FUNC(hls::stream<dma_data_packet> &axi_input_stream, hls::stream<dma_data_packet> &axi_output_stream) {

    // hls-fpga-machine-learning insert interface

    #pragma HLS DATAFLOW
    bool is_last = false;

    // hls-fpga-machine-learning insert stream decl

    load_input(axi_input_stream, model_input_stream, model_tlast_stream);
    MY_PROJECT(model_input_stream, model_output_stream);
    store_result(model_output_stream, axi_output_stream, model_tlast_stream);
}
