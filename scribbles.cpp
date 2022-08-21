    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer4_t layer4_out[N_EDGE*LAYER4_OUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::edgeblock<input2_t, input3_t, layer4_t, config4>(node_attr, edge_attr, edge_index, layer4_out, R1_w0, R1_b0, R1_w1, R1_b1, R1_w2, R1_b2, R1_w3, R1_b3); // R1

    layer5_t layer5_out[N_NODE*LAYER5_OUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::edge_aggregate<input2_t, input3_t, layer5_t, aggregation_config5>(layer4_out, edge_index, layer5_out); // aggr5

    layer6_t layer6_out[N_NODE*LAYER6_OUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::nodeblock<input_t, layer6_t, config6>(node_attr, layer5_out, layer6_out, O_w0, O_b0, O_w1, O_b1, O_w2, O_b2, O_w3, O_b3); // O

    layer7_t layer7_out[N_EDGE*LAYER7_OUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::edgeblock<input2_t, input3_t, layer7_t, config7>(layer6_out, layer4_out, edge_index, layer7_out, R2_w0, R2_b0, R2_w1, R2_b1, R2_w2, R2_b2, R2_w3, R2_b3); // R2

    layer8_t layer8_out[N_EDGE*LAYER8_OUT_DIM];
    nnet::residualBlock<input2_t,input2_t, layer8_t, config8>(layer6_out,layer7_out);

    nnet::sigmoid<layer8_t, result_t, sigmoid_config9>(layer7_out, layer9_out); // final_act
