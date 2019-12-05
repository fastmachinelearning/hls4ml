   mnist_mlp mnist_mlp_INST(
      .clk(clk),
      .rst(rst),
      .input1_rsc_rdy(input1_rsc_bus.rdy),
      .input1_rsc_vld(input1_rsc_bus.vld),
      .input1_rsc_dat(input1_rsc_bus.dat),
      .output1_rsc_rdy(output1_rsc_bus.rdy),
      .output1_rsc_vld(output1_rsc_bus.vld),
      .output1_rsc_dat(output1_rsc_bus.dat),
      .const_size_in_1_rsc_vld(const_size_in_1_rsc_bus.vld),
      .const_size_in_1_rsc_dat(const_size_in_1_rsc_bus.dat),
      .const_size_out_1_rsc_vld(const_size_out_1_rsc_bus.vld),
      .const_size_out_1_rsc_dat(const_size_out_1_rsc_bus.dat)
      );
