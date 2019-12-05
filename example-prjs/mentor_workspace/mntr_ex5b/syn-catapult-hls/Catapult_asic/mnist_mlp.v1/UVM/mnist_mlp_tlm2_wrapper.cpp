#include <uvmc.h>
using namespace uvmc;
#include "mnist_mlp_tlm2_wrapper.h"

int sc_main(int argc, char *argv[])
{
  mnist_mlp_tlm2_wrapper  mnist_mlp_tlm2_wrapper_INST("mnist_mlp_tlm2_wrapper_INST");
  uvmc_connect(mnist_mlp_tlm2_wrapper_INST.input1_rsc_target,"gp_input1_rsc_ae");
  uvmc_connect(mnist_mlp_tlm2_wrapper_INST.output1_rsc_initiator,"gp_output1_rsc_ap");
  uvmc_connect(mnist_mlp_tlm2_wrapper_INST.const_size_in_1_rsc_initiator,"gp_const_size_in_1_rsc_ap");
  uvmc_connect(mnist_mlp_tlm2_wrapper_INST.const_size_out_1_rsc_initiator,"gp_const_size_out_1_rsc_ap");
  sc_start(-1);
  return 0;
}
