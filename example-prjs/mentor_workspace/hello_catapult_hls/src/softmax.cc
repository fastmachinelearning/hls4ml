#include <ac_math/ac_softmax_pwl.h>
using namespace ac_math;

const int tb_size = 20;

typedef ac_fixed<18, 6, true, AC_TRN, AC_SAT> input_type;
typedef ac_fixed<18, 6, false, AC_TRN, AC_SAT> output_type;

#pragma hls_design top
void top_module(const input_type (&input)[tb_size], output_type (&output)[tb_size])
{
    ac_softmax_pwl(input,output);
}

#ifndef __SYNTHESIS__
#include <mc_scverify.h>

CCS_MAIN(int arg, char **argc) {
    // Input and output buffers.
    input_type input[tb_size];
    output_type output[tb_size];

    // Initialize input buffer.
    for (int i = 0; i < tb_size; i++) {
        input[i] = 9 - i;
    }

    // Top module.
    CCS_DESIGN(top_module)(input, output);

    CCS_RETURN (0);
}
#endif
