#include <ac_math/ac_sigmoid_pwl.h>

using namespace ac_math;

typedef ac_fixed<10, 5, false, AC_RND, AC_SAT> input_type;
typedef ac_fixed<20, 2, false, AC_RND, AC_SAT> output_type;

#pragma hls_design top
void project(
        const input_type &input,
        output_type &output
        )
{
    ac_sigmoid_pwl(input,output);
}

#ifndef __SYNTHESIS__
#include <mc_scverify.h>

CCS_MAIN(int arg, char **argc)
{
    input_type input = 3.5;
    output_type output;
    CCS_DESIGN(project)(input, output);
    CCS_RETURN (0);
}
#endif
