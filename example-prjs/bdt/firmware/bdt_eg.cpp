#include "BDT.h"
#include "parameters.h"
#include "bdt_eg.h"
void bdt_eg(input_arr_t x, score_arr_t score){
	#pragma HLS pipeline II = 1
	#pragma HLS unroll factor = 1
	#pragma HLS array_partition variable=x

	#pragma HLS array_partition variable=score

	bdt.decision_function(x, score);
}