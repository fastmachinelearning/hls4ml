#include "decision_tree.h"

#include "ap_int.h"
#include "ap_fixed.h"
#include "ap_cint.h"
#include <iostream>

typedef ap_fixed<18,8> input_t;
typedef ap_fixed<18,8> output_t;

int main(){

	input_t x[6][4] = {{0.7, 1.2, 15.0, 1.0}, // score: 1.04
					   {0.7, 1.5, 15.0, 1.0}, // score: -1.829
					   {0.3, 1.0, 30., 0.0 }, // score: 0.361
					   {0.3, 1.0, 100., 0.0 }, // score: -2.657
					   {0.3, 2.0, 100., 0.0 }, // score: -2.553
					   {0.3, 2.0, 100., 1.0 }}; // score: -4.232
	output_t y;

	for(int i = 0; i < 6; i++){
		y = tree(x[i]);
		std::cout << y << std::endl;

	}
}
