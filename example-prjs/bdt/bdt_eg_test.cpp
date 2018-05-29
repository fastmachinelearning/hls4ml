#include "BDT.h"
#include "firmware/parameters.h"
#include "firmware/bdt_eg.h"
int main(){
	input_arr_t x = {0, 0};
	score_arr_t score;
	myproject(x, score);
	for(int i = 0; i < n_classes; i++){
		std::cout << score[i] << ", ";
	}
	std::cout << std::endl;
	return 0;
}