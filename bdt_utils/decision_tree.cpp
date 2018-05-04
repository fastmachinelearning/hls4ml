#include "decision_tree.h"

output_t tree(input_t x[4]){//, output_t [1]){//bool activations[8]){
	
	int feature[15] = {2, 2, 3, -2, -2, 1, -2, -2, 1, 2, -2, -2, 3, -2, -2};
	double threshold[15] = {23.44139862,  11.824199676513672, 0.5, -2.0, -2.0, 1.4481849670410156, -2.0, -2.0, 1.160520076751709, 66.10154724121094, -2.0, -2.0, 0.5, -2.0, -2.0};
	double value[15] = {2.6207707758594277e-15, 0.1365256894460304, 0.16402818290828577, 1.2227455449746085, 0.7378096010736387, -0.0483438522135738, 1.0402677766761157, -1.8288521916244178, -0.5434167548988549, -0.18823180963055838, 0.36116594077887176, -2.657169427447734, -0.6867798397734569, -2.5527652622298365, -4.231537840365665};
	int children_left[15] = {1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1};
	int children_right[15] = {8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1};
	int parent[15] = {-1, 0, 1, 2, 2, 1, 5, 5, 0, 8, 9, 9, 8, 12, 12};
	
	bool comparison[15];
	bool active[15];
	bool active_leaf[8]; // Container for activations of leaves only
	double value_leaf[8]; // Container for values of leaves only
	
	// Execute all comparisons
	for(int i = 0; i < 15; i++){
		// Only non-leaf nodes do comparisons
		if(x[feature[i]] != -2){
			comparison[i] = x[feature[i]] <= threshold[i];
		}else{
			comparison[i] = true;
		}
	}
	
	// Determine node activity for all nodes
	int iLeaf = 0;
	for(int i = 0; i < 15; i++){
		// Root node is always active
		if(i == 0){
			active[i] = true;
		}else{
			// If this node is the left child of its parent
			if(i == children_left[parent[i]]){
				active[i] = comparison[parent[i]] && active[parent[i]];
			}else{ // Else it is the right child
				active[i] = !comparison[parent[i]] && active[parent[i]];
			}
		}
		if(children_left[i] == -1){ // is a leaf
			active_leaf[iLeaf] = active[i];
			value_leaf[iLeaf] = value[i];
			iLeaf++;
		}
	}
	
	// Set each bit of addr to the corresponding leaf activation
	ap_uint<8> active_leaf_addr;
	for(int i = 0; i < 8; i++){
		active_leaf_addr[i] = active_leaf[i];
	}

	// Select the score indexed by the active leaf
	output_t y;
	switch (active_leaf_addr){
	case 1 : y = value_leaf[0]; break;
	case 2 : y = value_leaf[1]; break;
	case 4 : y = value_leaf[2]; break;
	case 8 : y = value_leaf[3]; break;
	case 16 : y = value_leaf[4]; break;
	case 32 : y = value_leaf[5]; break;
	case 64 : y = value_leaf[6]; break;
	case 128 : y = value_leaf[7]; break;
	default : y = 0;
	}
	 
	return y;
}
