This is small conv1d 1 layer example. To run it:
1. Run the run_catapult.sh script.
2. Replace the conv1d_1layer_test.cpp in the my-Catapult-test with the conv1d_1layer_test.cpp a level up (if you would like the testbench to be self-checking).
3. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).

Note: You can create your own image matrix or filter and get the predictions by editing and running conv1d_1layer.py. 
