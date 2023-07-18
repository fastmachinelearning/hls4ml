This is small relu 1 layer example. To run it:
1. Run the run_catapult.sh script.
2. Replace the elu_test.cpp in the my-Catapult-test with the elu_test.cpp a level up (if you would like the testbench to be self-checking).
3. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).
4. Remove "#define USE_AC_MATH" from nnet_activation.h to test lookup table version of elu.

Note: You can create your own array and get the predictions by editing then running elu.py. 
