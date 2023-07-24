This is small softmax 1 layer example. To run it:
1. Run the run_catapult.sh script.
2. Replace the softmax_test.cpp in the my-Catapult-test with the softmax_test.cpp a level up (if you would like the testbench to be self-checking).
3. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).
4. Comment out "#define USE_AC_MATH" in nnet_activation.h.
5. In firmware/parameter.h change the softmax implementation to legacy.
6. Compile and run.

Note: You can create your own array and get the predictions by editing then running softmax.py. 
