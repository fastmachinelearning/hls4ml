This is small conv2d 1 layer applying a selu activation function example. To run it:
1. Run the run_catapult.sh script.
2. Replace the selu_test.cpp in the my-Catapult-test with the selu_test.cpp a level up (if you would like the testbench to be self-checking).
3. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).
4. If testing LUT version of elu (remove "#define USE_AC_MATH) Within the define.h file change conv2d_selu_table_t accuracy to "typedef ac_fixed<32,2,true> conv2d_selu_table_t".
5. Compile:
/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/bin/g++ -g -std=c++11 -DSC_INCLUDE_DYNAMIC_PROCESSES -Wl,-rpath=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/lib,-rpath=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/shared/lib ./selu_test.cpp ./firmware/selu.cpp -I/wv/USER/venv/hls4ml/example-prjs/selu/my-Catapult-test -I/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/shared/include -L/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/shared/lib -Wl,-Bstatic -lsystemc -Wl,-Bdynamic -lpthread -o /wv/USER/venv/hls4ml/example-prjs/selu/my-Catapult-test/selu

Note: You can create your own array and get the predictions by editing then running selu.py. 
