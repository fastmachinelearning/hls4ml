This is small conv2d 1 layer using hard_sigmoid example. To run it:
1. Run "python hard_sigmoid.py" for the .h5 and .json files to be generated.
2. Run "python catapult.py" and now there should be a my-Catapult-test directory.
3. Run the run_catapult.sh script if you have made changes and want a fresh my-Catapult-test.
4. Replace the hard_sigmoid_test.cpp in the my-Catapult-test with the hard_sigmoid_test.cpp a level up (if you would like the testbench to be self-checking).
5. Move tb_input_features.dat and tb_output_predictions.dat to my-Catapult-test/tb_data (if you want two pre-computed examples).
6. Compile (make sure it is pointing to your path):
	/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/bin/g++ -g -std=c++11 -DSC_INCLUDE_DYNAMIC_PROCESSES -Wl,-rpath=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/lib,-rpath=/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/shared/lib ./hard_sigmoid_test.cpp ./firmware/hard_sigmoid.cpp -I/wv/scratch-baimar9c/venv/hls4ml/example-prjs/hard_sigmoid/my-Catapult-test -I/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/shared/include -L/wv/hlsb/CATAPULT/TOT/CURRENT/aol/Mgc_home/shared/lib -Wl,-Bstatic -lsystemc -Wl,-Bdynamic -lpthread -o /wv/scratch-baimar9c/venv/hls4ml/example-prjs/hard_sigmoid/my-Catapult-test/hard_sigmoid

NOTE:
The vivado script indicates an error while compiling from a bug in the vivado template. It is not running at this time.
