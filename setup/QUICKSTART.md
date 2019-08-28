# Quick start

For basic concepts to understand the tool, please visit the <a href="../CONCEPTS.html">Concepts</a> chapter.  
Here we give line-by-line instructions on how to build a first project and explain the various steps to producing an HLS4ML IP.

## Running the tool

These instructions are for simply running the tool out-of-the-box and getting a feel for the workflow.  

```
git clone https://github.com/hls-fpga-machine-learning/hls4ml.git -b v0.1.5
source install_miniconda3.sh
source install.sh
cd hls4ml/keras-to-hls
source activate hls4ml-env
python keras-to-hls.py -c keras-config.yml
```

This will create a new HLS project directory with an implementation of a model from the `example-keras-model-files` directory.
The model files, along with other configuration parameters, are defined in the `keras-config.yml` file.
To run the HLS project, do:

```
cd my-hls-test
vivado_hls -f build.tcl
```

This will create a Vivado HLS project with your model implmentation!

##Existing examples

Other examples of various HLS projects with examples of different machine learning algorithm implementations is in the directory: `example-prjs`.
