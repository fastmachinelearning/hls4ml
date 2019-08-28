# Quick start

For basic concepts to understand the tool, please visit the <a href="../CONCEPTS.html">Concepts</a> chapter.  
Here we give line-by-line instructions on how to build a first project and explain the various steps to producing an HLS4ML IP.

## Running the tool

These instructions are for simply running the tool out-of-the-box and getting a feel for the workflow.  

1) First clone our latest version from github and install it from the main directory: 
```
git clone -b pack6 --single-branch https://github.com/vloncar/hls4ml.git
cd hls4ml/
pip install -U .
```
**NOTE FOR THIS STEP:**
- If you don't have privileges to install in the configured environment, you can also pass `--user` to the `pip` command as well. 
- In the future, we plan to support `hls4ml` as a package on PyPI. After that you can simply install the software with `pip install hls4ml`

2) Activate 

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
