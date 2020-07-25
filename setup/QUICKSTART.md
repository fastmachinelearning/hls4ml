# Quick start

For basic concepts to understand the tool, please visit the <a href="../CONCEPTS.html">Concepts</a> chapter. Here we give line-by-line instructions for simply running the tool out-of-the-box and getting a feel for the workflow.  

## Installation

```
pip install hls4ml
```

## Getting started

To with <span style="color:red">hls4ml</span>, we provide some default example models for conversion:

```python
import hls4ml

#Fetch a keras model from our example repository
#This will download our example model to your working directory and return an example configuration file
config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')

print(config) #You can print it to see some default parameters

#Convert it to a hls project
hls_model = hls4ml.converters.keras_to_hls(config)

# Print full list of example model if you want to explore more
hls4ml.utils.fetch_example_list()
```

After that, you can use <span style="color:red">Vivado HLS</span> to synthesize the model:

```python
#Use Vivado HLS to synthesize the model
#This might take several minutes
hls_model.build(synth=True)

#Print out the report if you want
hls4ml.report.read_vivado_report('my-hls-test')
```

Done! you've built your first project using <span style="color:red">hls4ml</span>! To learn more about our various API functionalities, check out our tutorials [here](https://github.com/hls-fpga-machine-learning/hls4ml-tutorial).

If you want to configure your model further, check out our [Configuration](../api/CONFIGURATION.md) page. 

Apart from our main API, we also support model conversion using a command line interface, check out our next section to find out more:

### Getting started with hls4ml command (optional)


To follow this tutorial, you must first download our `example-models` repository:

```bash
git clone https://github.com/hls-fpga-machine-learning/example-models.git
```
The model files, along with other configuration parameters, are defined in the `.yml` files.
Further information about `.yml` files can be found in [Configuration](../api/CONFIGURATION.md) page.

In order to create an example HLS project:

- Go to `example-models/` from the main directory: 

```bash
cd example-models/
```

- And use this command to translate a Keras model:

```bash
hls4ml convert -c keras-config.yml
```

This will create a new HLS project directory with an implementation of a model from the `example-models/keras/` directory.
To build the HLS project, do:

```bash
hls4ml build -p my-hls-test -a
```

This will create a Vivado HLS project with your model implmentation!

**NOTE:** For the last step, you can alternatively do the following to build the HLS project:

```Bash
cd my-hls-test
vivado_hls -f build_prj.tcl
```

`vivado_hls` can be controlled with:

```bash
vivado_hls -f build_prj.tcl "csim=1 synth=1 cosim=1 export=1"
```

Setting the additional parameters to `1` to `0` disables that step, but disabling `synth` also disables `cosim` and `export`.

**3) Further help:**

- For further information about how to use `hls4ml`, do: `hls4ml --help` or `hls4ml -h`

- If you need help for a particular `command`, `hls4ml command -h` will show help for the requested `command`

- We provide a detailed documentation for each of the command in the [Command Help](../COMMAND.md) section

**To uninstall `hls4ml`:** 

```bash
pip uninstall hls4ml
```

## Existing examples

- Examples of model files and weights can be found in [example_models](hhttps://github.com/hls-fpga-machine-learning/example-models) directory.

- Training codes and examples of resources needed to train the models can be found [here](https://github.com/hls-fpga-machine-learning/keras-training).

- Other examples of various HLS projects with examples of different machine learning algorithm implementations is in the directory [example-prjs](https://github.com/hls-fpga-machine-learning/hls4ml/tree/master/example-prjs).


