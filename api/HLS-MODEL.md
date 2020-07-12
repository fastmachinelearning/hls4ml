## HLS Model Class

---

This page documents our hls_model class usage. You can generate generate an hls model object from a keras model through the API:

```python
import hls4ml

# Generate a simple configuration from keras model
config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')

# Convert to an hls model
hls_model = hls4ml.converters.convert_from_keras_model(keras_model, hls_config=config, output_dir='test_prj')
```

After that, you can use several methods in that object. Here is a list of all the methods:

- [predict]()
- [trace]()
- [compile]()
- [build]()
- [write]()

Similar functionalities are also supported through command line interface. If you prefer using them, refer to Command Help section. 

---
### `write` method

---
### `compile` method

---
### `predict` method

---
### `build` method

---
### `trace` method