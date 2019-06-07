# Documentation for Large Layer

A collection of images and model for documenting and debugging the large layer implementation.

## Dia

The image file ([nnet_large_layer.dia](nnet_large_layer.dia)) is in [Dia Diagram Editor](https://wiki.gnome.org/Apps/Dia) format.

To convert it in PDF format:
```
make dia2pdf
```

### SVG Files

Dia does integrate a formula editor. The solution is to import image files (for the best quality use SVG) from an external editor. For example you can use an [Online LaTeX Equation Editor](https://latex.codecogs.com/eqneditor/editor.php).

## Debugging

[nnet_large_layer.py](nnet_large_layer.py) is Python implementation of the Large Layer. Pay attention it may be not upated to the C++ implementation [../nnet_utils/nnet_large_layer.h](../nnet_utils/nnet_large_layer.h).

To run it:
```
make run
```
