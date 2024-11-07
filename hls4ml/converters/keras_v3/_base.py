import typing
from typing import Any, Callable, Sequence

if typing.TYPE_CHECKING:
    import keras
    from keras.api import KerasTensor

T_kv3_handler = Callable[
    ['keras.Layer', Sequence['keras.KerasTensor'], Sequence['keras.KerasTensor']], tuple[dict[str, Any], ...]
]

registry: dict[str, T_kv3_handler] = {}


def register(cls: str | type):
    """Decorator to register a handler for a specific layer class. Suggested to decorate the `KerasV3LayerHandler` class.

    Parameters
    ----------
    cls : str|type
        If str, the key to register the handler under. If type, the class to register the handler for.

    Examples
    --------
    ```python
    @keras_dispatcher.register
    class MyLayerHandler(KerasV3LayerHandler):
        handles = ('my_package.src.submodule.MyLayer', 'MyLayer2')

        def handle(self, layer, inp_tensors, out_tensors):
            # handler code


    @keras_dispatcher.register('MyLayer3')
    def my_layer_handler(layer, inp_tensors, out_tensors):
        # handler code
    ```
    """

    def deco(func: T_kv3_handler):
        if isinstance(cls, str):
            registry[cls] = func
        for k in getattr(func, 'handles', ()):
            registry[k] = func
        return func

    if isinstance(cls, type):
        return deco(cls())
    return deco


def maybe_add_attrs(config: dict[str, Any], obj: Any, *attrs: str):
    for attr in attrs:
        if attr not in config and hasattr(obj, attr):
            config[attr] = getattr(obj, attr)


class KerasV3LayerHandler:
    """Base class for keras v3 layer handlers. Subclass this class to create a handler for a specific layer type."""

    handles = ()

    def __call__(
        self,
        layer: 'keras.Layer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        """Handle a keras layer. Return a tuple of dictionaries, each
        dictionary representing a layer (module) in the HLS model. One
        layer may correspond one or more dictionaries (e.g., layers with
        activation functions will be split into two layers).

        Some common attributes are automatically added to the dictionary
        if the handler returns a single dictionary. If the handler
        returns multiple dictionaries, the attributes must be added
        manually. Anything returned by the handler will override the
        automatic attributes.

        Automatic attributes: - name - class_name - module -
        input_keras_tensor_names - input_shape -
        output_keras_tensor_names

        If the layer has an activation function, an additional
        dictionary will be added to the return value representing the
        activation function.


        Parameters
        ----------
        layer : keras.Layer
            The layer to be converted to HLS configuration(s).
        in_tensors : Sequence[KerasTensor]
            The list of input tensors to the layer.
        out_tensors : Sequence[KerasTensor]
            The list of output tensors from the layer.

        Returns
        -------
        dict[str, Any] | tuple[dict[str, Any], ...]
            layer configuration(s) for the HLS model to be consumed by
            the ModelGraph constructor
        """  # noqa: E501
        import keras

        config0 = self.handle(layer, in_tensors, out_tensors)
        if isinstance(config0, tuple):
            return config0

        name = layer.name
        class_name = layer.__class__.__name__
        module = layer.__module__
        config1 = {
            'name': name,
            'class_name': class_name,
            'module': module,
            'input_keras_tensor_names': [t.name for t in in_tensors],
            'input_shape': [list(t.shape[1:]) for t in in_tensors],
            'output_keras_tensor_names': [t.name for t in out_tensors],
        }

        maybe_add_attrs(config1, layer, 'epsilon', 'use_bias', 'data_format')

        config1.update(config0)
        ret = (config1,)

        activation = getattr(layer, 'activation', None)
        if activation not in (keras.activations.linear, None):
            act_cls_name = activation.__class__.__name__
            act_config = {
                'class_name': 'Activation',
                'activation': act_cls_name,
                'name': f'{name}_{act_cls_name}',
            }
            ret = *ret, act_config
        return ret

    def handle(
        self,
        layer: 'keras.Layer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ) -> dict[str, Any] | tuple[dict[str, Any], ...]:
        return {}
