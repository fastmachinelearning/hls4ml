"""
keras_layers.py
================
Keras layer definitions for hexagonal-camera CNN models.

This file defines the three custom ``tf.keras.layers.Layer`` subclasses
required by the ``indexed_geometry`` hls4ml Extension API package:

    - ``NeighborGatherLayer``      — gathers neighbor features per pixel
    - ``IndexedConvolutionLayer``  — applies a Conv2D/Conv3D per pixel
    - ``IndexedPoolingLayer``      — applies a MaxPool/AvgPool per pixel

These layer definitions must be available at model-load time when using
``tf.keras.models.load_model()`` with ``custom_objects``.
"""

import tensorflow as tf


class NeighborGatherLayer(tf.keras.layers.Layer):
    """
    Gathers the features of each pixel's neighbors into a new dimension.

    Given an input tensor of shape ``[batch, n_pixels, n_features]`` and
    a fixed neighbor index map of shape ``[n_pixels, n_neighbors]``, this
    layer produces an output of shape:

        - ``[batch, n_pixels, n_neighbors, n_features]``  when ``use_3d_conv=False``
        - ``[batch, n_pixels, n_neighbors, n_time, n_features]``  when ``use_3d_conv=True``

    Border pixels have one or more neighbor slots set to ``-1`` in the
    index map. These slots are zero-masked in the output.

    Parameters
    ----------
    neighbor_indices : array-like, shape [n_pixels, n_neighbors]
        Integer index map. ``-1`` denotes a missing neighbor (border pixel).
    use_3d_conv : bool
        If ``True``, assumes a temporal dimension in the input and produces
        a 4-D gather output suitable for Conv3D. If ``False``, produces a
        3-D output suitable for Conv2D.
    """

    def __init__(self, neighbor_indices, use_3d_conv, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_indices = tf.convert_to_tensor(neighbor_indices, dtype=tf.int32)
        self.mask = tf.cast(tf.not_equal(self.neighbor_indices, -1), tf.float32)
        # tf.gather rejects -1 as an out-of-range index (it does not wrap
        # around); clamp border-pixel slots (index == -1) to a valid,
        # arbitrary index (0) for the gather itself. The gathered value at
        # those positions is discarded by the mask multiplication below,
        # so which valid index is substituted does not affect the result.
        self.gather_indices = tf.maximum(self.neighbor_indices, 0)
        self.use_3d_conv = use_3d_conv

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        tiled_indices = tf.expand_dims(self.gather_indices, axis=0)
        tiled_indices = tf.tile(tiled_indices, [batch_size, 1, 1])
        neighbor_feats = tf.gather(inputs, tiled_indices, batch_dims=1, axis=1)
        tiled_mask = tf.expand_dims(self.mask, axis=0)
        tiled_mask = tf.tile(tiled_mask, [batch_size, 1, 1])
        tiled_mask = tf.expand_dims(tiled_mask, axis=-1)
        if self.use_3d_conv:
            tiled_mask = tf.expand_dims(tiled_mask, axis=-1)
        neighbor_feats = neighbor_feats * tf.cast(tiled_mask, neighbor_feats.dtype)
        return neighbor_feats

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'neighbor_indices': self.neighbor_indices.numpy().tolist(),
                'use_3d_conv': self.use_3d_conv,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class IndexedConvolutionLayer(tf.keras.layers.Layer):
    """
    Applies a depthwise convolution over the neighbor dimension per pixel.

    Wraps a ``Conv2D`` (when ``use_3d_conv=False``) or ``Conv3D``
    (when ``use_3d_conv=True``) with ``kernel_size=(1, 7)`` /
    ``(1, 7, temporal_kernel_size)``, ``padding='valid'``,
    ``activation='relu'``.

    Input shape:

        - ``use_3d_conv=False``: ``[batch, n_pixels, n_neighbors, n_features_in]``
        - ``use_3d_conv=True``:  ``[batch, n_pixels, n_neighbors, n_time, n_features_in]``

    Output shape (after squeezing the neighbor dimension):

        - ``use_3d_conv=False``: ``[batch, n_pixels, filters]``
        - ``use_3d_conv=True``:  ``[batch, n_pixels, n_time, filters]``

    Parameters
    ----------
    use_3d_conv : bool
        Select Conv3D (``True``) or Conv2D (``False``).
    temporal_kernel_size : int
        Kernel size along the time axis (only used when ``use_3d_conv=True``).
    filters : int
        Number of output filters.
    name : str
        Layer name.
    """

    def __init__(self, use_3d_conv, temporal_kernel_size, filters, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.use_3d_conv = use_3d_conv
        self.temporal_kernel_size = temporal_kernel_size
        self.filters = filters
        if use_3d_conv:
            self.conv = tf.keras.layers.Conv3D(
                filters=filters,
                kernel_size=(1, 7, temporal_kernel_size),
                padding='valid',
                activation='relu',
                name=f'{name}_internal_conv3d',
            )
        else:
            self.conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(1, 7),
                padding='valid',
                activation='relu',
                name=f'{name}_internal_conv2d',
            )

    def call(self, x):
        x = self.conv(x)
        return tf.squeeze(x, axis=2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'use_3d_conv': self.use_3d_conv,
                'temporal_kernel_size': self.temporal_kernel_size,
                'filters': self.filters,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class IndexedPoolingLayer(tf.keras.layers.Layer):
    """
    Applies max or average pooling over the neighbor dimension per pixel.

    Wraps a ``MaxPool2D`` / ``AveragePooling2D`` (when ``use_3d_conv=False``)
    or ``MaxPool3D`` / ``AveragePooling3D`` (when ``use_3d_conv=True``) with
    ``pool_size=(1, 7)`` / ``(1, 7, temporal_pool_size)``,
    ``padding='valid'``.

    Input shape:

        - ``use_3d_conv=False``: ``[batch, n_pixels, n_neighbors, n_features]``
        - ``use_3d_conv=True``:  ``[batch, n_pixels, n_neighbors, n_time, n_features]``

    Output shape (after squeezing the neighbor dimension):

        - ``use_3d_conv=False``: ``[batch, n_pixels, n_features]``
        - ``use_3d_conv=True``:  ``[batch, n_pixels, n_time, n_features]``

    Parameters
    ----------
    use_3d_conv : bool
        Select 3-D pooling (``True``) or 2-D pooling (``False``).
    pooling_type : str
        ``'max'`` for max-pooling or ``'average'`` for average-pooling.
    temporal_pool_size : int
        Pool size along the time axis (only used when ``use_3d_conv=True``).
    name : str
        Layer name.
    """

    def __init__(self, use_3d_conv, pooling_type, temporal_pool_size, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.use_3d_conv = use_3d_conv
        self.pooling_type = pooling_type.lower()
        self.temporal_pool_size = temporal_pool_size
        pool_layer_name = f'{name}_internal_pool'
        if self.use_3d_conv:
            PoolLayer = tf.keras.layers.MaxPool3D if self.pooling_type == 'max' else tf.keras.layers.AveragePooling3D
            self.pool = PoolLayer(
                pool_size=(1, 7, self.temporal_pool_size),
                padding='valid',
                name=pool_layer_name,
            )
        else:
            PoolLayer = tf.keras.layers.MaxPool2D if self.pooling_type == 'max' else tf.keras.layers.AveragePooling2D
            self.pool = PoolLayer(
                pool_size=(1, 7),
                padding='valid',
                name=pool_layer_name,
            )

    def call(self, x):
        x = self.pool(x)
        return tf.squeeze(x, axis=2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'use_3d_conv': self.use_3d_conv,
                'pooling_type': self.pooling_type,
                'temporal_pool_size': self.temporal_pool_size,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
