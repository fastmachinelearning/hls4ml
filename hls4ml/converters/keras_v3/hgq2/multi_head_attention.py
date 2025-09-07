import typing
from collections.abc import Sequence
from inspect import Signature

import numpy as np

from ._base import QEinsumDenseHandler, QLayerHandler, register
from .einsum import QEinsumHandler
from .softmax import QSoftmaxHandler

if typing.TYPE_CHECKING:
    import hgq
    from keras import KerasTensor


@register
class QMultiHeadAttentionHandler(QLayerHandler):
    handles = ('hgq.layers.multi_head_attention.QMultiHeadAttention',)

    def handle(
        self,
        layer: 'hgq.layers.QMultiHeadAttention',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        # fmt: off
        assert len(in_tensors) in (2, 3, 4,), (
            'MultiHead layer must have 2 (Q, V), 3 (Q, V, K) or 4 (Q, V, K, M) input tensors'
        )
        # fmt: on
        assert len(out_tensors) == 1, 'Attention score output is not supported yet'
        assert len(in_tensors) <= 3, 'Mask tensor is not supported yet'
        tensor_q, *_ = in_tensors
        tensor_O, *tensor_attn = out_tensors

        node_index: int = tensor_q._keras_history.node_index  # type: ignore
        assert all(
            [node_index == inp._keras_history.node_index for inp in layer.input[1:]]
        ), f'Critical error handling layer {layer.name}'
        node = layer._inbound_nodes[node_index]

        args = node.arguments.args
        kwargs = node.arguments.kwargs
        sig: Signature = layer._call_signature

        # map everything to kwargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        tensor_q = bound.arguments['query']
        tensor_k = bound.arguments['key']
        tensor_v = bound.arguments['value']
        if tensor_k is None:
            tensor_k = tensor_v
        tensor_q_mask = bound.arguments['query_mask']
        tensor_k_mask = bound.arguments['key_mask']
        tensor_v_mask = bound.arguments['value_mask']
        tensor_attn_mask = bound.arguments['attention_mask']
        return_scores = bound.arguments['return_attention_scores']  # noqa: F841

        n_mask_def = np.sum(
            [
                tensor_q_mask is not None,
                tensor_k_mask is not None,
                tensor_v_mask is not None,
                tensor_attn_mask is not None,
            ]
        )
        assert n_mask_def <= 1, f'Layer {layer.name} has {n_mask_def} masks defined, expected at most 1'

        return self._handle(layer, tensor_q, tensor_O, node_index, tensor_k, tensor_v)

    def _handle(self, layer, tensor_q, tensor_O, node_index, tensor_k, tensor_v):
        from hgq.layers import QEinsum
        from keras import KerasTensor

        unique_name = f'{layer.name}_{node_index}'
        to_Q = layer.query_dense
        to_K = layer.key_dense
        to_V = layer.value_dense
        to_O = layer.output_dense
        softmax = layer._softmax

        Q_batch_shape = to_Q.full_output_shape
        K_batch_shape = to_K.full_output_shape
        V_batch_shape = to_V.full_output_shape
        # O_batch_shape = to_O.full_output_shape
        n_head = layer.num_heads
        score_batch_shape = (None, n_head, *Q_batch_shape[1:-2], *K_batch_shape[1:-2])

        einsum_QK = QEinsum(layer._dot_product_equation, name=f'{layer.name}_QK', enable_iq=False, enable_oq=False)
        einsum_sV = QEinsum(layer._combine_equation, name=f'{layer.name}_aV', enable_iq=False, enable_oq=False)

        tensor_Q = KerasTensor(name=f'{unique_name}_Q', shape=Q_batch_shape)
        tensor_K = KerasTensor(name=f'{unique_name}_K', shape=K_batch_shape)
        tensor_V = KerasTensor(name=f'{unique_name}_V', shape=V_batch_shape)

        pre_O_shape = (None, *tensor_q.shape[1:-1], layer.num_heads, layer.value_dim)
        tensor_pre_O = KerasTensor(name=f'{unique_name}_pre_O', shape=pre_O_shape)
        # tensor_O = KerasTensor(name=f'{name}_QK', shape=O_batch_shape)
        tensor_pre_score = KerasTensor(name=f'{unique_name}_pre_score', shape=score_batch_shape)
        tensor_score = KerasTensor(name=f'{unique_name}_score', shape=score_batch_shape)

        einsum_handler = QEinsumHandler()
        einsum_dense_handler = QEinsumDenseHandler()
        softmax_handler = QSoftmaxHandler()

        config_to_Q = einsum_dense_handler(to_Q, [tensor_q], [tensor_Q])
        config_to_K = einsum_dense_handler(to_K, [tensor_k], [tensor_K])
        config_to_V = einsum_dense_handler(to_V, [tensor_v], [tensor_V])
        config_einsum_KQ = einsum_handler(einsum_QK, [tensor_K, tensor_Q], [tensor_pre_score])
        config_softmax = softmax_handler(softmax, [tensor_pre_score], [tensor_score])
        config_einsum_sV = einsum_handler(einsum_sV, [tensor_score, tensor_V], [tensor_pre_O])
        config_to_O = einsum_dense_handler(to_O, [tensor_pre_O], [tensor_O])

        configs = (
            *config_to_Q,
            *config_to_K,
            *config_to_V,
            *config_einsum_KQ,
            *config_softmax,
            *config_einsum_sV,
            *config_to_O,
        )
        for conf in configs:
            conf['name'] = f'{layer.name}_{conf["name"]}'
        return configs


@register
class QLinformerAttentionHandler(QMultiHeadAttentionHandler):
    handles = ('hgq.layers.linformer_attention.QLinformerAttention',)

    def handle(
        self,
        layer: 'hgq.layers.linformer_attention.QLinformerAttention',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        from keras import KerasTensor

        # fmt: off
        assert len(in_tensors) in (2, 3, 4,), (
            'MultiHead layer must have 2 (Q, V), 3 (Q, V, K) or 4 (Q, V, K, M) input tensors'
        )
        # fmt: on
        assert len(out_tensors) == 1, 'Attention score output is not supported yet'
        assert len(in_tensors) <= 3, 'Mask tensor is not supported yet'
        tensor_q, *_ = in_tensors
        tensor_O, *tensor_attn = out_tensors
        unique_name: str = layer.name

        node_index: int = tensor_q._keras_history.node_index  # type: ignore
        assert all(
            [node_index == inp._keras_history.node_index for inp in layer.input[1:]]
        ), f'Critical error handling layer {layer.name}'
        node = layer._inbound_nodes[node_index]

        args = node.arguments.args
        kwargs = node.arguments.kwargs
        sig: Signature = layer._call_signature

        # map everything to kwargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        tensor_q = bound.arguments['query']
        tensor_k = bound.arguments['key']
        tensor_v = bound.arguments['value']
        if tensor_k is None:
            tensor_k = tensor_v
        tensor_q_mask = bound.arguments['query_mask']
        tensor_k_mask = bound.arguments['key_mask']
        tensor_v_mask = bound.arguments['value_mask']
        tensor_attn_mask = bound.arguments['attention_mask']
        return_scores = bound.arguments['return_attention_scores']  # noqa: F841

        n_mask_def = np.sum(
            [
                tensor_q_mask is not None,
                tensor_k_mask is not None,
                tensor_v_mask is not None,
                tensor_attn_mask is not None,
            ]
        )
        assert n_mask_def <= 1, f'Layer {layer.name} has {n_mask_def} masks defined, expected at most 1'

        tensor_k_proj = KerasTensor(layer._key_shape_proj, name=f'{unique_name}_k_proj')
        tensor_v_proj = KerasTensor(layer._value_shape_proj, name=f'{unique_name}_v_proj')
        einsum_dense_handler = QEinsumDenseHandler()

        k_proj = einsum_dense_handler(layer._lin_k_proj, [tensor_k], [tensor_k_proj])
        v_proj = einsum_dense_handler(layer._lin_v_proj, [tensor_v], [tensor_v_proj])

        layers = self._handle(layer, tensor_q, tensor_O, node_index, tensor_k_proj, tensor_v_proj)
        return *k_proj, *v_proj, *layers
