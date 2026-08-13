from hls4ml.model.layers import (
    Reshape,
    SparseActivation,
    SparseConv2D,
    SparseInputReduce,
    SparsePooling2D,
)
from hls4ml.model.optimizer import OptimizerPass

_SPARSE_TYPES = (SparseInputReduce, SparseConv2D, SparseActivation, SparsePooling2D)


class ConvertSparseFlatten(OptimizerPass):
    """Convert a Reshape (keras Flatten) fed by sparsepixels layers into SparseFlatten.

    The sparse layers carry the image as fixed-length feature/coordinate arrays, so unlike a
    dense-model flatten (a free reshape), this one is a real scatter back to a dense vector and
    needs its own kernel. The spatial size at the flatten is reconstructed by walking the sparse
    chain up to the input reduction and applying the pooling factors along the way.
    """

    def match(self, node):
        return isinstance(node, Reshape) and isinstance(node.get_input_node(), _SPARSE_TYPES)

    def transform(self, model, node):
        inp = node.get_input_node()
        n_sparse = inp.get_attr('n_sparse')
        n_chan = inp.get_attr('n_filt') if isinstance(inp, SparseConv2D) else inp.get_attr('n_chan')

        # Walk up to the input reduction, collecting pooling factors. Sparse convs are
        # same-padding stride-1, so only pooling changes the spatial size. Intermediate
        # non-sparse nodes (e.g. FixedPointQuantizer from the input quantizers) are stepped over.
        pool_sizes = []
        cur = inp
        while cur is not None and not isinstance(cur, SparseInputReduce):
            if isinstance(cur, SparsePooling2D):
                pool_sizes.append((cur.get_attr('pool_height'), cur.get_attr('pool_width')))
            cur = cur.get_input_node()
        if cur is None:
            return False
        height, width = cur.get_attr('in_height'), cur.get_attr('in_width')
        for pool_h, pool_w in pool_sizes:
            height, width = height // pool_h, width // pool_w

        attrs = {
            'n_sparse': n_sparse,
            'n_chan': n_chan,
            'out_height': height,
            'out_width': width,
        }
        new_node = model.make_node('SparseFlatten', node.name, attrs, node.inputs.copy())
        model.replace_node(node, new_node)
        return True
