from collections import namedtuple

import pytest

from hls4ml.model.optimizer.passes.infer_precision import InferPrecisionTypes
from hls4ml.model.types import (
    FixedPrecisionType,
    IntegerPrecisionType,
    NamedType,
    RoundingMode,
    SaturationMode,
    UnspecifiedPrecisionType,
)


class MockBackend:
    def convert_precision_string(self, precision_string):
        """
        Simple mock that expects a FixedPrecisionType object or None
        to be passed directly for testing purposes, or a simple string parser.
        """
        if isinstance(precision_string, (FixedPrecisionType, IntegerPrecisionType)):
            return precision_string
        return None


class MockConfig:
    def __init__(self, max_precision=None, default_precision=None):
        self.model_precision = {}
        if max_precision:
            self.model_precision['maximum'] = max_precision
        if default_precision:
            self.model_precision['default'] = default_precision

        self.backend = MockBackend()


class MockModel:
    def __init__(self, max_precision=None):
        default = FixedPrecisionType(width=16, integer=6)
        self.config = MockConfig(max_precision, default)


class MockVariable:
    def __init__(self, precision):
        self.type = namedtuple('Type', ['precision'])(precision)
        self.shape = [10, 10]


class MockWeight:
    def __init__(self, precision):
        self.precision = precision
        self.nonzeros = 10

    def update_precision(self, new_precision):
        self.precision = new_precision


class MockNode:
    def __init__(self, class_name, name='test_node', max_precision=None, inputs=None):
        self.class_name = class_name
        self.name = name
        self.model = MockModel(max_precision)
        self.attributes = {
            'n_in': 10,
            'n_out': 10,
            'n_chan': 3,
            'filt_height': 3,
            'filt_width': 3,
            'pool_height': 2,
            'pool_width': 2,
            'op': 'multiply',  # Default for merge tests
            'pool_op': 'average',
        }
        self.types = {
            'result_t': NamedType('result_t', UnspecifiedPrecisionType()),
            'accum_t': NamedType('accum_t', UnspecifiedPrecisionType()),
            'weight_t': NamedType('weight_t', FixedPrecisionType(8, 4)),
            'bias_t': NamedType('bias_t', FixedPrecisionType(8, 4)),
            'scale_t': NamedType('scale_t', FixedPrecisionType(8, 4)),
            'pointwise_t': NamedType('pointwise_t', FixedPrecisionType(8, 4)),
        }
        self.weights = {
            'weight': MockWeight(FixedPrecisionType(8, 4)),
            'bias': MockWeight(FixedPrecisionType(8, 4)),
            'scale': MockWeight(FixedPrecisionType(8, 4)),
            'pointwise': MockWeight(FixedPrecisionType(8, 4)),
        }

        # Setup inputs
        self.inputs = inputs if inputs else ['input_1']
        self._input_vars = {'input_1': MockVariable(FixedPrecisionType(16, 6))}
        if len(self.inputs) > 1:
            self._input_vars['input_2'] = MockVariable(FixedPrecisionType(16, 6))

    def get_attr(self, key, default=None):
        return self.attributes.get(key, default)

    def get_input_variable(self, input_name=None):
        if input_name is None:
            return self._input_vars[self.inputs[0]]
        return self._input_vars.get(input_name)

    def get_output_variable(self):
        return MockVariable(UnspecifiedPrecisionType())


@pytest.fixture
def optimizer():
    return InferPrecisionTypes()


class TestApplyMaxPrecisionConstraints:
    """
    Tests the logic of _apply_max_precision_constraints function directly.
    """

    def test_no_max_precision_set(self, optimizer):
        """If 'maximum' is not in config, return precision unchanged."""
        node = MockNode('Dense', max_precision=None)

        input_prec = FixedPrecisionType(width=20, integer=10)
        result = optimizer._apply_max_precision_constraints(node, input_prec)

        assert result.width == 20
        assert result.integer == 10

    def test_clamp_width(self, optimizer):
        """Should reduce width if input > max."""
        max_prec = FixedPrecisionType(width=16, integer=10)
        node = MockNode('Dense', max_precision=max_prec)

        input_prec = FixedPrecisionType(width=32, integer=10)
        result = optimizer._apply_max_precision_constraints(node, input_prec)

        assert result.width == 16
        assert result.integer == 10

    def test_clamp_integer(self, optimizer):
        """Should reduce integer bits if input > max."""
        max_prec = FixedPrecisionType(width=32, integer=5)
        node = MockNode('Dense', max_precision=max_prec)

        input_prec = FixedPrecisionType(width=32, integer=10)
        result = optimizer._apply_max_precision_constraints(node, input_prec)

        assert result.width == 32
        assert result.integer == 5

    def test_signedness_inheritance(self, optimizer):
        """Should always adopt the signedness of the maximum precision."""
        # Max is Unsigned (signed=0)
        max_prec = FixedPrecisionType(width=32, integer=10, signed=0)
        node = MockNode('Dense', max_precision=max_prec)

        # Input is Signed
        input_prec = FixedPrecisionType(width=32, integer=10, signed=1)
        result = optimizer._apply_max_precision_constraints(node, input_prec)

        assert result.signed == 0

    def test_mode_inheritance_from_max(self, optimizer):
        """If Max specifies rounding/sat modes, they should override input."""
        max_prec = FixedPrecisionType(
            16, 6, rounding_mode=RoundingMode.RND, saturation_mode=SaturationMode.SAT, saturation_bits=2
        )
        node = MockNode('Dense', max_precision=max_prec)

        # Input has different modes
        input_prec = FixedPrecisionType(16, 6, rounding_mode=RoundingMode.TRN, saturation_mode=SaturationMode.WRAP)

        result = optimizer._apply_max_precision_constraints(node, input_prec)

        assert result.rounding_mode == RoundingMode.RND
        assert result.saturation_mode == SaturationMode.SAT
        assert result.saturation_bits == 2

    def test_mode_preservation_when_max_is_none(self, optimizer):
        """If Max modes are default, input modes should be preserved."""
        # Create a max precision where modes are initialized with defaults
        max_prec = FixedPrecisionType(16, 6)

        node = MockNode('Dense', max_precision=max_prec)

        input_prec = FixedPrecisionType(16, 6, rounding_mode=RoundingMode.RND_ZERO, saturation_mode=SaturationMode.SAT_SYM)

        result = optimizer._apply_max_precision_constraints(node, input_prec)

        assert result.rounding_mode == RoundingMode.RND_ZERO
        assert result.saturation_mode == SaturationMode.SAT_SYM


class TestInferPrecision:
    """
    Tests that _infer_precision calls apply_max_constraints for specific layers.
    We verify this by setting a strict Max constraint and asserting the result_t
    complies with it.
    """

    # Define a strict constraint
    STRICT_MAX = FixedPrecisionType(width=4, integer=2, signed=True)

    @pytest.mark.parametrize(
        'layer_class',
        [
            'Dense',
            'Conv1D',
            'Conv2D',
            'PointwiseConv2D',
            'DepthwiseConv2D',
        ],
    )
    def test_common_precision_layers(self, optimizer, layer_class):
        """Tests layers that use _infer_common_precision."""
        node = MockNode(layer_class, max_precision=self.STRICT_MAX)

        node._input_vars['input_1'] = MockVariable(FixedPrecisionType(32, 16, signed=1))

        types_to_infer = ['result_t', 'accum_t']
        optimizer._infer_precision(node, types_to_infer)

        res_prec = node.types['result_t'].precision
        assert res_prec.width == 4
        assert res_prec.integer == 2

    def test_batch_normalization(self, optimizer):
        """Tests BN layer inference."""
        node = MockNode('BatchNormalization', max_precision=self.STRICT_MAX)
        node._input_vars['input_1'] = MockVariable(FixedPrecisionType(32, 16))

        types_to_infer = ['result_t']
        optimizer._infer_precision(node, types_to_infer)

        res_prec = node.types['result_t'].precision
        assert res_prec.width == 4
        assert res_prec.integer == 2

    def test_merge_multiply(self, optimizer):
        """Tests Merge layer with Multiply op."""
        node = MockNode('Merge', max_precision=self.STRICT_MAX, inputs=['input_1', 'input_2'])
        node.attributes['op'] = 'multiply'

        node._input_vars['input_1'] = MockVariable(FixedPrecisionType(20, 10))
        node._input_vars['input_2'] = MockVariable(FixedPrecisionType(20, 10))

        types_to_infer = ['result_t']
        optimizer._infer_precision(node, types_to_infer)

        res_prec = node.types['result_t'].precision
        assert res_prec.width == 4
        assert res_prec.integer == 2

    def test_merge_add(self, optimizer):
        """Tests Merge layer with Add op."""
        node = MockNode('Merge', max_precision=self.STRICT_MAX, inputs=['input_1', 'input_2'])
        node.attributes['op'] = 'add'

        node._input_vars['input_1'] = MockVariable(FixedPrecisionType(20, 10))
        node._input_vars['input_2'] = MockVariable(FixedPrecisionType(20, 10))

        types_to_infer = ['result_t']
        optimizer._infer_precision(node, types_to_infer)

        res_prec = node.types['result_t'].precision
        assert res_prec.width == 4
        assert res_prec.integer == 2

    def test_concatenate_same_input_precisions(self, optimizer):
        """
        Tests Concatenate layer. If precisions of both inputs are the same,
        max precision is ignored (see _infer_cat_precision function).
        """
        node = MockNode('Concatenate', max_precision=self.STRICT_MAX, inputs=['input_1', 'input_2'])

        node._input_vars['input_1'] = MockVariable(FixedPrecisionType(20, 10))
        node._input_vars['input_2'] = MockVariable(FixedPrecisionType(20, 10))

        types_to_infer = ['result_t']
        optimizer._infer_precision(node, types_to_infer)

        res_prec = node.types['result_t'].precision
        assert res_prec.width == 20
        assert res_prec.integer == 10

    def test_concatenate_different_input_precisions(self, optimizer):
        """Tests Concatenate layer."""
        node = MockNode('Concatenate', max_precision=self.STRICT_MAX, inputs=['input_1', 'input_2'])

        node._input_vars['input_1'] = MockVariable(FixedPrecisionType(20, 10))
        node._input_vars['input_2'] = MockVariable(FixedPrecisionType(16, 6))

        types_to_infer = ['result_t']
        optimizer._infer_precision(node, types_to_infer)

        res_prec = node.types['result_t'].precision
        assert res_prec.width == 4
        assert res_prec.integer == 2

    def test_dot(self, optimizer):
        """Tests Dot layer."""
        node = MockNode('Dot', max_precision=self.STRICT_MAX, inputs=['input_1', 'input_2'])

        node._input_vars['input_1'] = MockVariable(FixedPrecisionType(20, 10))
        node._input_vars['input_2'] = MockVariable(FixedPrecisionType(20, 10))

        types_to_infer = ['result_t']
        optimizer._infer_precision(node, types_to_infer)

        res_prec = node.types['result_t'].precision
        assert res_prec.width == 4
        assert res_prec.integer == 2
