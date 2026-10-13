"""Parameters exposed outside the HLS compute IP, and how their values map onto
the interface HLS builds for them.

An ``ExternalParameter`` records what the writer *asked* HLS for: the interface
kind and geometry it expects, and the structural rule (``FlatOrder`` + layout)
that turns a tensor into interface words. Synthesis is the only source of what
was built; consumers cross-check against it before relying on any ``expected_*``
field.

The written project carries these as ``firmware/weights/external_parameters.json``.
"""

import importlib
import json
import os

import numpy as np

from hls4ml.model.types import FixedPrecisionType, PrecisionType, Serializable

SCHEMA = 'hls4ml.external_parameter_manifest/v1'
SCHEMA_VERSION = 1
MANIFEST_FILENAME = 'external_parameters.json'


class ExternalParameterError(Exception):
    """The parameter description does not support the requested operation."""


class FlatOrder(Serializable):
    """Tensor -> flat scalar sequence, as an axis permutation then a ravel.

    ``tensor_axes`` name the tensor's own axes in order, ``shape`` is the shape it
    must have, ``axes`` is the enumeration order.
    """

    def __init__(self, tensor_axes, axes, shape):
        self.tensor_axes = list(tensor_axes)
        self.axes = list(axes)
        self.shape = [int(d) for d in shape]
        if sorted(self.tensor_axes) != sorted(self.axes) or len(set(self.axes)) != len(self.axes):
            raise ExternalParameterError(f'axes {self.axes} are not a permutation of {self.tensor_axes}')
        if len(self.shape) != len(self.tensor_axes) or any(d <= 0 for d in self.shape):
            raise ExternalParameterError(f'shape {self.shape} does not match axes {self.tensor_axes}')

    @property
    def size(self):
        return int(np.prod(self.shape))

    def flatten(self, tensor):
        array = np.asarray(tensor)
        if array.shape != tuple(self.shape):
            raise ExternalParameterError(f'tensor shape {array.shape} but {tuple(self.shape)} declared')
        permutation = [self.tensor_axes.index(axis) for axis in self.axes]
        return np.transpose(array, permutation).ravel().tolist()

    def serialize_state(self):
        return {'tensor_axes': self.tensor_axes, 'axes': self.axes, 'shape': self.shape}


class BlockLayout(Serializable):
    """Words of a reshaped memory: scalar ``f`` lands in word ``f % block_size``
    at lane ``f // block_size``."""

    mode = 'block'

    def __init__(self, block_size, lanes):
        self.block_size = int(block_size)
        self.lanes = int(lanes)
        if self.block_size <= 0 or self.lanes <= 0:
            raise ExternalParameterError(f'block layout needs positive block_size and lanes, got {block_size}, {lanes}')

    @property
    def capacity(self):
        return self.block_size * self.lanes

    @property
    def depth(self):
        return self.block_size

    def pack(self, codes, width):
        words = [0] * self.block_size
        for f, code in enumerate(codes):
            words[f % self.block_size] |= code << (width * (f // self.block_size))
        return words

    def serialize_state(self):
        return {'mode': self.mode, 'block_size': self.block_size, 'lanes': self.lanes}


class CompleteLayout(Serializable):
    """A fully partitioned parameter: one port per scalar, so the "words" are the
    scalars themselves."""

    mode = 'complete'
    depth = None

    def pack(self, codes, width):
        return list(codes)

    def serialize_state(self):
        return {'mode': self.mode}


_LAYOUTS = {BlockLayout.mode: BlockLayout, CompleteLayout.mode: CompleteLayout}


def _layout_from_state(state):
    if state is None:
        return None
    state = dict(state)
    mode = state.pop('mode', None)
    if mode not in _LAYOUTS:
        raise ExternalParameterError(f'unknown layout mode {mode!r}')
    try:
        return _LAYOUTS[mode](**state)
    except TypeError as exc:
        raise ExternalParameterError(f'malformed {mode} layout: {exc}')


class ExternalParameter(Serializable):
    """One parameter exposed on the compute IP's interface.

    Identity (``name``, ``layer``, ``role``) and precision are always present.
    ``interface_kind``, ``data_width``, ``depth``, ``flat_order`` and ``layout``
    are present only when the writer could describe the interface; otherwise
    ``note`` says why and the parameter cannot be packed.
    """

    KINDS = ('bram', 'scalar_bundle')

    def __init__(
        self,
        name,
        layer,
        layer_class,
        role,
        tensor_shape,
        n_scalars,
        precision,
        reuse_factor=None,
        strategy=None,
        kernel_variant=None,
        pragma=None,
        interface_kind=None,
        data_width=None,
        depth=None,
        flat_order=None,
        layout=None,
        note=None,
    ):
        self.name = name
        self.layer = layer
        self.layer_class = layer_class
        self.role = role
        self.tensor_shape = [int(d) for d in tensor_shape]
        self.n_scalars = int(n_scalars)
        self.precision = precision
        self.reuse_factor = reuse_factor
        self.strategy = strategy
        self.kernel_variant = kernel_variant
        self.pragma = pragma
        self.interface_kind = interface_kind
        self.data_width = data_width
        self.depth = depth
        self.flat_order = flat_order
        self.layout = layout
        self.note = note
        self._validate()

    def _validate(self):
        """An undescribed parameter carries identity only; a described one must be
        internally consistent, so nothing malformed survives to packing or RTL."""
        name = self.name
        if not name or not self.layer or not self.role:
            raise ExternalParameterError(f'{name!r}: name, layer and role are required')
        if not isinstance(self.precision, PrecisionType):
            raise ExternalParameterError(f'{name}: precision must be a PrecisionType, got {type(self.precision).__name__}')
        if self.n_scalars <= 0:
            raise ExternalParameterError(f'{name}: n_scalars must be positive, got {self.n_scalars}')
        if any(d <= 0 for d in self.tensor_shape) or int(np.prod(self.tensor_shape)) != self.n_scalars:
            raise ExternalParameterError(f'{name}: tensor_shape {self.tensor_shape} does not hold {self.n_scalars} scalars')
        if self.interface_kind is None:
            if any(v is not None for v in (self.flat_order, self.layout, self.data_width, self.depth)):
                raise ExternalParameterError(f'{name}: geometry is claimed without an interface kind')
            return
        if self.interface_kind not in self.KINDS:
            raise ExternalParameterError(f'{name}: unknown interface kind {self.interface_kind!r}')
        if self.flat_order is None or self.layout is None:
            raise ExternalParameterError(
                f'{name}: interface kind {self.interface_kind!r} claimed without flat_order and layout'
            )
        if self.flat_order.size != self.n_scalars:
            raise ExternalParameterError(
                f'{name}: shape {self.flat_order.shape} holds {self.flat_order.size} scalars, not {self.n_scalars}'
            )
        width = self.precision.width
        if self.interface_kind == 'bram':
            if not isinstance(self.layout, BlockLayout):
                raise ExternalParameterError(f'{name}: a bram interface needs a block layout, not {self.layout.mode!r}')
            if self.depth != self.layout.block_size or self.data_width != self.layout.lanes * width:
                raise ExternalParameterError(
                    f'{name}: geometry {self.data_width}x{self.depth} disagrees with layout '
                    f'{self.layout.lanes} lanes x {self.layout.block_size} words of {width}-bit scalars'
                )
            if not self.layout.capacity - self.layout.lanes < self.n_scalars <= self.layout.capacity:
                raise ExternalParameterError(
                    f'{name}: {self.n_scalars} scalars do not fill a {self.layout.block_size}-word block layout'
                )
        else:
            if not isinstance(self.layout, CompleteLayout):
                raise ExternalParameterError(f'{name}: a scalar bundle needs a complete layout, not {self.layout.mode!r}')
            if self.depth is not None or self.data_width != width:
                raise ExternalParameterError(f'{name}: a scalar bundle has no depth and is {width} bits wide')

    @property
    def key(self):
        return (self.layer, self.role)

    @property
    def described(self):
        """Whether the writer claimed an interface and a packing rule."""
        return self.interface_kind is not None and self.flat_order is not None and self.layout is not None

    def require_described(self):
        if not self.described:
            raise ExternalParameterError(f'{self.name}: {self.note or "no interface or layout is claimed"}')

    # --- values -> interface words -------------------------------------------

    def quantize(self, values):
        """Reals -> raw two's-complement codes of this parameter's precision."""
        reason = unencodable_reason(self.precision)
        if reason:
            raise ExternalParameterError(f'{self.name}: {reason}')
        return [quantize_fixed(v, self.precision.width, self.precision.integer) for v in values]

    def pack_flat(self, values):
        """Flat reals -> one int per word (per scalar port for a complete layout)."""
        self.require_described()
        values = list(values)
        if len(values) != self.n_scalars:
            raise ExternalParameterError(f'{self.name}: got {len(values)} scalars, {self.n_scalars} declared')
        return self.layout.pack(self.quantize(values), self.precision.width)

    def pack(self, tensor):
        """Tensor in the declared shape -> interface words."""
        self.require_described()
        return self.pack_flat(self.flat_order.flatten(tensor))

    # --- serialization -------------------------------------------------------

    def serialize_state(self):
        return {
            'name': self.name,
            'layer': self.layer,
            'layer_class': self.layer_class,
            'role': self.role,
            'tensor_shape': self.tensor_shape,
            'n_scalars': self.n_scalars,
            'precision': self.precision.serialize(),
            'reuse_factor': self.reuse_factor,
            'strategy': self.strategy,
            'kernel_variant': self.kernel_variant,
            'pragma': self.pragma,
            'expected_interface_kind': self.interface_kind,
            'expected_data_width': self.data_width,
            'expected_depth': self.depth,
            'flat_order': self.flat_order.serialize_state() if self.flat_order else None,
            'layout': self.layout.serialize_state() if self.layout else None,
            'note': self.note,
        }

    @classmethod
    def deserialize(cls, state):
        state = dict(state)
        state['precision'] = _deserialize_precision(state.pop('precision'))
        state['interface_kind'] = state.pop('expected_interface_kind')
        state['data_width'] = state.pop('expected_data_width')
        state['depth'] = state.pop('expected_depth')
        state['flat_order'] = FlatOrder(**state['flat_order']) if state['flat_order'] else None
        state['layout'] = _layout_from_state(state['layout'])
        return cls(**state)


def _deserialize_precision(serialized):
    """Rebuild whichever PrecisionType was written; ``class_name`` names it, as
    ``Serializable`` intends. A precision this module cannot pack still round-trips
    and is refused at ``quantize()``."""
    try:
        module_name, class_name = serialized['class_name'].rsplit('.', 1)
        cls = getattr(importlib.import_module(module_name), class_name, None)
        if not (isinstance(cls, type) and issubclass(cls, PrecisionType)):
            raise TypeError(f'{serialized["class_name"]!r} is not a PrecisionType')
        return cls.deserialize(serialized['state'])
    except (KeyError, TypeError, ValueError, AttributeError, ImportError) as exc:
        raise ExternalParameterError(f'cannot deserialize precision {serialized!r}: {exc}')


def unencodable_reason(precision):
    """Why ``quantize_fixed`` cannot encode this precision, or None if it can.

    Only ap_fixed with AP_TRN / AP_WRAP and no saturation bits is implemented: a
    nonzero N changes AP_WRAP from plain modulo wrap to sign-preserving wrap.
    """
    if not isinstance(precision, FixedPrecisionType):
        return f'precision {precision} is not a fixed-point type'
    rounding, saturation = str(precision.rounding_mode), str(precision.saturation_mode)
    if rounding != 'TRN' or saturation != 'WRAP':
        return f'precision {precision} uses rounding={rounding}/saturation={saturation}; only TRN/WRAP can be encoded'
    if precision.saturation_bits:
        return f'precision {precision} uses {precision.saturation_bits} saturation bits; only 0 can be encoded'
    return None


def quantize_fixed(value, width, integer):
    """ap_fixed<width,integer> with AP_TRN / AP_WRAP and no saturation bits."""
    from fractions import Fraction

    scaled = Fraction(value).limit_denominator(1 << 30) * (1 << (width - integer))
    code = scaled.numerator // scaled.denominator  # AP_TRN: floor toward -inf
    return code % (1 << width)  # AP_WRAP


class ExternalParameterManifest(Serializable):
    """What a written project exposes: its external parameters plus the project
    facts a consumer needs to find and check the synthesized interface."""

    def __init__(self, project_name, backend, io_type, part, clock_period, parameters, hls4ml_version=None):
        self.project_name = project_name
        self.backend = backend
        self.io_type = io_type
        self.part = part
        self.clock_period = clock_period
        self.parameters = list(parameters)
        self.hls4ml_version = hls4ml_version
        names = [p.name for p in self.parameters]
        keys = [p.key for p in self.parameters]
        if len(set(names)) != len(names) or len(set(keys)) != len(keys):
            raise ExternalParameterError(f'manifest has duplicate parameters: names {names}, keys {keys}')

    def __iter__(self):
        return iter(self.parameters)

    def get(self, layer, role):
        return next((p for p in self.parameters if p.key == (layer, role)), None)

    @property
    def described(self):
        return [p for p in self.parameters if p.described]

    @property
    def undescribed(self):
        return [p for p in self.parameters if not p.described]

    def serialize_state(self):
        return {
            'schema': SCHEMA,
            'schema_version': SCHEMA_VERSION,
            'hls4ml_version': self.hls4ml_version,
            'project_name': self.project_name,
            'backend': self.backend,
            'io_type': self.io_type,
            'part': self.part,
            'clock_period': self.clock_period,
            'ports': [p.serialize_state() for p in self.parameters],
        }

    @classmethod
    def deserialize(cls, state):
        if state.get('schema') != SCHEMA:
            raise ExternalParameterError(f'manifest schema is {state.get("schema")!r}, expected {SCHEMA!r}')
        if state.get('schema_version') != SCHEMA_VERSION:
            raise ExternalParameterError(f'manifest schema version {state.get("schema_version")!r} is not {SCHEMA_VERSION}')
        for key in ('project_name', 'backend', 'part', 'clock_period'):
            if not state.get(key):
                raise ExternalParameterError(f'manifest has no {key}')
        return cls(
            state['project_name'],
            state['backend'],
            state.get('io_type'),
            state['part'],
            state['clock_period'],
            [ExternalParameter.deserialize(p) for p in state['ports']],
            state.get('hls4ml_version'),
        )

    @staticmethod
    def path_in(project_dir):
        return os.path.join(project_dir, 'firmware', 'weights', MANIFEST_FILENAME)

    def save(self, project_dir):
        path = self.path_in(project_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(self.serialize_state(), fh, indent=2)
        return path

    @classmethod
    def load(cls, project_dir):
        path = cls.path_in(project_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(f'no manifest at {path}: the written project exposes no external parameters')
        with open(path) as fh:
            return cls.deserialize(json.load(fh))
