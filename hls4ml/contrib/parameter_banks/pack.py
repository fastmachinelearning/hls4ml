"""Pack trained parameter sets into bank images.

Each ``ExternalParameter`` knows how to turn its own tensor into interface words;
this module stacks those words per bank and checks that a set of parameter sets
describes hardware the synthesized IP can actually hold.
"""

import numpy as np

from hls4ml.model.external_parameters import ExternalParameterError, ExternalParameterManifest


class PackingUnsupported(Exception):
    """The parameter sets cannot be packed for this model."""


class BankImage:
    """The packed contents of one external parameter across every bank."""

    def __init__(self, parameter, per_bank, bank_stride_words=None):
        self.parameter = parameter
        self.per_bank = per_bank  # bank -> words (BRAM) or scalar codes (bundle)
        self.bank_stride_words = bank_stride_words

    @property
    def name(self):
        return self.parameter.name

    @property
    def kind(self):
        return self.parameter.interface_kind

    @property
    def data_width(self):
        return self.parameter.data_width

    @property
    def image(self):
        """BRAM only: the depth-stacked ``$readmemh`` image, banks padded to the stride."""
        if self.kind != 'bram':
            raise PackingUnsupported(f'{self.name}: a {self.kind} port has no memory image; load it through the loader')
        image = []
        for words in self.per_bank:
            image.extend(words)
            image.extend([0] * (self.bank_stride_words - len(words)))  # padding words are never addressed
        return image

    def write_mem(self, path):
        """Write the image as a ``$readmemh`` file: one word per line, zero-padded
        to the full logical width."""
        digits = (self.data_width + 3) // 4
        with open(path, 'w') as fh:
            for word in self.image:
                fh.write(f'{word:0{digits}x}\n')
        return path


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def pack_banks(model, banks):
    """Pack one complete parameter set per bank. Returns ``{port_name: BankImage}``.

    ``banks`` is a list of ``{(layer, role): tensor}``, one per bank, covering
    *every* parameter of the model. Only parameters exposed on the interface may
    differ between banks; the rest are compiled into the IP, so banks that disagree
    on one describe hardware that cannot be built and are rejected.

    ``model`` is the written ModelGraph the IP was built from -- normally bank 0.
    That the *fixed* parameters supplied here match the ones compiled into the IP
    is the caller's responsibility.
    """
    if not hasattr(model, 'get_layers'):
        raise PackingUnsupported(f'pack_banks needs a ModelGraph, got {type(model).__name__}')
    if len(banks) < 2:
        raise PackingUnsupported(f'need at least 2 banks, got {len(banks)}')

    try:
        manifest = ExternalParameterManifest.load(model.config.get_output_dir())
    except FileNotFoundError as exc:
        raise PackingUnsupported(str(exc))
    inventory = {(layer.name, role) for layer in model.get_layers() for role in getattr(layer, 'weights', {})}
    external = {p.key: p for p in manifest}
    foreign = sorted(set(external) - inventory)
    if foreign:
        raise PackingUnsupported(f'manifest parameters {foreign} are not parameters of this model; wrong ModelGraph?')

    # Each bank has to be complete on its own: a union would let one bank omit a
    # fixed parameter that another supplies, indistinguishable from agreement.
    for i, bank in enumerate(banks):
        missing = inventory - set(bank)
        if missing:
            raise PackingUnsupported(f'bank {i} is not a complete parameter set; missing {sorted(missing)}')
        unknown = set(bank) - inventory
        if unknown:
            raise PackingUnsupported(f'bank {i} has entries that are not parameters of this model: {sorted(unknown)}')

    differing = [
        key
        for key in sorted(inventory - set(external), key=str)
        if any(not np.array_equal(np.asarray(banks[0][key]), np.asarray(bank[key])) for bank in banks[1:])
    ]
    if differing:
        raise PackingUnsupported(
            f'{differing} are compiled into the compute IP, but the supplied banks disagree on them; '
            'every bank must share the same fixed parameters'
        )

    images = {}
    for key, parameter in external.items():
        try:
            per_bank = [parameter.pack(bank[key]) for bank in banks]
        except ExternalParameterError as exc:
            raise PackingUnsupported(str(exc))
        stride = _next_pow2(parameter.depth) if parameter.interface_kind == 'bram' else None
        images[parameter.name] = BankImage(parameter, per_bank, stride)
    return images
