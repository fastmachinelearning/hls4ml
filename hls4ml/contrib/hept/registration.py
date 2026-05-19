from __future__ import annotations

import shutil
from pathlib import Path

import hls4ml

from .keras import parse_hept_attention_layer
from .layers import HEPTAttention
from .templates import HEPTConfigTemplate, HEPTFunctionTemplate


_EXTERNAL_HELPER_HEADERS = (
    "nnet_layernorm_stream.h",
    "nnet_qkv_stream.h",
    "nnet_e2lsh_stream.h",
)

_LAYER_REGISTERED = False
_TEMPLATE_BACKENDS: set[str] = set()


def _bundle_root() -> Path:
    return Path(__file__).resolve().parent


def _normalize_external_root(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    root = Path(path).expanduser().resolve()
    if (root / "nnet_utils").is_dir():
        root = root / "nnet_utils"
    return root


def _register_python_objects():
    global _LAYER_REGISTERED

    if _LAYER_REGISTERED:
        return

    hls4ml.model.layers.register_layer("HEPTAttention", HEPTAttention)
    hls4ml.converters.register_keras_v2_layer_handler("HEPTAttention", parse_hept_attention_layer)
    _LAYER_REGISTERED = True


def _register_backend_templates(backend_name: str):
    backend_key = backend_name.lower()
    if backend_key in _TEMPLATE_BACKENDS:
        return

    backend = hls4ml.backends.get_backend(backend_name)
    backend.register_template(HEPTConfigTemplate)
    backend.register_template(HEPTFunctionTemplate)
    _TEMPLATE_BACKENDS.add(backend_key)


def _resolve_helper_root(external_root: Path | None) -> Path:
    if external_root is None:
        return _bundle_root() / "nnet_utils"

    missing = [header for header in _EXTERNAL_HELPER_HEADERS if not (external_root / header).exists()]
    if missing:
        raise FileNotFoundError(
            "Could not find the expected HEPT helper headers under "
            f"{external_root}: {', '.join(missing)}"
        )

    return external_root


def _register_backend_sources(backend_name: str, external_root: Path | None):
    backend = hls4ml.backends.get_backend(backend_name)
    helper_root = _resolve_helper_root(external_root)

    backend.register_source(_bundle_root() / "nnet_utils" / "nnet_hept.h")
    for header in _EXTERNAL_HELPER_HEADERS:
        backend.register_source(helper_root / header)


def register(
    hept_hls_root: str | Path | None = None,
    backends: tuple[str, ...] | list[str] = ("Vivado", "Vitis"),
):
    """Register the HEPT extension.

    Parameters
    ----------
    hept_hls_root:
        Optional path to either:
          * `.../HEPT/firmware`
          * `.../HEPT/firmware/nnet_utils`
        from an existing `Howard1011/HEPT_HLS` checkout.

        When omitted, the extension uses the helper headers vendored into this
        patch. Supply a path here only if you explicitly want to override the
        bundled copies with an external checkout.

    backends:
        Backends to register against. Defaults to Vivado and Vitis because the
        imported helper headers use Xilinx HLS pragmas.
    """
    _register_python_objects()
    external_root = _normalize_external_root(hept_hls_root)

    for backend_name in backends:
        _register_backend_templates(backend_name)
        _register_backend_sources(backend_name, external_root)


def vendor_external_headers(
    hept_hls_root: str | Path,
    destination_root: str | Path | None = None,
) -> list[Path]:
    """Copy helper headers from an external HEPT_HLS checkout.

    By default, the destination is the bundled `nnet_utils` directory used by
    :func:`register`, which lets you refresh the vendored copies in your fork.
    """
    source_root = _normalize_external_root(hept_hls_root)
    if source_root is None:
        raise ValueError("hept_hls_root must not be None.")

    if destination_root is None:
        destination = _bundle_root() / "nnet_utils"
    else:
        destination = Path(destination_root).expanduser().resolve()

    destination.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for header in _EXTERNAL_HELPER_HEADERS:
        source = source_root / header
        if not source.exists():
            raise FileNotFoundError(f"Missing external HEPT helper header: {source}")
        target = destination / header
        shutil.copy2(source, target)
        copied.append(target)

    return copied
