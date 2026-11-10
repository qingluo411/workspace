"""DLC GLM-5.3 text-serving compatibility bootstrap.

The default PyPI torchaudio/torchvision wheels are built for CUDA and cannot
be imported by the CPU-only DLC PyTorch build used here.  vLLM/Transformers
only need these packages for optional multimodal/media paths, but some of
their modules import them unconditionally.  When this file is put on
``PYTHONPATH`` (the launcher does that), install minimal importable stubs so
GLM-5.3 text inference can reach the DLC model code.

Set ``GLM5_VLLM_MEDIA_STUBS=0`` to disable this behavior for an environment
with working torchaudio/torchvision wheels.
"""

from __future__ import annotations

import os
import sys
import types
from importlib.machinery import ModuleSpec


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, None)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _install_torchaudio_stub() -> None:
    torchaudio = _module("torchaudio")
    torchaudio.__version__ = "2.5.0"
    # These are imported lazily by audio processors; GLM-5.3 text does not
    # call them.  Keep the modules importable and fail clearly if used.
    for name in (
        "torchaudio.functional",
        "torchaudio.transforms",
        "torchaudio.compliance",
    ):
        _module(name)


def _install_torchvision_stub() -> None:
    torchvision = _module("torchvision")
    torchvision.__version__ = "0.20.1"

    io = _module("torchvision.io")
    transforms = _module("torchvision.transforms")
    functional = _module("torchvision.transforms.functional")
    transforms_v2 = _module("torchvision.transforms.v2")
    functional_v2 = _module("torchvision.transforms.v2.functional")

    torchvision.io = io
    torchvision.transforms = transforms
    transforms.functional = functional
    transforms.v2 = transforms_v2
    transforms_v2.functional = functional_v2

    class ImageReadMode:
        UNCHANGED = 0
        GRAY = 1
        GRAY_ALPHA = 2
        RGB = 3
        RGB_ALPHA = 4

    class InterpolationMode:
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest-exact"
        BOX = "box"
        BILINEAR = "bilinear"
        HAMMING = "hamming"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "torchvision media support is unavailable in the DLC environment"
        )

    io.ImageReadMode = ImageReadMode
    io.decode_image = unavailable
    transforms.InterpolationMode = InterpolationMode
    functional.InterpolationMode = InterpolationMode
    functional_v2.InterpolationMode = InterpolationMode
    functional.pil_to_tensor = unavailable
    functional_v2.pil_to_tensor = unavailable

    # Transformers checks these submodules as part of torchvision's optional
    # import surface.  Keep them importable without pulling in incompatible
    # native extensions.
    for name in (
        "torchvision.datasets",
        "torchvision.models",
        "torchvision.ops",
        "torchvision.utils",
        "torchvision._meta_registrations",
    ):
        setattr(torchvision, name.rsplit(".", 1)[-1], _module(name))


def _install_optional_media_stubs() -> None:
    if os.environ.get("GLM5_VLLM_MEDIA_STUBS", "1") == "0":
        return

    try:
        import torchaudio  # noqa: F401
    except Exception:
        sys.modules.pop("torchaudio", None)
        _install_torchaudio_stub()

    try:
        import torchvision  # noqa: F401
    except Exception:
        # A partially-imported package must not be reused by transformers.
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                del sys.modules[name]
        _install_torchvision_stub()


_install_optional_media_stubs()
