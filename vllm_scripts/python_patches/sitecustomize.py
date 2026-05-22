# SPDX-License-Identifier: Apache-2.0
"""Runtime-only Python patches for vllm_scripts launches."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from enum import Enum


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


if _truthy_env("VLLM_XCPU_DISABLE_TORCHVISION"):
    _real_find_spec = importlib.util.find_spec

    def _find_spec_without_torchvision(name, *args, **kwargs):
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return _real_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _find_spec_without_torchvision

    class _InterpolationMode(Enum):
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest-exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    def _unavailable(*args, **kwargs):
        raise RuntimeError(
            "torchvision is disabled by VLLM_XCPU_DISABLE_TORCHVISION. "
            "This launcher supports text-only model execution; image/video "
            "processing is not available."
        )

    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    transforms_v2 = types.ModuleType("torchvision.transforms.v2")
    functional = types.ModuleType("torchvision.transforms.functional")
    functional_v2 = types.ModuleType("torchvision.transforms.v2.functional")

    functional.InterpolationMode = _InterpolationMode
    functional_v2.InterpolationMode = _InterpolationMode
    functional_v2.resize = _unavailable
    functional_v2.normalize = _unavailable
    functional_v2.to_dtype = _unavailable
    functional_v2.to_image = _unavailable

    transforms.InterpolationMode = _InterpolationMode
    transforms.functional = functional
    transforms.v2 = transforms_v2
    transforms_v2.functional = functional_v2
    torchvision.transforms = transforms

    sys.modules.setdefault("torchvision", torchvision)
    sys.modules.setdefault("torchvision.transforms", transforms)
    sys.modules.setdefault("torchvision.transforms.functional", functional)
    sys.modules.setdefault("torchvision.transforms.v2", transforms_v2)
    sys.modules.setdefault("torchvision.transforms.v2.functional", functional_v2)
