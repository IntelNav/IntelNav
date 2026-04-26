"""Runtime GPU backend detection.

Goal: one shard binary that runs on any contributor's machine. On start we
probe the host and pick the fastest available backend that llama.cpp knows
how to drive. The user never edits a config.

Probe order (fastest → slowest):

1. **CUDA**   — NVIDIA consumer + datacenter GPUs
2. **ROCm**   — AMD Instinct / recent Radeon
3. **Metal**  — Apple M-series (always available on Darwin arm64)
4. **Vulkan** — fallback for any GPU with a modern driver
                (NVIDIA, AMD, Intel Arc, integrated)
5. **SYCL**   — Intel Arc / Data Center GPU Max when oneAPI is present
6. **CPU**    — always available

What "available" means here is:

* the relevant shared library (libcuda.so, libhsa-runtime64.so, libvulkan.so,
  libsycl.so) loads via ctypes; OR
* on macOS arm64 we assume Metal is always there.

We do NOT probe the device list — that's llama-cpp-python's job. We just
say "this backend's runtime is installed on the host" and let the caller
pass the matching build option.

This module is intentionally cheap — zero model loading, zero subprocesses.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import enum
import platform
import sys
from dataclasses import dataclass
from typing import Optional


class GpuBackend(str, enum.Enum):
    CUDA   = "cuda"
    ROCM   = "rocm"
    METAL  = "metal"
    VULKAN = "vulkan"
    SYCL   = "sycl"
    CPU    = "cpu"


@dataclass
class BackendProbe:
    chosen: GpuBackend
    available: list[GpuBackend]
    reason: str                        # human-readable one-liner

    def summary(self) -> str:
        alt = ", ".join(b.value for b in self.available if b is not self.chosen) or "—"
        return f"backend: {self.chosen.value}  (fallbacks: {alt})  — {self.reason}"


def _lib_loadable(candidates: list[str]) -> bool:
    """True if any of the candidate shared libs can be dlopen'd by name."""
    for name in candidates:
        # ctypes.util.find_library strips lib / .so — pass the short name.
        found = ctypes.util.find_library(name)
        if found:
            try:
                ctypes.CDLL(found)
                return True
            except OSError:
                continue
        # Try the literal soname too, in case find_library misses it.
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _probe_cuda() -> bool:
    return _lib_loadable(["cuda", "libcuda.so.1", "libcudart.so",
                          "libcudart.so.12", "libcudart.so.11"])


def _probe_rocm() -> bool:
    return _lib_loadable(["hsa-runtime64", "libhsa-runtime64.so.1",
                          "amdhip64", "libamdhip64.so"])


def _probe_vulkan() -> bool:
    return _lib_loadable(["vulkan", "libvulkan.so.1"])


def _probe_sycl() -> bool:
    return _lib_loadable(["sycl", "libsycl.so", "libsycl.so.7", "libsycl.so.8"])


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() in {"arm64", "aarch64"}


def detect() -> BackendProbe:
    """Run the probe chain and return the best backend + why.

    Selection is advisory: the caller can override with an explicit
    ``--backend`` flag to force (e.g. benchmarking CPU against Vulkan on
    the same box).
    """
    available: list[GpuBackend] = []

    if _is_apple_silicon():
        available.append(GpuBackend.METAL)

    if _probe_cuda():
        available.append(GpuBackend.CUDA)
    if _probe_rocm():
        available.append(GpuBackend.ROCM)
    if _probe_vulkan():
        available.append(GpuBackend.VULKAN)
    if _probe_sycl():
        available.append(GpuBackend.SYCL)

    available.append(GpuBackend.CPU)  # always last-resort

    # priority order inside `available`
    priority = [
        GpuBackend.CUDA,  GpuBackend.ROCM,   GpuBackend.METAL,
        GpuBackend.VULKAN, GpuBackend.SYCL,  GpuBackend.CPU,
    ]
    chosen = next(b for b in priority if b in available)

    reason = {
        GpuBackend.CUDA:   "CUDA runtime found (libcuda/libcudart)",
        GpuBackend.ROCM:   "ROCm runtime found (libhsa-runtime64/libamdhip64)",
        GpuBackend.METAL:  "Apple Silicon detected",
        GpuBackend.VULKAN: "Vulkan loader found (libvulkan)",
        GpuBackend.SYCL:   "oneAPI SYCL runtime found",
        GpuBackend.CPU:    "no GPU runtime detected — CPU fallback",
    }[chosen]

    return BackendProbe(chosen=chosen, available=available, reason=reason)


def force(name: str) -> BackendProbe:
    """Manual override — trust the caller even if the runtime isn't installed."""
    try:
        b = GpuBackend(name.lower())
    except ValueError as e:
        raise ValueError(
            f"unknown backend {name!r}; expected one of: "
            + ", ".join(x.value for x in GpuBackend)
        ) from e
    return BackendProbe(chosen=b, available=[b], reason=f"forced via CLI override")


def n_gpu_layers_for(backend: GpuBackend, requested: Optional[int]) -> int:
    """Translate the user's --gpu-layers hint into a llama.cpp integer.

    * CPU      → 0 (ignore any positive hint)
    * Others   → requested if given, else -1 (offload all layers)
    """
    if backend is GpuBackend.CPU:
        return 0
    return -1 if requested is None else int(requested)
