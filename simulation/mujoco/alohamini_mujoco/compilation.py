"""Bound compiler-only asset caching while preserving native model compilation."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock

import mujoco

# Cache eviction affects recompilation time, never compiled geometry or dynamics.
# Keep export/replay recompilation within a worker's memory budget without
# retaining two full sets of processed CAD assets under different resource paths.
COMPILER_CACHE_BYTES = 64 * 1024 * 1024
_LOCK = RLock()


@contextmanager
def compiler_cache():
    with _LOCK:
        cache = mujoco.mj_getCache()
        previous = mujoco.mj_getCacheCapacity(cache)
        mujoco.mj_setCacheCapacity(cache, min(previous, COMPILER_CACHE_BYTES))
        try:
            yield
        finally:
            mujoco.mj_setCacheCapacity(cache, previous)


def compile_xml(xml, *, assets=None):
    with compiler_cache():
        # Explicit specs release parsed/processed source assets with this scope;
        # mj_loadXML otherwise retains a global "last XML" specification.
        spec = mujoco.MjSpec.from_string(xml, assets=assets)
        return spec.compile()


def load_xml(path):
    with compiler_cache():
        spec = mujoco.MjSpec.from_file(str(path))
        return spec.compile()
