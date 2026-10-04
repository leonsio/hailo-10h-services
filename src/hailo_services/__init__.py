"""Hailo-10H shared inference services."""

from .litert_optimizations import install as _install_litert_optimizations

_install_litert_optimizations()
del _install_litert_optimizations
