"""Hailo-10H shared inference services."""

from .ha_routing import install as _install_ha_routing
from .litert_optimizations import install as _install_litert_optimizations

_install_litert_optimizations()
_install_ha_routing()
del _install_litert_optimizations
del _install_ha_routing
