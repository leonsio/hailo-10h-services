"""Hailo-10H shared inference services."""

from .ha_action_verification import install as _install_ha_action_verification
from .ha_routing import install as _install_ha_routing
from .ha_state_routing import install as _install_ha_state_routing
from .ha_state_routing_fixes import install as _install_ha_state_routing_fixes
from .litert_optimizations import install as _install_litert_optimizations

_install_litert_optimizations()
_install_ha_routing()
_install_ha_state_routing()
_install_ha_state_routing_fixes()
_install_ha_action_verification()
del _install_litert_optimizations
del _install_ha_routing
del _install_ha_state_routing
del _install_ha_state_routing_fixes
del _install_ha_action_verification
