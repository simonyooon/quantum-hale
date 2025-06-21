"""
Quantum HALE Drone System - Main Package

A comprehensive simulation framework for High-Altitude Long-Endurance (HALE) 
drones with quantum-secured communications and advanced network simulation capabilities.
"""

__version__ = "1.0.0"
__author__ = "Quantum HALE Team"
__email__ = "contact@quantum-hale.com"

from . import quantum_comms
from . import network_sim
from . import flight_sim
from . import integration
from . import utils

__all__ = [
    "quantum_comms",
    "network_sim", 
    "flight_sim",
    "integration",
    "utils"
] 