"""
Flight Simulation Package for Quantum HALE Drone System
=====================================================

This package provides flight dynamics simulation, autonomy engine,
sensor fusion, and Gazebo interface for HALE drone operations.

Modules:
- hale_dynamics: Flight dynamics and control systems
- autonomy_engine: Autonomous mission planning and execution
- sensor_fusion: Multi-sensor data fusion and processing
- gazebo_interface: Gazebo simulation environment interface

Author: Quantum HALE Development Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Quantum HALE Development Team"

from .hale_dynamics import HALEDynamics
from .autonomy_engine import AutonomyEngine
from .sensor_fusion import SensorFusion
from .gazebo_interface import GazeboInterface

__all__ = [
    'HALEDynamics',
    'AutonomyEngine', 
    'SensorFusion',
    'GazeboInterface'
] 