"""
Integration Package for Quantum HALE Drone System
===============================================

This package provides integration and orchestration between different
components of the HALE drone system, including simulation coordination,
data collection, and metrics analysis.

Modules:
- simulation_orchestrator: Coordinates all simulation components
- data_collector: Collects and stores simulation data
- metrics_analyzer: Analyzes performance and security metrics

Author: Quantum HALE Development Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Quantum HALE Development Team"

from .simulation_orchestrator import SimulationOrchestrator
from .data_collector import DataCollector
from .metrics_analyzer import MetricsAnalyzer

__all__ = [
    'SimulationOrchestrator',
    'DataCollector',
    'MetricsAnalyzer'
] 