"""
Integration tests for Quantum HALE Drone System.

This package contains integration tests that verify the interaction
between different components of the quantum-secured HALE drone simulation framework.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Integration test configuration
INTEGRATION_CONFIG = {
    "timeout": 60,  # seconds
    "retries": 2,
    "parallel": False,  # Integration tests should run sequentially
    "coverage": True,
    "verbose": True
}

# Test fixtures for integration tests
@pytest.fixture
def simulation_environment():
    """Set up a complete simulation environment for integration testing."""
    from integration.simulation_orchestrator import SimulationOrchestrator
    from utils.config import ConfigManager
    
    config_manager = ConfigManager()
    orchestrator = SimulationOrchestrator(config_manager)
    
    yield orchestrator
    
    # Cleanup
    orchestrator.shutdown()

@pytest.fixture
def quantum_network_setup():
    """Set up quantum network components for integration testing."""
    from quantum_comms.pqc_handshake import PQCHandshake
    from quantum_comms.qkd_simulation import QKDSimulation
    from network_sim.ns3_wrapper import NS3Wrapper
    
    # Initialize components
    pqc_config = {"key_encapsulation": "Kyber768", "security_level": 3}
    qkd_config = {"protocol": "BB84", "key_length": 256}
    ns3_config = {"simulation_time": 10, "log_level": "error"}
    
    pqc = PQCHandshake(pqc_config)
    qkd = QKDSimulation(qkd_config)
    ns3 = NS3Wrapper(ns3_config)
    
    return {
        "pqc": pqc,
        "qkd": qkd,
        "ns3": ns3
    } 