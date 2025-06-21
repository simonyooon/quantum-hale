"""
Unit tests for Quantum HALE Drone System.

This package contains unit tests for individual components and modules
of the quantum-secured HALE drone simulation framework.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Test fixtures and utilities
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "simulation": {
            "duration": 100,
            "timestep": 0.1,
            "random_seed": 42
        },
        "quantum": {
            "security_level": 3,
            "key_length": 256
        }
    }

@pytest.fixture
def mock_quantum_state():
    """Mock quantum state for testing."""
    return {
        "qubits": 8,
        "fidelity": 0.95,
        "entanglement": True
    } 