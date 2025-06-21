"""
Test package for Quantum HALE Drone System.

This package contains unit tests, integration tests, and test fixtures
for the quantum-secured HALE drone simulation framework.
"""

__version__ = "1.0.0"
__author__ = "Quantum HALE Team"
__email__ = "team@quantum-hale.com"

# Test configuration
TEST_CONFIG = {
    "timeout": 30,  # seconds
    "retries": 3,
    "parallel": True,
    "coverage": True,
    "verbose": True
}

# Import test modules
from . import unit
from . import integration
from . import fixtures 