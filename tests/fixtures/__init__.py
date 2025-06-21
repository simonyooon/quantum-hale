"""
Test fixtures for Quantum HALE Drone System.

This package contains test data, configurations, and fixtures
used across unit and integration tests.
"""

import os
import json
import yaml
from pathlib import Path

# Fixture data directory
FIXTURES_DIR = Path(__file__).parent

def load_test_config(config_name):
    """Load test configuration from fixtures."""
    config_path = FIXTURES_DIR / f"{config_name}.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def load_test_data(data_name):
    """Load test data from fixtures."""
    data_path = FIXTURES_DIR / f"{data_name}.json"
    if data_path.exists():
        with open(data_path, 'r') as f:
            return json.load(f)
    return {}

def get_fixture_path(fixture_name):
    """Get the full path to a fixture file."""
    return FIXTURES_DIR / fixture_name 