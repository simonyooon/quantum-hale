"""
Configuration Management for Quantum HALE Drone System
===================================================

This module provides configuration management utilities for loading
and managing configuration files in various formats.

Author: Quantum HALE Development Team
License: MIT
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    duration: float = 3600.0
    timestep: float = 0.01
    real_time_factor: float = 1.0
    random_seed: int = 42
    output_directory: str = "simulation_results"


@dataclass
class NetworkConfig:
    """Configuration for network simulation"""
    topology: str = "mesh"
    num_drones: int = 3
    frequency_band: float = 2.4e9
    jamming_power_dbm: float = 50.0
    ground_stations: list = field(default_factory=list)


@dataclass
class QuantumConfig:
    """Configuration for quantum communications"""
    security_level: int = 3
    key_length: int = 256
    fidelity_threshold: float = 0.95
    enable_qkd: bool = True
    handshake_timeout_ms: int = 5000


@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "quantum-hale-token"
    influxdb_org: str = "quantum-hale"
    influxdb_bucket: str = "simulation-data"
    redis_url: str = "redis://localhost:6379/0"


class ConfigManager:
    """
    Configuration manager for Quantum HALE Drone System
    
    Handles loading, validation, and access to configuration files
    in various formats (YAML, JSON, environment variables).
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration storage
        self.configs = {}
        self.environment_vars = {}
        
        # Load environment variables
        self._load_environment_vars()
        
        logging.info(f"Config Manager initialized with config directory: {config_dir}")
        
    def _load_environment_vars(self):
        """Load configuration from environment variables"""
        env_prefix = "QUANTUM_HALE_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                self.environment_vars[config_key] = value
                
        logging.debug(f"Loaded {len(self.environment_vars)} environment variables")
        
    def load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        filepath = self.config_dir / filename
        
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
                
            self.configs[filename] = config
            logging.info(f"Loaded YAML config from {filepath}")
            return config
            
        except FileNotFoundError:
            logging.warning(f"Config file not found: {filepath}")
            return {}
        except yaml.YAMLError as e:
            logging.error(f"Failed to parse YAML config {filepath}: {e}")
            return {}
        except Exception as e:
            logging.error(f"Failed to load YAML config {filepath}: {e}")
            return {}
            
    def load_json_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        filepath = self.config_dir / filename
        
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
                
            self.configs[filename] = config
            logging.info(f"Loaded JSON config from {filepath}")
            return config
            
        except FileNotFoundError:
            logging.warning(f"Config file not found: {filepath}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON config {filepath}: {e}")
            return {}
        except Exception as e:
            logging.error(f"Failed to load JSON config {filepath}: {e}")
            return {}
            
    def save_yaml_config(self, config: Dict[str, Any], filename: str):
        """Save configuration to YAML file"""
        filepath = self.config_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
                
            self.configs[filename] = config
            logging.info(f"Saved YAML config to {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to save YAML config {filepath}: {e}")
            
    def save_json_config(self, config: Dict[str, Any], filename: str):
        """Save configuration to JSON file"""
        filepath = self.config_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.configs[filename] = config
            logging.info(f"Saved JSON config to {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to save JSON config {filepath}: {e}")
            
    def get_config(self, filename: str) -> Dict[str, Any]:
        """Get configuration by filename"""
        if filename not in self.configs:
            # Try to load the config
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                self.load_yaml_config(filename)
            elif filename.endswith('.json'):
                self.load_json_config(filename)
                
        return self.configs.get(filename, {})
        
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (searches all configs and env vars)"""
        # First check environment variables
        if key in self.environment_vars:
            return self.environment_vars[key]
            
        # Then check all loaded configs
        for config in self.configs.values():
            if isinstance(config, dict):
                keys = key.split('.')
                value = config
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        value = None
                        break
                if value is not None:
                    return value
                    
        return default
        
    def set_value(self, key: str, value: Any, filename: str = None):
        """Set configuration value"""
        if filename:
            # Set in specific config file
            config = self.get_config(filename)
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            
            # Save the config
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                self.save_yaml_config(config, filename)
            elif filename.endswith('.json'):
                self.save_json_config(config, filename)
        else:
            # Set as environment variable
            self.environment_vars[key] = value
            
    def load_simulation_config(self) -> SimulationConfig:
        """Load simulation configuration"""
        config_data = self.get_config("simulation_params.yaml")
        
        return SimulationConfig(
            duration=config_data.get('simulation', {}).get('duration', 3600.0),
            timestep=config_data.get('simulation', {}).get('timestep', 0.01),
            real_time_factor=config_data.get('simulation', {}).get('real_time_factor', 1.0),
            random_seed=config_data.get('simulation', {}).get('random_seed', 42),
            output_directory=config_data.get('simulation', {}).get('output_directory', "simulation_results")
        )
        
    def load_network_config(self) -> NetworkConfig:
        """Load network configuration"""
        config_data = self.get_config("network_topology.yaml")
        
        return NetworkConfig(
            topology=config_data.get('network', {}).get('topology', 'mesh'),
            num_drones=config_data.get('network', {}).get('num_drones', 3),
            frequency_band=config_data.get('network', {}).get('frequency_band', 2.4e9),
            jamming_power_dbm=config_data.get('network', {}).get('jamming_power_dbm', 50.0),
            ground_stations=config_data.get('network', {}).get('ground_stations', [])
        )
        
    def load_quantum_config(self) -> QuantumConfig:
        """Load quantum configuration"""
        config_data = self.get_config("pqc_settings.yaml")
        
        return QuantumConfig(
            security_level=config_data.get('security', {}).get('category', 3),
            key_length=config_data.get('quantum_simulation', {}).get('key_length', 256),
            fidelity_threshold=config_data.get('quantum_simulation', {}).get('fidelity_threshold', 0.95),
            enable_qkd=config_data.get('quantum_simulation', {}).get('enable_qkd', True),
            handshake_timeout_ms=config_data.get('security', {}).get('handshake_timeout', 5000)
        )
        
    def load_database_config(self) -> DatabaseConfig:
        """Load database configuration"""
        # Try to load from config file first
        config_data = self.get_config("database_config.yaml")
        
        return DatabaseConfig(
            influxdb_url=config_data.get('influxdb', {}).get('url', "http://localhost:8086"),
            influxdb_token=config_data.get('influxdb', {}).get('token', "quantum-hale-token"),
            influxdb_org=config_data.get('influxdb', {}).get('org', "quantum-hale"),
            influxdb_bucket=config_data.get('influxdb', {}).get('bucket', "simulation-data"),
            redis_url=config_data.get('redis', {}).get('url', "redis://localhost:6379/0")
        )
        
    def create_default_configs(self):
        """Create default configuration files if they don't exist"""
        default_configs = {
            "simulation_params.yaml": {
                "simulation": {
                    "name": "quantum_hale_baseline",
                    "duration": 3600.0,
                    "timestep": 0.01,
                    "real_time_factor": 1.0,
                    "random_seed": 42,
                    "output_directory": "simulation_results"
                },
                "environment": {
                    "atmosphere": {
                        "density_model": "exponential",
                        "wind_model": "turbulence",
                        "weather_effects": True
                    },
                    "terrain": {
                        "type": "flat",
                        "altitude": 0,
                        "obstacles": []
                    }
                },
                "drones": [
                    {
                        "id": "DRONE_001",
                        "type": "hale_platform",
                        "initial_position": [0, 0, 20000],
                        "initial_velocity": [50, 0, 0],
                        "battery_capacity": 100000,
                        "payload_weight": 75
                    }
                ]
            },
            "pqc_settings.yaml": {
                "algorithms": {
                    "key_encapsulation": {
                        "primary": "Kyber768",
                        "fallback": "Kyber512"
                    },
                    "digital_signature": {
                        "primary": "Dilithium3",
                        "fallback": "Dilithium2"
                    },
                    "hash_function": "SHA3-256"
                },
                "security": {
                    "category": 3,
                    "session_timeout": 3600,
                    "key_rotation_interval": 1800,
                    "max_handshake_retries": 3,
                    "handshake_timeout": 5000
                },
                "quantum_simulation": {
                    "qkd_protocol": "BB84",
                    "key_length": 256,
                    "fidelity_threshold": 0.95,
                    "error_correction": True,
                    "privacy_amplification": True
                }
            },
            "network_topology.yaml": {
                "network": {
                    "topology": "mesh",
                    "num_drones": 3,
                    "frequency_band": 2.4e9,
                    "jamming_power_dbm": 50.0,
                    "ground_stations": [
                        {"id": "GROUND_001", "position": [0, 0, 100], "communication_range": 200000}
                    ]
                },
                "rf_propagation": {
                    "model": "free_space",
                    "frequency": 2.4e9,
                    "antenna_gain": 0.0,
                    "path_loss_exponent": 2.0
                },
                "jamming": {
                    "sources": [
                        {
                            "position": [25000, 25000, 0],
                            "frequency_range": [2.4e9, 2.5e9],
                            "power": 50,
                            "active_time": [1800, 3600]
                        }
                    ]
                }
            }
        }
        
        for filename, config in default_configs.items():
            filepath = self.config_dir / filename
            if not filepath.exists():
                self.save_yaml_config(config, filename)
                logging.info(f"Created default config: {filename}")
                
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        try:
            self._validate_dict(config, schema)
            return True
        except ValueError as e:
            logging.error(f"Config validation failed: {e}")
            return False
            
    def _validate_dict(self, data: Dict[str, Any], schema: Dict[str, Any]):
        """Recursively validate dictionary against schema"""
        for key, expected_type in schema.items():
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
                
            if isinstance(expected_type, dict):
                if not isinstance(data[key], dict):
                    raise ValueError(f"Expected dict for key {key}, got {type(data[key])}")
                self._validate_dict(data[key], expected_type)
            else:
                if not isinstance(data[key], expected_type):
                    raise ValueError(f"Expected {expected_type} for key {key}, got {type(data[key])}")
                    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations"""
        return self.configs.copy()
        
    def reload_configs(self):
        """Reload all configuration files"""
        config_files = list(self.configs.keys())
        self.configs.clear()
        
        for filename in config_files:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                self.load_yaml_config(filename)
            elif filename.endswith('.json'):
                self.load_json_config(filename)
                
        logging.info("All configurations reloaded") 