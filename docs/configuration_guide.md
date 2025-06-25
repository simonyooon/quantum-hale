# Configuration Guide

## Overview

The Quantum HALE Drone simulation platform uses YAML-based configuration files to control all aspects of the simulation. This guide explains each configuration file, its parameters, and how to customize them for different scenarios.

## Configuration Files

### 1. Aircraft Parameters (`configs/aircraft_parameters.yaml`)

Defines aircraft specifications for different HALE drone types.

#### Aircraft Types

**Zephyr S (Solar-Powered)**
```yaml
zephyr_s:
  # Physical dimensions
  wingspan: 25.0          # meters
  wing_area: 31.5         # m²
  length: 7.5             # meters
  mass_empty: 75.0        # kg
  mass_max_takeoff: 140.0 # kg
  
  # Aerodynamic coefficients
  cl_alpha: 5.73          # lift curve slope (1/rad)
  cd0: 0.015              # zero-lift drag coefficient
  oswald_efficiency: 0.85 # Oswald efficiency factor
  aspect_ratio: 19.8      # wing aspect ratio
  
  # Propulsion (electric motors)
  thrust_max: 800.0       # N
  specific_fuel_consumption: 0.0  # kg/N/s (electric)
  propeller_efficiency: 0.85
  
  # Performance
  stall_speed: 12.0       # m/s
  max_speed: 35.0         # m/s
  service_ceiling: 70000  # meters (70km)
  range_max: 5000000      # meters (5000km)
  endurance_max: 2592000  # seconds (30 days)
```

**Turboprop HALE**
```yaml
hale_turboprop:
  # Physical dimensions
  wingspan: 20.0          # meters
  wing_area: 24.0         # m²
  mass_empty: 2223.0      # kg
  mass_max_takeoff: 4763.0 # kg
  
  # Propulsion (turboprop)
  thrust_max: 12000.0     # N
  specific_fuel_consumption: 0.0003  # kg/N/s
  
  # Performance
  stall_speed: 25.0       # m/s
  max_speed: 120.0        # m/s
  service_ceiling: 15000  # meters (15km)
  endurance_max: 43200    # seconds (12 hours)
```

#### PID Controller Gains

Each aircraft type has optimized PID gains:

```yaml
pid_gains:
  zephyr_s:
    altitude_controller:
      kp: 0.08
      ki: 0.005
      kd: 0.03
      output_limit: 0.3
    speed_controller:
      kp: 0.15
      ki: 0.01
      kd: 0.05
      output_limit: 0.8
    # ... additional controllers
```

#### Flight Envelope Limits

```yaml
envelope_limits:
  zephyr_s:
    max_altitude: 70000
    min_altitude: 100
    max_airspeed: 35.0
    min_airspeed: 14.0
    max_load_factor: 2.0
    max_angle_of_attack: 12.0  # degrees
```

### 2. Simulation Parameters (`configs/simulation_params.yaml`)

Controls overall simulation behavior and component settings.

#### General Simulation Settings

```yaml
simulation:
  duration: 3600.0          # Simulation duration in seconds
  timestep: 0.01            # Simulation time step in seconds
  real_time_factor: 1.0     # Real-time factor (1.0 = real-time)
  num_drones: 3             # Number of HALE drones
  output_directory: "data/simulation_results"
  log_level: "INFO"
```

#### Flight Simulation Settings

```yaml
flight:
  aircraft_type: "zephyr_s"  # Aircraft type from aircraft_parameters.yaml
  flight_mode: "auto"        # manual, altitude_hold, heading_hold, auto
  initial_altitude: 20000.0  # Initial altitude in meters
  initial_speed: 50.0        # Initial airspeed in m/s
  initial_heading: 0.0       # Initial heading in degrees
  
  # Flight control parameters
  altitude_setpoint: 20000.0 # Target altitude in meters
  speed_setpoint: 50.0       # Target airspeed in m/s
  heading_setpoint: 0.0      # Target heading in degrees
  
  # Environmental conditions
  wind_conditions: [5.0, 2.0, 0.0]  # Wind velocity [north, east, up] in m/s
  turbulence_intensity: 0.1  # Turbulence intensity (0.0-1.0)
  
  # Flight envelope limits
  max_altitude: 70000.0      # Maximum altitude in meters
  min_altitude: 100.0        # Minimum safe altitude in meters
  max_speed: 35.0            # Maximum airspeed in m/s
  min_speed: 14.0            # Minimum airspeed in m/s
```

#### Network Simulation Settings

```yaml
network:
  topology_type: "mesh"      # mesh, star, ring
  protocol: "aodv"           # aodv, olsr, dsdv
  num_nodes: 5               # Number of network nodes
  transmission_range: 50000  # Transmission range in meters
  data_rate: 1000000         # Data rate in bits per second
  
  # Ground stations
  ground_stations:
    - [40.7128, -74.0060]    # New York
    - [51.5074, -0.1278]     # London
    - [35.6762, 139.6503]    # Tokyo
  
  # Jamming simulation
  jamming_enabled: false     # Enable jamming simulation
  jamming_sources:           # Jamming source locations
    # - [45.0, -75.0, 0.0]   # Example jamming source
```

#### Quantum Communication Settings

```yaml
quantum:
  qkd_enabled: true          # Enable QKD simulation
  pqc_enabled: true          # Enable post-quantum cryptography
  key_length: 256            # Quantum key length in bits
  key_rate: 1000             # Key generation rate in bits per second
  error_rate: 0.01           # Quantum bit error rate
  
  # Quantum hardware parameters
  photon_detection_efficiency: 0.8
  dark_count_rate: 100       # Dark counts per second
  coherence_time: 1e-6       # Photon coherence time in seconds
```

#### Autonomy and Mission Settings

```yaml
autonomy:
  mission_type: "surveillance"  # surveillance, communication, research
  waypoints:                    # Mission waypoints [lat, lon, alt]
    - [40.7128, -74.0060, 20000.0]  # New York
    - [51.5074, -0.1278, 20000.0]   # London
    - [35.6762, 139.6503, 20000.0]  # Tokyo
    - [40.7128, -74.0060, 20000.0]  # Return to start
  
  # Autonomous behavior parameters
  collision_avoidance: true   # Enable collision avoidance
  energy_management: true     # Enable energy management
  weather_avoidance: true     # Enable weather avoidance
  
  # Decision making parameters
  decision_frequency: 1.0     # Decision making frequency in Hz
  planning_horizon: 300.0     # Planning horizon in seconds
```

#### Performance Monitoring Settings

```yaml
performance:
  metrics_enabled: true       # Enable performance metrics collection
  sampling_rate: 10.0         # Metrics sampling rate in Hz
  storage_interval: 60.0      # Data storage interval in seconds
  
  # Performance thresholds
  max_cpu_usage: 80.0         # Maximum CPU usage percentage
  max_memory_usage: 80.0      # Maximum memory usage percentage
  max_latency: 100.0          # Maximum network latency in ms
```

### 3. Network Topology (`configs/network_topology.yaml`)

Defines network topology and routing configurations.

```yaml
network_topology:
  type: "mesh"
  nodes:
    - id: "DRONE_001"
      position: [0.0, 0.0, 20000.0]
      capabilities: ["qkd", "mesh_routing"]
    - id: "DRONE_002"
      position: [10000.0, 0.0, 20000.0]
      capabilities: ["qkd", "mesh_routing"]
    - id: "GROUND_001"
      position: [0.0, 0.0, 0.0]
      capabilities: ["qkd", "ground_station"]
  
  links:
    - source: "DRONE_001"
      destination: "DRONE_002"
      type: "wireless"
      bandwidth: 1000000
    - source: "DRONE_001"
      destination: "GROUND_001"
      type: "wireless"
      bandwidth: 500000
```

### 4. PQC Settings (`configs/pqc_settings.yaml`)

Configures post-quantum cryptography parameters.

```yaml
pqc_settings:
  algorithms:
    - name: "Kyber"
      security_level: 256
      key_size: 3168
      ciphertext_size: 1568
    - name: "Dilithium"
      security_level: 256
      key_size: 1952
      signature_size: 3366
  
  handshake:
    timeout: 30.0            # Handshake timeout in seconds
    retry_attempts: 3        # Number of retry attempts
    key_refresh_interval: 3600.0  # Key refresh interval in seconds
  
  security:
    forward_secrecy: true    # Enable forward secrecy
    perfect_forward_secrecy: true  # Enable perfect forward secrecy
    key_confirmation: true   # Enable key confirmation
```

## Configuration Best Practices

### 1. Aircraft Selection

**Choose aircraft type based on mission requirements:**

- **Zephyr S**: Long-endurance surveillance, solar-powered
- **Turboprop**: Medium-range missions, conventional fuel
- **Research**: High-altitude research missions

### 2. Flight Mode Selection

**Select appropriate flight mode:**

- **Manual**: For testing and debugging
- **Altitude Hold**: For stable altitude operations
- **Heading Hold**: For directional missions
- **Auto**: For full autonomous operation

### 3. Network Configuration

**Optimize network for mission requirements:**

- **Mesh Topology**: For multi-drone coordination
- **Star Topology**: For centralized control
- **Ring Topology**: For redundant communications

### 4. Quantum Settings

**Configure quantum parameters based on hardware:**

- **Key Rate**: Match to quantum hardware capabilities
- **Error Rate**: Set based on environmental conditions
- **Key Length**: Choose based on security requirements

## Environment-Specific Configurations

### Development Environment

```yaml
simulation:
  duration: 60.0             # Short duration for testing
  timestep: 0.1              # Larger timestep for speed
  real_time_factor: 10.0     # Faster than real-time

performance:
  metrics_enabled: true
  sampling_rate: 1.0         # Lower sampling for speed
```

### Production Environment

```yaml
simulation:
  duration: 86400.0          # 24-hour simulation
  timestep: 0.01             # High precision
  real_time_factor: 1.0      # Real-time simulation

performance:
  metrics_enabled: true
  sampling_rate: 10.0        # High sampling rate
  storage_interval: 60.0     # Frequent data storage
```

### Testing Environment

```yaml
simulation:
  duration: 300.0            # 5-minute test runs
  timestep: 0.05             # Balanced precision/speed

testing:
  test_mode: true            # Enable test mode
  validation_enabled: true   # Enable validation checks
  error_tolerance: 0.01      # Error tolerance for validation
```

## Configuration Validation

### Schema Validation

The system validates all configuration files against predefined schemas:

```python
# Example validation
from src.utils.config import validate_config

config = load_yaml("configs/simulation_params.yaml")
validate_config(config, "simulation_params_schema")
```

### Parameter Ranges

Key parameters have defined ranges:

- **Altitude**: 0-100,000 meters
- **Airspeed**: 0-500 m/s
- **Timestep**: 0.001-1.0 seconds
- **Real-time factor**: 0.1-100.0

### Dependency Validation

The system checks for configuration dependencies:

- Aircraft type must exist in aircraft_parameters.yaml
- Network nodes must have valid positions
- Quantum settings must be compatible

## Configuration Examples

### Surveillance Mission

```yaml
flight:
  aircraft_type: "zephyr_s"
  flight_mode: "auto"
  altitude_setpoint: 20000.0
  speed_setpoint: 25.0

autonomy:
  mission_type: "surveillance"
  waypoints:
    - [40.7128, -74.0060, 20000.0]  # New York
    - [40.7589, -73.9851, 20000.0]  # Manhattan
    - [40.7505, -73.9934, 20000.0]  # Brooklyn
```

### Communication Relay

```yaml
flight:
  aircraft_type: "hale_turboprop"
  flight_mode: "altitude_hold"
  altitude_setpoint: 15000.0

network:
  topology_type: "mesh"
  protocol: "olsr"
  num_nodes: 5

quantum:
  qkd_enabled: true
  key_rate: 2000
  error_rate: 0.005
```

### Research Mission

```yaml
flight:
  aircraft_type: "hale_research"
  flight_mode: "auto"
  altitude_setpoint: 18000.0

autonomy:
  mission_type: "research"
  waypoints:
    - [0.0, 0.0, 18000.0]
    - [10.0, 0.0, 18000.0]
    - [20.0, 0.0, 18000.0]

performance:
  metrics_enabled: true
  sampling_rate: 20.0
  storage_interval: 30.0
```

## Troubleshooting

### Common Configuration Issues

1. **Invalid Aircraft Type**
   - Error: "Aircraft type 'invalid_type' not found"
   - Solution: Check aircraft_parameters.yaml for valid types

2. **Invalid Parameter Values**
   - Error: "Parameter 'altitude' out of range"
   - Solution: Check parameter ranges in documentation

3. **Missing Dependencies**
   - Error: "Required parameter 'thrust_max' missing"
   - Solution: Ensure all required parameters are defined

4. **Configuration Conflicts**
   - Error: "Conflicting parameters detected"
   - Solution: Check for parameter conflicts in configuration

### Debugging Configuration

Enable debug logging to troubleshoot configuration issues:

```yaml
simulation:
  log_level: "DEBUG"
```

Use the configuration validator:

```bash
python -m src.utils.config --validate configs/simulation_params.yaml
```

## Advanced Configuration

### Custom Aircraft Types

Define custom aircraft by adding to aircraft_parameters.yaml:

```yaml
custom_aircraft:
  wingspan: 30.0
  wing_area: 40.0
  mass_max_takeoff: 2000.0
  # ... additional parameters
```

### Custom Mission Profiles

Create custom mission profiles:

```yaml
custom_missions:
  patrol_mission:
    waypoints:
      - [lat1, lon1, alt1]
      - [lat2, lon2, alt2]
    flight_mode: "auto"
    duration: 7200.0
```

### Performance Tuning

Optimize performance for specific hardware:

```yaml
performance:
  max_threads: 8
  memory_limit: "4GB"
  cache_size: 1000
  compression_level: 6
```

This configuration guide provides comprehensive coverage of all configuration options and best practices for customizing the Quantum HALE Drone simulation platform. 