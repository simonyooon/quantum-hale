# Project Structure Documentation

## Overview

The Quantum HALE Drone project is organized as a modular, scalable simulation platform with clear separation of concerns and comprehensive testing. This document outlines the project structure, component relationships, and development guidelines.

## Directory Structure

```
quantum-hale-drone/
├── src/                          # Source code
│   ├── __init__.py
│   ├── flight_sim/               # Flight dynamics & control
│   │   ├── __init__.py
│   │   ├── hale_dynamics.py      # 6-DOF physics simulation
│   │   ├── pid_controller.py     # Multi-loop flight control
│   │   ├── gazebo_interface.py   # Gazebo/ROS2 integration
│   │   ├── autonomy_engine.py    # Autonomous mission planning
│   │   ├── sensor_fusion.py      # Sensor data processing
│   │   └── pid_controller.py     # Flight control algorithms
│   ├── quantum_comms/            # Quantum communication
│   │   ├── __init__.py
│   │   ├── qkd_simulation.py     # Quantum key distribution
│   │   ├── pqc_handshake.py      # Post-quantum cryptography
│   │   └── crypto_utils.py       # Cryptographic utilities
│   ├── network_sim/              # Network simulation
│   │   ├── __init__.py
│   │   ├── ns3_wrapper.py        # NS-3 integration
│   │   ├── mesh_routing.py       # Mesh networking protocols
│   │   ├── rf_propagation.py     # RF propagation modeling
│   │   └── jamming_models.py     # Jamming & countermeasures
│   ├── integration/              # System integration
│   │   ├── __init__.py
│   │   ├── simulation_orchestrator.py  # Main orchestrator
│   │   ├── data_collector.py     # Telemetry collection
│   │   └── metrics_analyzer.py   # Performance analysis
│   └── utils/                    # Utilities & configuration
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── logging_setup.py      # Logging configuration
│       └── performance_monitor.py # Performance monitoring
├── configs/                      # Configuration files
│   ├── aircraft_parameters.yaml  # Aircraft specifications
│   ├── simulation_params.yaml    # Simulation parameters
│   ├── network_topology.yaml     # Network configuration
│   ├── pqc_settings.yaml         # Quantum settings
│   └── flight_missions.yaml      # Mission definitions
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── unit/                     # Unit tests
│   │   ├── __init__.py
│   │   ├── test_flight_dynamics.py      # Flight dynamics tests
│   │   ├── test_network_simulation.py   # Network tests
│   │   ├── test_pqc_handshake.py        # Quantum tests
│   │   └── test_enhanced_network_simulation.py
│   ├── integration/              # Integration tests
│   │   ├── __init__.py
│   │   ├── test_end_to_end.py           # End-to-end tests
│   │   ├── test_performance.py          # Performance tests
│   │   └── test_advanced_network_integration.py
│   └── fixtures/                 # Test data
│       ├── __init__.py
│       ├── simulation_configs.yaml
│       └── test_data.json
├── data/                         # Simulation data
│   ├── simulation_results/       # Simulation outputs
│   ├── performance_logs/         # Performance metrics
│   └── test_vectors/             # Test data
├── docs/                         # Documentation
│   ├── project_structure.md      # This file
│   ├── simulation_stack_requirements.md
│   ├── quantum_hale_whitepaper.md
│   └── api/                      # API documentation
├── deployment/                   # Deployment configurations
│   ├── docker/                   # Docker configurations
│   ├── kubernetes/               # K8s manifests
│   └── scripts/                  # Deployment scripts
├── models/                       # 3D models & environments
│   ├── gazebo/                   # Gazebo models
│   │   └── environments/         # Simulation environments
│   └── simulation/               # Simulation models
│       ├── flight_dynamics/      # Flight models
│       └── rf_models/            # RF propagation models
├── scripts/                      # Utility scripts
│   ├── run_simulations.py        # Main simulation runner
│   └── setup_environment.sh      # Environment setup
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Development dependencies
├── Dockerfile.development        # Development container
├── Dockerfile.simulation         # Simulation container
├── docker-compose.yml            # Multi-service setup
├── README.md                     # Project overview
├── LICENSE                       # MIT License
└── .gitignore                    # Git ignore rules
```

## Component Architecture

### 1. Flight Simulation Layer (`src/flight_sim/`)

**Core Components:**
- **`hale_dynamics.py`**: 6-DOF physics simulation with realistic aerodynamic modeling
- **`pid_controller.py`**: Advanced PID controllers with anti-windup and filtering
- **`gazebo_interface.py`**: Gazebo/ROS2 integration for visualization and sensor data
- **`autonomy_engine.py`**: Autonomous mission planning and execution
- **`sensor_fusion.py`**: Multi-sensor data fusion and processing

**Key Features:**
- Realistic atmospheric modeling (troposphere/stratosphere)
- Multiple aircraft configurations (Zephyr S, turboprop, research)
- Flight envelope protection and safety systems
- Multi-loop control architecture (altitude, speed, heading)

### 2. Quantum Communications Layer (`src/quantum_comms/`)

**Core Components:**
- **`qkd_simulation.py`**: Quantum Key Distribution simulation
- **`pqc_handshake.py`**: Post-quantum cryptography protocols
- **`crypto_utils.py`**: Cryptographic utilities and key management

**Key Features:**
- BB84 and E91 QKD protocol implementations
- Post-quantum cryptographic algorithms
- Quantum-classical hybrid communication modes
- Security analysis and threat modeling

### 3. Network Simulation Layer (`src/network_sim/`)

**Core Components:**
- **`ns3_wrapper.py`**: NS-3 network simulator integration
- **`mesh_routing.py`**: Mesh networking protocols (AODV, OLSR, DSDV)
- **`rf_propagation.py`**: High-altitude RF propagation modeling
- **`jamming_models.py`**: Jamming detection and countermeasures

**Key Features:**
- Realistic network simulation with NS-3
- Multi-protocol routing with dynamic topology adaptation
- Atmospheric RF propagation effects
- Jamming resistance and mitigation strategies

### 4. Integration Layer (`src/integration/`)

**Core Components:**
- **`simulation_orchestrator.py`**: Main simulation orchestrator
- **`data_collector.py`**: Telemetry and performance data collection
- **`metrics_analyzer.py`**: Performance analysis and optimization

**Key Features:**
- Coordinated multi-component simulation
- Real-time data collection and analysis
- Performance monitoring and optimization
- Error handling and recovery mechanisms

### 5. Utilities Layer (`src/utils/`)

**Core Components:**
- **`config.py`**: Configuration management and validation
- **`logging_setup.py`**: Centralized logging configuration
- **`performance_monitor.py`**: Performance monitoring and profiling

**Key Features:**
- YAML-based configuration management
- Structured logging with multiple output formats
- Performance profiling and bottleneck detection
- Error reporting and debugging utilities

## Configuration Management

### Aircraft Parameters (`configs/aircraft_parameters.yaml`)

Defines aircraft specifications for different HALE drone types:

```yaml
zephyr_s:
  wingspan: 25.0
  wing_area: 31.5
  mass_max_takeoff: 140.0
  service_ceiling: 70000
  # ... additional parameters
```

### Simulation Parameters (`configs/simulation_params.yaml`)

Controls simulation behavior and component settings:

```yaml
simulation:
  duration: 3600.0
  timestep: 0.01
  num_drones: 3

flight:
  aircraft_type: "zephyr_s"
  flight_mode: "auto"
  # ... flight parameters
```

## Testing Strategy

### Unit Tests (`tests/unit/`)

**Coverage Areas:**
- Flight dynamics physics and control algorithms
- Network simulation and routing protocols
- Quantum communication protocols
- Utility functions and data structures

**Test Structure:**
```python
class TestHALEDynamics(unittest.TestCase):
    def test_aerodynamic_forces(self):
        # Test aerodynamic force calculations
        
    def test_flight_envelope_protection(self):
        # Test envelope protection mechanisms
```

### Integration Tests (`tests/integration/`)

**Coverage Areas:**
- End-to-end system simulation
- Multi-component interaction
- Performance under load
- Error handling and recovery

**Test Structure:**
```python
class TestEndToEnd(unittest.TestCase):
    def test_full_simulation_run(self):
        # Test complete simulation workflow
        
    def test_multi_drone_coordination(self):
        # Test multi-drone scenarios
```

## Data Flow

### Simulation Execution Flow

1. **Initialization**: Load configurations, initialize components
2. **Setup**: Create aircraft, network topology, quantum channels
3. **Execution**: Run simulation loop with coordinated component updates
4. **Data Collection**: Gather telemetry, performance metrics, logs
5. **Analysis**: Process results, generate reports, save data

### Component Communication

```
Orchestrator
    ↓
Flight Dynamics ←→ PID Controllers
    ↓
Gazebo Interface ←→ Sensor Data
    ↓
Network Simulation ←→ Mesh Routing
    ↓
Quantum Communications ←→ QKD/PQC
    ↓
Data Collection ←→ Performance Monitoring
```

## Development Guidelines

### Code Style
- **Python**: PEP 8 compliance with 120-character line limit
- **Documentation**: Google-style docstrings for all public methods
- **Type Hints**: Comprehensive type annotations
- **Error Handling**: Graceful error handling with meaningful messages

### Testing Requirements
- **Unit Tests**: Minimum 90% code coverage
- **Integration Tests**: All major workflows covered
- **Performance Tests**: Baseline performance metrics
- **Documentation**: All public APIs documented

### Configuration Management
- **YAML Format**: Human-readable configuration files
- **Validation**: Schema validation for all configurations
- **Defaults**: Sensible defaults for all parameters
- **Documentation**: Comprehensive parameter documentation

## Performance Considerations

### Simulation Performance
- **Real-time Factor**: Target 1.0x real-time simulation
- **Memory Usage**: Efficient data structures and garbage collection
- **CPU Utilization**: Multi-threading for independent components
- **I/O Operations**: Asynchronous data collection and storage

### Scalability
- **Multi-drone Support**: Efficient handling of multiple aircraft
- **Network Scaling**: Support for large network topologies
- **Quantum Scaling**: Efficient quantum state simulation
- **Data Storage**: Compressed storage for large datasets

## Future Extensions

### Planned Components
- **Machine Learning**: ML-based control and optimization
- **Hardware Integration**: Real quantum hardware support
- **Visualization**: Real-time 3D visualization
- **Cloud Deployment**: Cloud-based simulation platform

### Architecture Evolution
- **Microservices**: Component-based deployment
- **API Gateway**: RESTful API for external integration
- **Message Queues**: Asynchronous component communication
- **Containerization**: Full container-based deployment

## Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements-dev.txt`
3. Run tests: `python -m pytest tests/ -v`
4. Follow coding standards and testing requirements

### Pull Request Process
1. Create feature branch from `main`
2. Implement changes with comprehensive tests
3. Update documentation as needed
4. Submit pull request with detailed description
5. Address review comments and ensure CI passes

This structure provides a solid foundation for a complex, multi-disciplinary simulation platform while maintaining clarity, testability, and extensibility.