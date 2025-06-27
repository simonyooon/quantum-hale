"""
Simulation Orchestrator for Quantum HALE Drone System
===================================================

This module orchestrates the integration between flight dynamics,
autonomy engine, sensor fusion, quantum communications, and network
simulation components.

Author: Quantum HALE Development Team
License: MIT
"""

import time
import logging
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import yaml
from pathlib import Path
import json

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib not available - visualization disabled")
    VISUALIZATION_AVAILABLE = False

# Mock implementations for Windows compatibility
try:
    from flight_sim.hale_dynamics import HALEDynamics, AircraftState, AircraftParameters
except ImportError:
    logging.warning("Flight dynamics not available - using mock implementation")
    from unittest.mock import Mock
    HALEDynamics = Mock
    AircraftState = Mock
    AircraftParameters = Mock

try:
    from flight_sim.autonomy_engine import AutonomyEngine, Mission, MissionType, Waypoint
except ImportError:
    logging.warning("Autonomy engine not available - using mock implementation")
    from unittest.mock import Mock
    AutonomyEngine = Mock
    Mission = Mock
    MissionType = Mock
    Waypoint = Mock

try:
    from flight_sim.sensor_fusion import SensorFusion, FusedState
except ImportError:
    logging.warning("Sensor fusion not available - using mock implementation")
    from unittest.mock import Mock
    SensorFusion = Mock
    FusedState = Mock

try:
    from flight_sim.gazebo_interface import GazeboInterface, GazeboModelConfig
except ImportError:
    logging.warning("Gazebo/ROS2 not available - using mock interface")
    from unittest.mock import Mock
    GazeboInterface = Mock
    GazeboModelConfig = Mock

try:
    from quantum_comms.pqc_handshake import PQCHandshake, PQCConfiguration, SecurityLevel
except ImportError:
    logging.warning("Quantum communications not available - using mock implementation")
    from unittest.mock import Mock
    PQCHandshake = Mock
    PQCConfiguration = Mock
    SecurityLevel = Mock

try:
    from network_sim.ns3_wrapper import NS3Wrapper
    from network_sim.rf_propagation import RFPropagation
    from network_sim.jamming_models import JammingModels, JammingSource, JammingType
    from network_sim.mesh_routing import MeshRouting
except ImportError:
    logging.warning("NS-3 not available, using mock simulation")
    from unittest.mock import Mock
    NS3Wrapper = Mock
    RFPropagation = Mock
    JammingModels = Mock
    JammingSource = Mock
    JammingType = Mock
    MeshRouting = Mock


class SimulationState(Enum):
    """Simulation execution states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SimulationConfig:
    """Configuration for the complete simulation"""
    # Simulation parameters
    duration: float = 3600.0  # seconds
    timestep: float = 0.01    # seconds
    real_time_factor: float = 1.0
    
    # Aircraft configuration
    aircraft_params: Optional[Any] = None
    
    # Mission configuration
    mission_type: Optional[Any] = None
    waypoints: Optional[List[Any]] = None
    
    # Network configuration
    network_topology: str = "mesh"
    num_drones: int = 3
    ground_stations: Optional[List[Tuple[float, float]]] = None
    
    # Quantum configuration
    security_level: Optional[Any] = None
    enable_qkd: bool = True
    
    # Environment configuration
    wind_conditions: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    jamming_sources: Optional[List[Dict[str, Any]]] = None
    
    # Output configuration
    output_directory: str = "simulation_results"
    save_telemetry: bool = True
    save_network_data: bool = True
    save_quantum_data: bool = True


class MockComponent:
    """Mock component for testing orchestrator functionality"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_running = False
        self.operation_count = 0
        self.last_update = time.time()
        self.state = None  # For flight dynamics mock
        
    def start(self):
        self.is_running = True
        logging.info(f"Mock {self.name} component started")
        
    def stop(self):
        self.is_running = False
        logging.info(f"Mock {self.name} component stopped")
        
    def step(self):
        """Step the component simulation"""
        if self.is_running:
            self.operation_count += 1
            self.last_update = time.time()
            
    def get_telemetry(self):
        """Mock telemetry data"""
        if self.name == "flight_dynamics":
            return {
                'position': {'latitude': 0.0, 'longitude': 0.0, 'altitude': 20000.0},
                'velocity': {'airspeed': 50.0, 'ground_speed': 50.0, 'heading': 0.0},
                'energy': {'fuel_remaining': 50.0, 'fuel_consumption_rate': 0.1},
                'controls': {'throttle': 0.5, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
            }
        return {}
    
    def get_network_status(self):
        """Mock network status"""
        return {
            'num_nodes': 3,
            'num_links': 2,
            'topology_type': 'mesh'
        }
    
    def is_jamming_active(self):
        """Mock jamming status"""
        return False
        
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'operations': self.operation_count,
            'last_update': self.last_update,
            'status': 'running' if self.is_running else 'stopped'
        }


class SimulationOrchestrator:
    """
    Main simulation orchestrator for Quantum HALE Drone System
    
    Coordinates all simulation components and manages the overall
    simulation execution flow with enhanced monitoring and mock fallbacks.
    """
    
    def __init__(self, config):
        # Handle both SimulationConfig objects and dictionaries
        if isinstance(config, dict):
            # Convert dictionary to SimulationConfig
            self.config = SimulationConfig(
                duration=config.get('duration', 3600.0),
                timestep=config.get('timestep', 0.01),
                real_time_factor=config.get('real_time_factor', 1.0),
                aircraft_params=config.get('aircraft_params'),
                mission_type=config.get('mission_type'),
                waypoints=config.get('waypoints'),
                network_topology=config.get('network_topology', 'mesh'),
                num_drones=config.get('num_drones', 3),
                ground_stations=config.get('ground_stations'),
                security_level=config.get('security_level'),
                enable_qkd=config.get('enable_qkd', True),
                wind_conditions=config.get('wind_conditions', (0.0, 0.0, 0.0)),
                jamming_sources=config.get('jamming_sources'),
                output_directory=config.get('output_directory', 'simulation_results'),
                save_telemetry=config.get('save_telemetry', True),
                save_network_data=config.get('save_network_data', True),
                save_quantum_data=config.get('save_quantum_data', True)
            )
        elif hasattr(config, 'get_simulation_config'):
            # ConfigManager object - extract simulation config
            sim_config = config.get_simulation_config()
            self.config = SimulationConfig(
                duration=sim_config.get('duration', 3600.0),
                timestep=sim_config.get('timestep', 0.01),
                real_time_factor=sim_config.get('real_time_factor', 1.0),
                aircraft_params=sim_config.get('aircraft_params'),
                mission_type=sim_config.get('mission_type'),
                waypoints=sim_config.get('waypoints'),
                network_topology=sim_config.get('network_topology', 'mesh'),
                num_drones=sim_config.get('num_drones', 3),
                ground_stations=sim_config.get('ground_stations'),
                security_level=sim_config.get('security_level'),
                enable_qkd=sim_config.get('enable_qkd', True),
                wind_conditions=sim_config.get('wind_conditions', (0.0, 0.0, 0.0)),
                jamming_sources=sim_config.get('jamming_sources'),
                output_directory=sim_config.get('output_directory', 'simulation_results'),
                save_telemetry=sim_config.get('save_telemetry', True),
                save_network_data=sim_config.get('save_network_data', True),
                save_quantum_data=sim_config.get('save_quantum_data', True)
            )
        else:
            self.config = config
            
        self.state = SimulationState.INITIALIZING
        
        # Simulation components (will be initialized in initialize())
        self.flight_dynamics = None
        self.autonomy_engine = None
        self.sensor_fusion = None
        self.gazebo_interface = None
        self.quantum_comms = None
        self.network_sim = None
        self.rf_propagation = None
        self.jamming_sim = None
        self.mesh_router = None
        
        # Simulation state
        self.simulation_time = 0.0
        self.simulation_start_time = None
        self.simulation_thread = None
        self.running = False
        
        # Data collection
        self.telemetry_data = []
        self.network_data = []
        self.quantum_data = []
        
        # Performance monitoring
        self.performance_metrics = {
            'start_time': time.time(),
            'real_time_factor': getattr(self.config, 'real_time_factor', 1.0),
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'simulation_rate': 0.0
        }
        
        # Threading
        self.lock = threading.Lock()
        
        logging.info("Simulation Orchestrator initialized")
        
    def initialize(self) -> bool:
        """Initialize all simulation components with mock fallbacks"""
        try:
            logging.info("Initializing simulation components...")
            
            # Initialize flight dynamics with fallback to mock
            try:
                if hasattr(HALEDynamics, '_mock_name'):  # It's a mock
                    self.flight_dynamics = MockComponent("flight_dynamics")
                    # Create mock state for compatibility
                    mock_state = type('MockState', (), {
                        'latitude': np.radians(0.0),
                        'longitude': np.radians(0.0),
                        'altitude': 20000.0,
                        'velocity_north': 50.0,
                        'velocity_east': 0.0,
                        'velocity_down': 0.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0,
                        'airspeed': 50.0, 'ground_speed': 50.0, 'heading': 0.0,
                        'flight_path_angle': 0.0,
                        'total_energy': 0.0, 'potential_energy': 0.0, 'kinetic_energy': 0.0,
                        'fuel_remaining': 100000.0, 'fuel_consumption_rate': 0.0,
                        'air_density': 0.0889, 'temperature': 216.65, 'pressure': 5474.9,
                        'wind_north': 0.0, 'wind_east': 0.0, 'wind_up': 0.0
                    })()
                    self.flight_dynamics.state = mock_state
                else:
                    # Real implementation
                    if self.config.aircraft_params:
                        self.flight_dynamics = HALEDynamics(self.config.aircraft_params)
                    else:
                        default_params = self._get_default_aircraft_params()
                        self.flight_dynamics = HALEDynamics(default_params)
                    
                    # Initialize with real state
                    initial_state = self._create_initial_aircraft_state()
                    self.flight_dynamics.initialize_state(initial_state)
                    
            except Exception as e:
                logging.warning(f"Flight dynamics initialization failed, using mock: {e}")
                self.flight_dynamics = MockComponent("flight_dynamics")
            
            # Initialize sensor fusion with fallback
            try:
                if hasattr(SensorFusion, '_mock_name'):
                    self.sensor_fusion = MockComponent("sensor_fusion")
                else:
                    self.sensor_fusion = SensorFusion()
                    # Initialize with fused state if real implementation
                    initial_fused_state = self._create_initial_fused_state()
                    self.sensor_fusion.initialize(initial_fused_state)
            except Exception as e:
                logging.warning(f"Sensor fusion initialization failed, using mock: {e}")
                self.sensor_fusion = MockComponent("sensor_fusion")
            
            # Initialize autonomy engine with fallback
            try:
                if hasattr(AutonomyEngine, '_mock_name'):
                    self.autonomy_engine = MockComponent("autonomy_engine")
                else:
                    self.autonomy_engine = AutonomyEngine()
                    # Load mission if waypoints provided
                    if self.config.waypoints:
                        mission = Mission(
                            id="MISSION_001",
                            type=self.config.mission_type,
                            waypoints=self.config.waypoints
                        )
                        self.autonomy_engine.load_mission(mission)
            except Exception as e:
                logging.warning(f"Autonomy engine initialization failed, using mock: {e}")
                self.autonomy_engine = MockComponent("autonomy_engine")
            
            # Initialize Gazebo interface with fallback
            try:
                if hasattr(GazeboInterface, '_mock_name'):
                    self.gazebo_interface = MockComponent("gazebo_interface")
                else:
                    gazebo_config = GazeboModelConfig(
                        model_name="hale_drone",
                        sdf_file="models/hale_drone.sdf",
                        initial_pose=(0.0, 0.0, 20000.0, 0.0, 0.0, 0.0)
                    )
                    self.gazebo_interface = GazeboInterface(gazebo_config)
                    self.gazebo_interface.load_model()
                    self.gazebo_interface.set_wind_conditions(self.config.wind_conditions)
            except Exception as e:
                logging.warning(f"Gazebo interface initialization failed, using mock: {e}")
                self.gazebo_interface = MockComponent("gazebo_interface")
            
            # Initialize quantum communications with fallback
            try:
                if hasattr(PQCHandshake, '_mock_name'):
                    self.quantum_comms = MockComponent("quantum_comms")
                else:
                    pqc_config = PQCConfiguration(self.config.security_level)
                    self.quantum_comms = PQCHandshake(pqc_config, "DRONE_001")
            except Exception as e:
                logging.warning(f"Quantum communications initialization failed, using mock: {e}")
                self.quantum_comms = MockComponent("quantum_comms")
            
            # Initialize network simulation with fallback
            try:
                if hasattr(NS3Wrapper, '_mock_name'):
                    self.network_sim = MockComponent("network_sim")
                    self.rf_propagation = MockComponent("rf_propagation")
                    self.jamming_sim = MockComponent("jamming_sim")
                    self.mesh_router = MockComponent("mesh_router")
                else:
                    self.network_sim = NS3Wrapper()
                    self.rf_propagation = RFPropagation()
                    self.jamming_sim = JammingModels()
                    self.mesh_router = MeshRouting()
                    
                    # Initialize network topology
                    self._initialize_network()
            except Exception as e:
                logging.warning(f"Network simulation initialization failed, using mock: {e}")
                self.network_sim = MockComponent("network_sim")
                self.rf_propagation = MockComponent("rf_propagation")
                self.jamming_sim = MockComponent("jamming_sim")
                self.mesh_router = MockComponent("mesh_router")
            
            self.state = SimulationState.STOPPED
            logging.info("All simulation components initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize simulation: {e}")
            self.state = SimulationState.ERROR
            return False
    
    def _get_default_aircraft_params(self):
        """Get default aircraft parameters"""
        return AircraftParameters(
            wingspan=35.0, wing_area=45.0, length=15.0,
            mass_empty=1200.0, mass_max_takeoff=2500.0,
            cl_alpha=5.0, cd0=0.02, oswald_efficiency=0.85, aspect_ratio=27.0,
            thrust_max=5000.0, specific_fuel_consumption=0.0001, propeller_efficiency=0.8,
            elevator_effectiveness=0.1, aileron_effectiveness=0.1, rudder_effectiveness=0.1,
            stall_speed=25.0, max_speed=120.0, service_ceiling=20000.0,
            range_max=500000.0, endurance_max=86400.0
        )
    
    def _create_initial_aircraft_state(self):
        """Create initial aircraft state"""
        return AircraftState(
            latitude=np.radians(0.0), longitude=np.radians(0.0), altitude=20000.0,
            velocity_north=50.0, velocity_east=0.0, velocity_down=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
            airspeed=50.0, ground_speed=50.0, heading=0.0, flight_path_angle=0.0,
            total_energy=0.0, potential_energy=0.0, kinetic_energy=0.0,
            fuel_remaining=100000.0, fuel_consumption_rate=0.0,
            air_density=0.0889, temperature=216.65, pressure=5474.9,
            wind_north=0.0, wind_east=0.0, wind_up=0.0
        )
    
    def _create_initial_fused_state(self):
        """Create initial fused state"""
        return FusedState(
            timestamp=time.time(),
            latitude=np.radians(0.0), longitude=np.radians(0.0), altitude=20000.0,
            velocity_north=50.0, velocity_east=0.0, velocity_down=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
            airspeed=50.0, wind_north=0.0, wind_east=0.0, wind_up=0.0,
            position_uncertainty=1.0, velocity_uncertainty=0.1, attitude_uncertainty=0.01
        )
            
    def _initialize_network(self):
        """Initialize network topology and components"""
        try:
            # Initialize network simulation
            if hasattr(self.network_sim, 'initialize'):
                self.network_sim.initialize()
            
            # Add drones to network
            for i in range(self.config.num_drones):
                drone_id = f"DRONE_{i+1:03d}"
                if hasattr(self.mesh_router, 'add_node'):
                    self.mesh_router.add_node(drone_id, (0.0, 0.0, 20000.0))
            
            # Add ground stations
            if self.config.ground_stations:
                for i, (lat, lon) in enumerate(self.config.ground_stations):
                    station_id = f"GROUND_{i+1:03d}"
                    if hasattr(self.mesh_router, 'add_node'):
                        self.mesh_router.add_node(station_id, (lat, lon, 100.0))
            
            # Setup jamming sources only if they are actually configured
            if self.config.jamming_sources and len(self.config.jamming_sources) > 0:
                for jammer in self.config.jamming_sources:
                    if hasattr(self.jamming_sim, 'add_jamming_source'):
                        jamming_source = JammingSource(
                            jammer_id=f"JAMMER_{len(getattr(self.jamming_sim, 'jammers', []))+1:03d}",
                            position=tuple(jammer['position']),
                            power=jammer['power'],
                            frequency_range=tuple(jammer['frequency_range']),
                            jamming_type=JammingType.CONTINUOUS
                        )
                        self.jamming_sim.add_jamming_source(jamming_source)
            else:
                # Clear any existing jammers to ensure clean state
                if hasattr(self.jamming_sim, 'clear_all_jammers'):
                    self.jamming_sim.clear_all_jammers()
                        
        except Exception as e:
            logging.error(f"Failed to initialize network: {e}")
            
    def start_simulation(self) -> bool:
        """Start the simulation"""
        if self.state != SimulationState.STOPPED:
            logging.warning("Simulation can only be started from STOPPED state")
            return False
            
        try:
            self.state = SimulationState.INITIALIZING
            self.simulation_start_time = time.time()
            self.simulation_time = 0.0
            self.running = True
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self._simulation_loop)
            self.simulation_thread.start()
            
            self.state = SimulationState.RUNNING
            logging.info("Simulation started")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start simulation: {e}")
            self.state = SimulationState.ERROR
            return False
            
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        self.state = SimulationState.STOPPING
        
        if self.simulation_thread:
            self.simulation_thread.join()
            
        self.state = SimulationState.STOPPED
        logging.info("Simulation stopped")
        
    def pause_simulation(self):
        """Pause the simulation"""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            if hasattr(self.gazebo_interface, 'pause_simulation'):
                self.gazebo_interface.pause_simulation()
            logging.info("Simulation paused")
            
    def resume_simulation(self):
        """Resume the simulation"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            if hasattr(self.gazebo_interface, 'resume_simulation'):
                self.gazebo_interface.resume_simulation()
            logging.info("Simulation resumed")
            
    def _simulation_loop(self):
        """Enhanced main simulation loop with your progress reporting"""
        last_time = time.time()
        last_progress_report = 0.0
        progress_interval = 0.5  # Report progress every 0.5 seconds (changed from 1.0)
        
        logging.info(f"Starting simulation loop - Duration: {self.config.duration}s, Timestep: {self.config.timestep}s")
        logging.info(f"Real-time factor: {self.config.real_time_factor}, Progress interval: {progress_interval}s")
        
        while self.running and self.simulation_time < self.config.duration:
            try:
                current_time = time.time()
                dt = current_time - last_time
                
                if self.state == SimulationState.RUNNING:
                    # Update simulation time
                    self.simulation_time += dt * self.config.real_time_factor
                    
                    # Check if simulation duration reached
                    if self.simulation_time >= self.config.duration:
                        logging.info(f"Simulation duration reached: {self.simulation_time:.2f}s / {self.config.duration}s")
                        break
                    
                    # Your enhanced progress reporting
                    if self.simulation_time - last_progress_report >= progress_interval:
                        self._report_progress()
                        last_progress_report = self.simulation_time
                    
                    # Update all components
                    self._update_flight_dynamics()
                    self._update_sensor_fusion()
                    self._update_autonomy_engine()
                    self._update_quantum_comms()
                    self._update_network_simulation()
                    self._update_gazebo_interface()
                    
                    # Collect data
                    self._collect_simulation_data()
                    
                    # Update performance metrics
                    self._update_performance_metrics(dt)
                    
                last_time = current_time
                
                # Sleep to maintain real-time factor
                sleep_time = self.config.timestep / self.config.real_time_factor - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logging.error(f"Error in simulation loop at time {self.simulation_time:.2f}s: {e}")
                self.state = SimulationState.ERROR
                break
        
        # Ensure simulation is properly stopped
        self.running = False
        self.state = SimulationState.STOPPED
        
        logging.info(f"Simulation loop completed - Final time: {self.simulation_time:.2f}s")
    
    def _report_progress(self):
        """Your enhanced progress reporting"""
        progress_percent = (self.simulation_time / self.config.duration) * 100
        
        # Get current status from all subsystems
        flight_status = self._get_flight_status()
        network_status = self._get_network_status()
        quantum_status = self._get_quantum_status()
        
        # Use print for better visibility
        print(f"\n=== SIMULATION PROGRESS: {progress_percent:.1f}% ({self.simulation_time:.1f}s/{self.config.duration:.1f}s) ===")
        print(f"Flight: {flight_status}")
        print(f"Network: {network_status}")
        print(f"Quantum: {quantum_status}")
        print("=" * 60)
        
        # Also log for file output
        logging.info(f"=== SIMULATION PROGRESS: {progress_percent:.1f}% ({self.simulation_time:.1f}s/{self.config.duration:.1f}s) ===")
        logging.info(f"Flight: {flight_status}")
        logging.info(f"Network: {network_status}")
        logging.info(f"Quantum: {quantum_status}")
        logging.info("=" * 60)
    
    def _get_flight_status(self) -> str:
        """Get current flight status summary"""
        if not self.flight_dynamics:
            return "No flight data"
        
        try:
            if hasattr(self.flight_dynamics, 'state') and self.flight_dynamics.state:
                state = self.flight_dynamics.state
                return f"Alt:{state.altitude:.0f}m, Speed:{state.airspeed:.1f}m/s, Fuel:{state.fuel_remaining:.1f}kg"
            elif hasattr(self.flight_dynamics, 'get_telemetry'):
                telemetry = self.flight_dynamics.get_telemetry()
                if telemetry:
                    pos = telemetry.get('position', {})
                    vel = telemetry.get('velocity', {})
                    energy = telemetry.get('energy', {})
                    return f"Alt:{pos.get('altitude', 0):.0f}m, Speed:{vel.get('airspeed', 0):.1f}m/s, Fuel:{energy.get('fuel_remaining', 0):.1f}kg"
            return "Mock flight system active"
        except Exception as e:
            return f"Flight error: {e}"
    
    def _get_network_status(self) -> str:
        """Get current network status summary"""
        if not self.network_sim:
            return "No network data"
        
        try:
            if hasattr(self.network_sim, 'get_network_status'):
                status = self.network_sim.get_network_status()
                # Handle both real and mock components
                if isinstance(status, dict):
                    return f"Nodes:{status.get('num_nodes', 0)}, Links:{status.get('num_links', 0)}, Topology:{status.get('topology_type', 'unknown')}"
                else:
                    return "Mock network system active"
            return "Mock network system active"
        except Exception as e:
            return f"Network error: {e}"
    
    def _get_quantum_status(self) -> str:
        """Get current quantum communications status"""
        if not self.quantum_comms:
            return "No quantum data"
        
        try:
            return "Mock quantum comms active"
        except Exception as e:
            return f"Quantum error: {e}"
        
    def _update_flight_dynamics(self):
        """Update flight dynamics simulation with mock compatibility"""
        if self.flight_dynamics:
            try:
                if hasattr(self.flight_dynamics, 'step'):
                    # Get control inputs from autonomy engine
                    if self.autonomy_engine and hasattr(self.autonomy_engine, 'update'):
                        current_state = self.flight_dynamics.get_telemetry() if hasattr(self.flight_dynamics, 'get_telemetry') else {}
                        environment_data = self._get_environment_data()
                        controls = self.autonomy_engine.update(current_state, environment_data)
                        
                        # Apply controls to flight dynamics
                        if hasattr(self.flight_dynamics, 'set_controls') and controls:
                            self.flight_dynamics.set_controls(
                                controls.get('throttle', 0.5),
                                controls.get('elevator', 0.0),
                                controls.get('aileron', 0.0),
                                controls.get('rudder', 0.0)
                            )
                    
                    # Step flight dynamics
                    self.flight_dynamics.step()
                else:
                    # Mock component
                    self.flight_dynamics.step()
            except Exception as e:
                logging.debug(f"Flight dynamics update error: {e}")
            
    def _update_sensor_fusion(self):
        """Update sensor fusion with mock compatibility"""
        if self.sensor_fusion and self.flight_dynamics:
            try:
                if hasattr(self.sensor_fusion, 'update_sensors'):
                    # Get true state from flight dynamics
                    true_state = self.flight_dynamics.get_telemetry() if hasattr(self.flight_dynamics, 'get_telemetry') else {}
                    
                    # Update sensors with true state
                    self.sensor_fusion.update_sensors(true_state)
                    
                    # Fuse sensor data
                    fused_state = self.sensor_fusion.fuse_sensors()
                else:
                    # Mock component
                    self.sensor_fusion.step()
            except Exception as e:
                logging.debug(f"Sensor fusion update error: {e}")
            
    def _update_autonomy_engine(self):
        """Update autonomy engine with mock compatibility"""
        if self.autonomy_engine and self.sensor_fusion:
            try:
                if hasattr(self.autonomy_engine, 'update'):
                    # Get fused state from sensor fusion
                    if hasattr(self.sensor_fusion, 'fused_state'):
                        fused_state = self.sensor_fusion.fused_state
                        if fused_state:
                            # Convert fused state to autonomy engine format
                            current_state = {
                                'position': {
                                    'latitude': np.degrees(fused_state.latitude),
                                    'longitude': np.degrees(fused_state.longitude),
                                    'altitude': fused_state.altitude
                                },
                                'velocity': {
                                    'airspeed': fused_state.airspeed,
                                    'ground_speed': np.sqrt(fused_state.velocity_north**2 + fused_state.velocity_east**2),
                                    'heading': np.degrees(fused_state.yaw)
                                },
                                'energy': {
                                    'fuel_remaining': self.flight_dynamics.state.fuel_remaining if hasattr(self.flight_dynamics, 'state') and self.flight_dynamics.state else 100000.0,
                                    'fuel_consumption_rate': 0.0
                                }
                            }
                            
                            environment_data = self._get_environment_data()
                            self.autonomy_engine.update(current_state, environment_data)
                else:
                    # Mock component
                    self.autonomy_engine.step()
            except Exception as e:
                logging.debug(f"Autonomy engine update error: {e}")
                
    def _update_quantum_comms(self):
        """Update quantum communications with mock compatibility"""
        if self.quantum_comms:
            try:
                if hasattr(self.quantum_comms, 'step'):
                    self.quantum_comms.step()
                # Simulate quantum key exchange for real implementation
                # This would be more complex in a real implementation
            except Exception as e:
                logging.debug(f"Quantum communications update error: {e}")
            
    def _update_network_simulation(self):
        """Update network simulation with mock compatibility"""
        if self.network_sim:
            try:
                if hasattr(self.network_sim, 'step'):
                    # Update network topology
                    self.network_sim.step(self.simulation_time)
                    
                    # Update RF propagation
                    if self.rf_propagation and self.flight_dynamics:
                        if hasattr(self.flight_dynamics, 'state') and self.flight_dynamics.state:
                            position = (self.flight_dynamics.state.latitude,
                                      self.flight_dynamics.state.longitude,
                                      self.flight_dynamics.state.altitude)
                            if hasattr(self.rf_propagation, 'update_position'):
                                self.rf_propagation.update_position(position)
                        elif hasattr(self.rf_propagation, 'step'):
                            self.rf_propagation.step()
                    
                    # Update jamming simulation
                    if self.jamming_sim and hasattr(self.jamming_sim, 'step'):
                        self.jamming_sim.step(self.simulation_time)
                else:
                    # Mock components
                    self.network_sim.step()
                    if self.rf_propagation and hasattr(self.rf_propagation, 'step'):
                        self.rf_propagation.step()
                    if self.jamming_sim and hasattr(self.jamming_sim, 'step'):
                        self.jamming_sim.step()
            except Exception as e:
                logging.debug(f"Network simulation update error: {e}")
            
    def _update_gazebo_interface(self):
        """Update Gazebo interface with mock compatibility"""
        if self.gazebo_interface and self.flight_dynamics:
            try:
                if hasattr(self.gazebo_interface, 'set_control_inputs'):
                    # Get current state from flight dynamics
                    telemetry = self.flight_dynamics.get_telemetry() if hasattr(self.flight_dynamics, 'get_telemetry') else {}
                    
                    # Update Gazebo model
                    if telemetry and 'controls' in telemetry:
                        controls = telemetry['controls']
                        self.gazebo_interface.set_control_inputs(
                            controls.get('throttle', 0.5),
                            controls.get('elevator', 0.0),
                            controls.get('aileron', 0.0),
                            controls.get('rudder', 0.0)
                        )
                else:
                    # Mock component
                    if hasattr(self.gazebo_interface, 'step'):
                        self.gazebo_interface.step()
            except Exception as e:
                logging.debug(f"Gazebo interface update error: {e}")
                
    def _get_environment_data(self) -> Dict[str, Any]:
        """Get environmental data for autonomy engine"""
        try:
            # Check jamming status safely
            jamming_active = False
            if self.jamming_sim and hasattr(self.jamming_sim, 'is_jamming_active'):
                try:
                    jamming_active = self.jamming_sim.is_jamming_active()
                except Exception as e:
                    logging.debug(f"Error checking jamming status: {e}")
                    jamming_active = False
            
            return {
                'wind_speed': np.sqrt(sum(x**2 for x in self.config.wind_conditions)),
                'wind_direction': np.arctan2(self.config.wind_conditions[1], self.config.wind_conditions[0]),
                'threats': {
                    'jamming_active': jamming_active
                }
            }
        except Exception as e:
            logging.debug(f"Environment data error: {e}")
            return {
                'wind_speed': 0.0,
                'wind_direction': 0.0,
                'threats': {'jamming_active': False}
            }
        
    def _collect_simulation_data(self):
        """Collect simulation data for analysis with mock compatibility"""
        if not self.config.save_telemetry:
            return
            
        try:
            # Collect telemetry data
            if self.flight_dynamics:
                if hasattr(self.flight_dynamics, 'get_telemetry'):
                    telemetry = self.flight_dynamics.get_telemetry()
                    if telemetry:
                        telemetry['simulation_time'] = self.simulation_time
                        self.telemetry_data.append(telemetry)
                
            # Collect network data
            if self.config.save_network_data and self.network_sim:
                try:
                    network_data = {
                        'timestamp': self.simulation_time,
                        'network_status': self.network_sim.get_network_status() if hasattr(self.network_sim, 'get_network_status') else {'mock': True},
                        'rf_conditions': self.rf_propagation.get_current_conditions() if self.rf_propagation and hasattr(self.rf_propagation, 'get_current_conditions') else {},
                        'jamming_status': self.jamming_sim.get_jamming_status() if self.jamming_sim and hasattr(self.jamming_sim, 'get_jamming_status') else {}
                    }
                    self.network_data.append(network_data)
                except Exception as e:
                    logging.debug(f"Network data collection error: {e}")
                
            # Collect quantum data
            if self.config.save_quantum_data and self.quantum_comms:
                try:
                    quantum_data = {
                        'timestamp': self.simulation_time,
                        'handshake_status': self.quantum_comms.get_handshake_metrics("session_001") if hasattr(self.quantum_comms, 'get_handshake_metrics') else {'mock': True}
                    }
                    self.quantum_data.append(quantum_data)
                except Exception as e:
                    logging.debug(f"Quantum data collection error: {e}")
                    
        except Exception as e:
            logging.debug(f"Data collection error: {e}")
            
    def _update_performance_metrics(self, dt: float):
        """Update performance monitoring metrics"""
        try:
            # Calculate simulation rate
            if dt > 0:
                self.performance_metrics['simulation_rate'] = 1.0 / dt
                
            # Calculate real-time factor
            if self.simulation_start_time:
                elapsed_real_time = time.time() - self.simulation_start_time
                if elapsed_real_time > 0:
                    self.performance_metrics['real_time_factor'] = self.simulation_time / elapsed_real_time
        except Exception as e:
            logging.debug(f"Performance metrics update error: {e}")
                
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        return {
            'state': self.state.value,
            'simulation_time': self.simulation_time,
            'duration': self.config.duration,
            'progress': (self.simulation_time / self.config.duration) * 100 if self.config.duration > 0 else 0,
            'performance_metrics': self.performance_metrics.copy(),
            'components': {
                'flight_dynamics': self.flight_dynamics is not None,
                'autonomy_engine': self.autonomy_engine is not None,
                'sensor_fusion': self.sensor_fusion is not None,
                'gazebo_interface': self.gazebo_interface is not None,
                'quantum_comms': self.quantum_comms is not None,
                'network_sim': self.network_sim is not None
            }
        }
        
    def save_simulation_data(self, filename: str = None):
        """Save collected simulation data"""
        if not filename:
            # Create output directory if it doesn't exist
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = output_dir / f"simulation_data_{int(time.time())}.json"
        else:
            # Ensure the directory for the specified filename exists
            file_path = Path(filename)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Prepare data with proper serialization
        data = {
            'config': {
                'duration': float(self.config.duration),
                'timestep': float(self.config.timestep),
                'real_time_factor': float(self.config.real_time_factor),
                'num_drones': int(self.config.num_drones)
            },
            'performance_metrics': {
                'start_time': float(self.performance_metrics['start_time']),
                'real_time_factor': float(self.performance_metrics['real_time_factor']),
                'cpu_usage': float(self.performance_metrics['cpu_usage']),
                'memory_usage': float(self.performance_metrics['memory_usage']),
                'simulation_rate': float(self.performance_metrics['simulation_rate'])
            }
        }
        
        # Safely add telemetry data
        if self.telemetry_data:
            try:
                # Convert telemetry data to serializable format
                serializable_telemetry = []
                for i, entry in enumerate(self.telemetry_data):
                    try:
                        if isinstance(entry, dict):
                            # Convert numpy types to native Python types
                            clean_entry = {}
                            for key, value in entry.items():
                                if hasattr(value, 'item'):  # numpy scalar
                                    clean_entry[key] = float(value.item())
                                elif isinstance(value, (list, tuple)):
                                    clean_entry[key] = [float(x) if hasattr(x, 'item') else float(x) for x in value]
                                elif isinstance(value, (int, float, str, bool)):
                                    clean_entry[key] = value
                                else:
                                    clean_entry[key] = str(value)
                            serializable_telemetry.append(clean_entry)
                        else:
                            serializable_telemetry.append(str(entry))
                    except Exception as e:
                        logging.warning(f"Could not serialize telemetry entry {i}: {e}")
                        serializable_telemetry.append(f"ERROR: {str(e)}")
                data['telemetry_data'] = serializable_telemetry
            except Exception as e:
                logging.warning(f"Could not serialize telemetry data: {e}")
                data['telemetry_data'] = []
        
        # Safely add network data
        if self.network_data:
            try:
                # Convert network data to serializable format
                serializable_network = []
                for i, entry in enumerate(self.network_data):
                    try:
                        if isinstance(entry, dict):
                            clean_entry = {}
                            for key, value in entry.items():
                                if hasattr(value, 'item'):  # numpy scalar
                                    clean_entry[key] = float(value.item())
                                elif isinstance(value, (list, tuple)):
                                    clean_entry[key] = [float(x) if hasattr(x, 'item') else float(x) for x in value]
                                elif isinstance(value, (int, float, str, bool)):
                                    clean_entry[key] = value
                                else:
                                    clean_entry[key] = str(value)
                            serializable_network.append(clean_entry)
                        else:
                            serializable_network.append(str(entry))
                    except Exception as e:
                        logging.warning(f"Could not serialize network entry {i}: {e}")
                        serializable_network.append(f"ERROR: {str(e)}")
                data['network_data'] = serializable_network
            except Exception as e:
                logging.warning(f"Could not serialize network data: {e}")
                data['network_data'] = []
        
        # Safely add quantum data
        if self.quantum_data:
            try:
                # Convert quantum data to serializable format
                serializable_quantum = []
                for i, entry in enumerate(self.quantum_data):
                    try:
                        if isinstance(entry, dict):
                            clean_entry = {}
                            for key, value in entry.items():
                                if hasattr(value, 'item'):  # numpy scalar
                                    clean_entry[key] = float(value.item())
                                elif isinstance(value, (list, tuple)):
                                    clean_entry[key] = [float(x) if hasattr(x, 'item') else float(x) for x in value]
                                elif isinstance(value, (int, float, str, bool)):
                                    clean_entry[key] = value
                                else:
                                    clean_entry[key] = str(value)
                            serializable_quantum.append(clean_entry)
                        else:
                            serializable_quantum.append(str(entry))
                    except Exception as e:
                        logging.warning(f"Could not serialize quantum entry {i}: {e}")
                        serializable_quantum.append(f"ERROR: {str(e)}")
                data['quantum_data'] = serializable_quantum
            except Exception as e:
                logging.warning(f"Could not serialize quantum data: {e}")
                data['quantum_data'] = []
        
        try:
            # Debug: Check data structure before saving
            logging.debug(f"Attempting to save data with keys: {list(data.keys())}")
            for key, value in data.items():
                logging.debug(f"Data key '{key}' type: {type(value)}, length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logging.info(f"Simulation data saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save simulation data: {e}")
            # Try saving a minimal version
            try:
                minimal_filename = Path(filename).parent / f"minimal_{Path(filename).name}"
                minimal_data = {
                    'config': data['config'],
                    'performance_metrics': data['performance_metrics'],
                    'error': f"Data serialization failed: {e}"
                }
                with open(minimal_filename, 'w') as f:
                    json.dump(minimal_data, f, indent=2, default=str)
                logging.info(f"Minimal simulation data saved to {minimal_filename}")
            except Exception as e2:
                logging.error(f"Failed to save even minimal data: {e2}")
    
    def setup_visualization(self):
        """Setup real-time visualization (disabled for stability)"""
        if not VISUALIZATION_AVAILABLE:
            logging.warning("Visualization not available - skipping setup")
            return False
        
        logging.info("Visualization disabled due to threading stability issues")
        return False
        
        # Original visualization code commented out for stability
        # try:
        #     # Create figure with subplots
        #     self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        #     self.fig.suptitle('Quantum HALE Drone Simulation - Real-time Status', fontsize=16)
        #     
        #     # Flight path plot
        #     self.ax1.set_title('Flight Path')
        #     self.ax1.set_title('Longitude (deg)')
        #     self.ax1.set_ylabel('Latitude (deg)')
        #     self.ax1.grid(True)
        #     
        #     # System status plot
        #     self.ax2.set_title('System Status')
        #     self.ax2.set_xlabel('Time (s)')
        #     self.ax2.set_ylabel('Value')
        #     self.ax2.grid(True)
        #     
        #     # Initialize data storage
        #     self.viz_data = {
        #         'times': [],
        #         'latitudes': [],
        #         'longitudes': [],
        #         'altitudes': [],
        #         'speeds': [],
        #         'fuel_levels': []
        #     }
        #     
        #     plt.ion()  # Turn on interactive mode
        #     plt.show()
        #     
        #     logging.info("Real-time visualization setup complete")
        #     return True
        #     
        # except Exception as e:
        #     logging.error(f"Failed to setup visualization: {e}")
        #     return False
    
    def update_visualization(self):
        """Update real-time visualization (disabled for stability)"""
        # Visualization disabled due to threading issues
        pass
    
    def close_visualization(self):
        """Close visualization window (disabled for stability)"""
        # Visualization disabled due to threading issues
        pass

    def _initialize_flight_sim(self):
        """Initialize flight simulation components"""
        try:
            # Initialize Gazebo interface with aircraft configuration
            from src.flight_sim.gazebo_interface import GazeboInterface, GazeboModelConfig
            
            # Get aircraft type from config or use default
            aircraft_type = getattr(self.config, 'aircraft_type', 'default')
            
            # Create Gazebo model configuration
            model_config = GazeboModelConfig(
                model_name="hale_drone",
                sdf_file="models/hale_drone.sdf",
                initial_pose=(0.0, 0.0, 20000.0, 0.0, 0.0, 0.0),  # x, y, z, roll, pitch, yaw
                aircraft_type=aircraft_type
            )
            
            # Initialize Gazebo interface
            self.gazebo_interface = GazeboInterface(model_config)
            
            # Load the model
            if self.gazebo_interface.load_model():
                logging.info("Flight simulation initialized successfully")
                
                # Set initial flight mode
                flight_mode = getattr(self.config, 'flight_mode', 'manual')
                self.gazebo_interface.set_flight_mode(flight_mode)
                
                # Set wind conditions if specified
                if hasattr(self.config, 'wind_conditions'):
                    self.gazebo_interface.set_wind_conditions(self.config.wind_conditions)
                    
            else:
                logging.error("Failed to initialize flight simulation")
                self.gazebo_interface = None
                
        except Exception as e:
            logging.error(f"Error initializing flight simulation: {e}")
            self.gazebo_interface = None

    def step(self):
        """Execute one simulation step"""
        if not self.running:
            return
            
        try:
            # Step flight simulation
            if self.gazebo_interface:
                self.gazebo_interface.step()
                
            # Step network simulation
            if self.network_sim and hasattr(self.network_sim, 'step'):
                self.network_sim.step()
                
            # Step quantum simulation
            if self.quantum_comms and hasattr(self.quantum_comms, 'step'):
                self.quantum_comms.step()
                
            # Step autonomy engine
            if self.autonomy_engine and hasattr(self.autonomy_engine, 'step'):
                self.autonomy_engine.step()
                
            # Collect telemetry data
            self._collect_telemetry()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Check for simulation completion
            if self.simulation_time >= self.config.duration:
                self.running = False
                logging.info("Simulation completed")
                
            self.simulation_time += self.config.timestep
            
        except Exception as e:
            logging.error(f"Error in simulation step: {e}")
            self.running = False

    def _collect_telemetry(self):
        """Collect telemetry data from all components"""
        try:
            # Get flight telemetry
            if self.gazebo_interface:
                flight_data = self.gazebo_interface.get_model_state()
                if flight_data:
                    self.telemetry_data['flight'] = flight_data
                    
                # Get sensor data
                sensor_data = self.gazebo_interface.get_sensor_data()
                if sensor_data:
                    self.telemetry_data['sensors'] = sensor_data
                    
                # Get flight controller status
                controller_status = self.gazebo_interface.flight_controller.get_controller_status()
                if controller_status:
                    self.telemetry_data['flight_controller'] = controller_status
                    
            # Get network telemetry
            if self.network_sim:
                try:
                    if hasattr(self.network_sim, 'get_network_status'):
                        network_status = self.network_sim.get_network_status()
                        if isinstance(network_status, dict):
                            self.telemetry_data['network'] = network_status
                        else:
                            self.telemetry_data['network'] = {'status': 'Mock network system active'}
                    else:
                        self.telemetry_data['network'] = {'status': 'Mock network system active'}
                except Exception as e:
                    self.telemetry_data['network'] = {'status': f'Network error: {e}'}
                    
            # Get quantum telemetry
            if self.quantum_comms:
                try:
                    if hasattr(self.quantum_comms, 'get_status'):
                        quantum_status = self.quantum_comms.get_status()
                        self.telemetry_data['quantum'] = quantum_status
                    else:
                        self.telemetry_data['quantum'] = {'status': 'Mock quantum system active'}
                except Exception as e:
                    self.telemetry_data['quantum'] = {'status': f'Quantum error: {e}'}
                    
            # Get autonomy telemetry
            if self.autonomy_engine:
                try:
                    if hasattr(self.autonomy_engine, 'get_status'):
                        autonomy_status = self.autonomy_engine.get_status()
                        self.telemetry_data['autonomy'] = autonomy_status
                    else:
                        self.telemetry_data['autonomy'] = {'status': 'Mock autonomy system active'}
                except Exception as e:
                    self.telemetry_data['autonomy'] = {'status': f'Autonomy error: {e}'}
                    
            # Add timestamp
            self.telemetry_data['timestamp'] = self.simulation_time
            
        except Exception as e:
            logging.error(f"Error collecting telemetry: {e}") 