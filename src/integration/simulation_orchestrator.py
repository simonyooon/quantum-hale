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

# Import simulation components
from flight_sim.hale_dynamics import HALEDynamics, AircraftState, AircraftParameters
from flight_sim.autonomy_engine import AutonomyEngine, Mission, MissionType, Waypoint
from flight_sim.sensor_fusion import SensorFusion, FusedState
from flight_sim.gazebo_interface import GazeboInterface, GazeboModelConfig
from quantum_comms.pqc_handshake import PQCHandshake, PQCConfiguration, SecurityLevel
from network_sim.ns3_wrapper import NS3Wrapper
from network_sim.rf_propagation import RFPropagation
from network_sim.jamming_models import JammingModels, JammingSource, JammingType
from network_sim.mesh_routing import MeshRouting


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
    aircraft_params: AircraftParameters = None
    
    # Mission configuration
    mission_type: MissionType = MissionType.ISR_PATROL
    waypoints: List[Waypoint] = None
    
    # Network configuration
    network_topology: str = "mesh"
    num_drones: int = 3
    ground_stations: List[Tuple[float, float]] = None
    
    # Quantum configuration
    security_level: SecurityLevel = SecurityLevel.CATEGORY_3
    enable_qkd: bool = True
    
    # Environment configuration
    wind_conditions: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    jamming_sources: List[Dict[str, Any]] = None
    
    # Output configuration
    output_directory: str = "simulation_results"
    save_telemetry: bool = True
    save_network_data: bool = True
    save_quantum_data: bool = True


class SimulationOrchestrator:
    """
    Main simulation orchestrator for Quantum HALE Drone System
    
    Coordinates all simulation components and manages the overall
    simulation execution flow.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = SimulationState.INITIALIZING
        
        # Simulation components
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
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'simulation_rate': 0.0,
            'real_time_factor': 0.0
        }
        
        # Threading
        self.lock = threading.Lock()
        
        logging.info("Simulation Orchestrator initialized")
        
    def initialize(self) -> bool:
        """Initialize all simulation components"""
        try:
            logging.info("Initializing simulation components...")
            
            # Initialize flight dynamics
            if self.config.aircraft_params:
                self.flight_dynamics = HALEDynamics(self.config.aircraft_params)
            else:
                # Use default aircraft parameters
                default_params = AircraftParameters(
                    wingspan=35.0,
                    wing_area=45.0,
                    length=15.0,
                    mass_empty=1200.0,
                    mass_max_takeoff=2500.0,
                    cl_alpha=5.0,
                    cd0=0.02,
                    oswald_efficiency=0.85,
                    aspect_ratio=27.0,
                    thrust_max=5000.0,
                    specific_fuel_consumption=0.0001,
                    propeller_efficiency=0.8,
                    elevator_effectiveness=0.1,
                    aileron_effectiveness=0.1,
                    rudder_effectiveness=0.1,
                    stall_speed=25.0,
                    max_speed=120.0,
                    service_ceiling=20000.0,
                    range_max=500000.0,
                    endurance_max=86400.0
                )
                self.flight_dynamics = HALEDynamics(default_params)
            
            # Initialize sensor fusion
            self.sensor_fusion = SensorFusion()
            
            # Initialize autonomy engine
            self.autonomy_engine = AutonomyEngine()
            
            # Initialize Gazebo interface
            gazebo_config = GazeboModelConfig(
                model_name="hale_drone",
                sdf_file="models/hale_drone.sdf",
                initial_pose=(0.0, 0.0, 20000.0, 0.0, 0.0, 0.0)
            )
            self.gazebo_interface = GazeboInterface(gazebo_config)
            
            # Initialize quantum communications
            pqc_config = PQCConfiguration(self.config.security_level)
            self.quantum_comms = PQCHandshake(pqc_config, "DRONE_001")
            
            # Initialize network simulation
            self.network_sim = NS3Wrapper()
            self.rf_propagation = RFPropagation()
            self.jamming_sim = JammingModels()
            self.mesh_router = MeshRouting()
            
            # Initialize aircraft state
            initial_state = AircraftState(
                latitude=np.radians(0.0),
                longitude=np.radians(0.0),
                altitude=20000.0,
                velocity_north=50.0,
                velocity_east=0.0,
                velocity_down=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                roll_rate=0.0,
                pitch_rate=0.0,
                yaw_rate=0.0,
                airspeed=50.0,
                ground_speed=50.0,
                heading=0.0,
                flight_path_angle=0.0,
                total_energy=0.0,
                potential_energy=0.0,
                kinetic_energy=0.0,
                fuel_remaining=100000.0,
                fuel_consumption_rate=0.0,
                air_density=0.0889,
                temperature=216.65,
                pressure=5474.9,
                wind_north=0.0,
                wind_east=0.0,
                wind_up=0.0
            )
            
            self.flight_dynamics.initialize_state(initial_state)
            
            # Initialize sensor fusion
            fused_state = FusedState(
                timestamp=time.time(),
                latitude=initial_state.latitude,
                longitude=initial_state.longitude,
                altitude=initial_state.altitude,
                velocity_north=initial_state.velocity_north,
                velocity_east=initial_state.velocity_east,
                velocity_down=initial_state.velocity_down,
                roll=initial_state.roll,
                pitch=initial_state.pitch,
                yaw=initial_state.yaw,
                roll_rate=initial_state.roll_rate,
                pitch_rate=initial_state.pitch_rate,
                yaw_rate=initial_state.yaw_rate,
                airspeed=initial_state.airspeed,
                wind_north=initial_state.wind_north,
                wind_east=initial_state.wind_east,
                wind_up=initial_state.wind_up,
                position_uncertainty=1.0,
                velocity_uncertainty=0.1,
                attitude_uncertainty=0.01
            )
            self.sensor_fusion.initialize(fused_state)
            
            # Load mission
            if self.config.waypoints:
                mission = Mission(
                    id="MISSION_001",
                    type=self.config.mission_type,
                    waypoints=self.config.waypoints
                )
                self.autonomy_engine.load_mission(mission)
            
            # Load Gazebo model
            self.gazebo_interface.load_model()
            
            # Set wind conditions
            self.gazebo_interface.set_wind_conditions(self.config.wind_conditions)
            
            # Initialize network topology
            self._initialize_network()
            
            self.state = SimulationState.STOPPED
            logging.info("All simulation components initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize simulation: {e}")
            self.state = SimulationState.ERROR
            return False
            
    def _initialize_network(self):
        """Initialize network topology and components"""
        try:
            # Initialize network simulation
            self.network_sim.initialize()
            
            # Add drones to network
            for i in range(self.config.num_drones):
                drone_id = f"DRONE_{i+1:03d}"
                self.mesh_router.add_node(drone_id, (0.0, 0.0, 20000.0))
            
            # Add ground stations
            if self.config.ground_stations:
                for i, (lat, lon) in enumerate(self.config.ground_stations):
                    station_id = f"GROUND_{i+1:03d}"
                    self.mesh_router.add_node(station_id, (lat, lon, 100.0))
            
            # Setup jamming sources
            if self.config.jamming_sources:
                for jammer in self.config.jamming_sources:
                    jamming_source = JammingSource(
                        jammer_id=f"JAMMER_{len(self.jamming_sim.jammers)+1:03d}",
                        position=tuple(jammer['position']),
                        power=jammer['power'],
                        frequency_range=tuple(jammer['frequency_range']),
                        jamming_type=JammingType.CONTINUOUS
                    )
                    self.jamming_sim.add_jamming_source(jamming_source)
                    
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
            self.gazebo_interface.pause_simulation()
            logging.info("Simulation paused")
            
    def resume_simulation(self):
        """Resume the simulation"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            self.gazebo_interface.resume_simulation()
            logging.info("Simulation resumed")
            
    def _simulation_loop(self):
        """Main simulation loop"""
        last_time = time.time()
        
        while self.running and self.simulation_time < self.config.duration:
            try:
                current_time = time.time()
                dt = current_time - last_time
                
                if self.state == SimulationState.RUNNING:
                    # Update simulation time
                    self.simulation_time += dt * self.config.real_time_factor
                    
                    # Update flight dynamics
                    self._update_flight_dynamics()
                    
                    # Update sensor fusion
                    self._update_sensor_fusion()
                    
                    # Update autonomy engine
                    self._update_autonomy_engine()
                    
                    # Update quantum communications
                    self._update_quantum_comms()
                    
                    # Update network simulation
                    self._update_network_simulation()
                    
                    # Update Gazebo interface
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
                logging.error(f"Error in simulation loop: {e}")
                self.state = SimulationState.ERROR
                break
                
    def _update_flight_dynamics(self):
        """Update flight dynamics simulation"""
        if self.flight_dynamics:
            # Get control inputs from autonomy engine
            if self.autonomy_engine:
                current_state = self.flight_dynamics.get_telemetry()
                environment_data = self._get_environment_data()
                controls = self.autonomy_engine.update(current_state, environment_data)
                
                # Apply controls to flight dynamics
                self.flight_dynamics.set_controls(
                    controls['throttle'],
                    controls['elevator'],
                    controls['aileron'],
                    controls['rudder']
                )
            
            # Step flight dynamics
            self.flight_dynamics.step()
            
    def _update_sensor_fusion(self):
        """Update sensor fusion"""
        if self.sensor_fusion and self.flight_dynamics:
            # Get true state from flight dynamics
            true_state = self.flight_dynamics.get_telemetry()
            
            # Update sensors with true state
            self.sensor_fusion.update_sensors(true_state)
            
            # Fuse sensor data
            fused_state = self.sensor_fusion.fuse_sensors()
            
    def _update_autonomy_engine(self):
        """Update autonomy engine"""
        if self.autonomy_engine and self.sensor_fusion:
            # Get fused state from sensor fusion
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
                        'fuel_remaining': self.flight_dynamics.state.fuel_remaining if self.flight_dynamics.state else 0.0,
                        'fuel_consumption_rate': 0.0
                    }
                }
                
                environment_data = self._get_environment_data()
                self.autonomy_engine.update(current_state, environment_data)
                
    def _update_quantum_comms(self):
        """Update quantum communications"""
        if self.quantum_comms:
            # Simulate quantum key exchange
            # This would be more complex in a real implementation
            pass
            
    def _update_network_simulation(self):
        """Update network simulation"""
        if self.network_sim:
            # Update network topology
            self.network_sim.step(self.simulation_time)
            
            # Update RF propagation
            if self.flight_dynamics and self.flight_dynamics.state:
                position = (self.flight_dynamics.state.latitude,
                          self.flight_dynamics.state.longitude,
                          self.flight_dynamics.state.altitude)
                self.rf_propagation.update_position(position)
                
            # Update jamming simulation
            self.jamming_sim.step(self.simulation_time)
            
    def _update_gazebo_interface(self):
        """Update Gazebo interface"""
        if self.gazebo_interface and self.flight_dynamics:
            # Get current state from flight dynamics
            telemetry = self.flight_dynamics.get_telemetry()
            
            # Update Gazebo model
            if telemetry:
                self.gazebo_interface.set_control_inputs(
                    telemetry['controls']['throttle'],
                    telemetry['controls']['elevator'],
                    telemetry['controls']['aileron'],
                    telemetry['controls']['rudder']
                )
                
    def _get_environment_data(self) -> Dict[str, Any]:
        """Get environmental data for autonomy engine"""
        return {
            'wind_speed': np.sqrt(sum(x**2 for x in self.config.wind_conditions)),
            'wind_direction': np.arctan2(self.config.wind_conditions[1], self.config.wind_conditions[0]),
            'threats': {
                'jamming_active': self.jamming_sim.is_jamming_active() if self.jamming_sim else False
            }
        }
        
    def _collect_simulation_data(self):
        """Collect simulation data for analysis"""
        if not self.config.save_telemetry:
            return
            
        # Collect telemetry data
        if self.flight_dynamics:
            telemetry = self.flight_dynamics.get_telemetry()
            telemetry['simulation_time'] = self.simulation_time
            self.telemetry_data.append(telemetry)
            
        # Collect network data
        if self.config.save_network_data and self.network_sim:
            network_data = {
                'timestamp': self.simulation_time,
                'network_status': self.network_sim.get_network_status(),
                'rf_conditions': self.rf_propagation.get_current_conditions() if self.rf_propagation else {},
                'jamming_status': self.jamming_sim.get_jamming_status() if self.jamming_sim else {}
            }
            self.network_data.append(network_data)
            
        # Collect quantum data
        if self.config.save_quantum_data and self.quantum_comms:
            quantum_data = {
                'timestamp': self.simulation_time,
                'handshake_status': self.quantum_comms.get_handshake_metrics("session_001") if hasattr(self.quantum_comms, 'get_handshake_metrics') else {}
            }
            self.quantum_data.append(quantum_data)
            
    def _update_performance_metrics(self, dt: float):
        """Update performance monitoring metrics"""
        # Calculate simulation rate
        if dt > 0:
            self.performance_metrics['simulation_rate'] = 1.0 / dt
            
        # Calculate real-time factor
        if self.simulation_start_time:
            elapsed_real_time = time.time() - self.simulation_start_time
            if elapsed_real_time > 0:
                self.performance_metrics['real_time_factor'] = self.simulation_time / elapsed_real_time
                
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
            filename = f"simulation_data_{int(time.time())}.yaml"
            
        data = {
            'config': self.config.__dict__,
            'telemetry_data': self.telemetry_data,
            'network_data': self.network_data,
            'quantum_data': self.quantum_data,
            'performance_metrics': self.performance_metrics
        }
        
        try:
            with open(filename, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            logging.info(f"Simulation data saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save simulation data: {e}") 