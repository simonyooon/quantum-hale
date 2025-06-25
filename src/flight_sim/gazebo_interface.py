"""
HALE Drone Gazebo Interface
==========================

This module provides integration with the Gazebo simulation environment
for High-Altitude Long-Endurance drone visualization and physics simulation.

Author: Quantum HALE Development Team
License: MIT
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import yaml
from pathlib import Path

# Import flight dynamics
from .hale_dynamics import HALEDynamics, AircraftState, AircraftParameters
from .pid_controller import FlightController

# Gazebo/ROS2 imports (would be available in simulation environment)
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Pose, Twist, Vector3
    from sensor_msgs.msg import Imu, NavSatFix, FluidPressure
    from std_msgs.msg import Float64, Bool
    from gazebo_msgs.msg import ModelState, LinkState
    from gazebo_msgs.srv import SetModelState, GetModelState
    GAZEBO_AVAILABLE = True
except ImportError:
    # Mock classes for development without ROS2/Gazebo
    class Node:
        def __init__(self, name):
            self.name = name
    class Pose:
        def __init__(self):
            self.position = Vector3()
            self.orientation = Vector3()
    class Twist:
        def __init__(self):
            self.linear = Vector3()
            self.angular = Vector3()
    class Vector3:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
    GAZEBO_AVAILABLE = False


@dataclass
class GazeboModelConfig:
    """Configuration for Gazebo model"""
    model_name: str
    sdf_file: str
    initial_pose: Tuple[float, float, float, float, float, float]  # x, y, z, roll, pitch, yaw
    physics_engine: str = "ode"
    gravity: bool = True
    wind_model: str = "constant"
    wind_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    aircraft_type: str = "default"  # Aircraft type from config


class GazeboInterface(Node):
    """
    Enhanced Gazebo simulation interface for HALE drone
    
    Provides methods to control and monitor the drone in Gazebo,
    including physics simulation, sensor data, and visualization.
    Integrates with the flight dynamics and PID controllers.
    """
    
    def __init__(self, model_config: GazeboModelConfig):
        if GAZEBO_AVAILABLE:
            super().__init__('gazebo_interface')
        else:
            super().__init__('gazebo_interface')
            
        self.model_config = model_config
        self.model_name = model_config.model_name
        
        # Load aircraft parameters
        self.aircraft_params = self._load_aircraft_parameters(model_config.aircraft_type)
        
        # Initialize flight dynamics
        self.flight_dynamics = HALEDynamics(self.aircraft_params)
        
        # Initialize flight controller
        self.flight_controller = FlightController()
        
        # Simulation state
        self.simulation_time = 0.0
        self.simulation_running = False
        self.paused = False
        self.dt = 0.01  # Simulation time step
        
        # Model state
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.model_loaded = False
        
        # Sensor data
        self.imu_data = None
        self.gps_data = None
        self.air_data = None
        
        # Control inputs
        self.control_inputs = {
            'throttle': 0.0,
            'elevator': 0.0,
            'aileron': 0.0,
            'rudder': 0.0
        }
        
        # Flight mode
        self.flight_mode = "manual"  # manual, altitude_hold, heading_hold, auto
        
        # Publishers and subscribers
        self.publishers = {}
        self.subscribers = {}
        
        # Initialize ROS2/Gazebo interface
        if GAZEBO_AVAILABLE:
            self._setup_ros_interface()
        else:
            logging.warning("Gazebo/ROS2 not available - using enhanced mock interface")
            
        logging.info(f"Gazebo Interface initialized for model: {model_config.model_name}")
        
    def _load_aircraft_parameters(self, aircraft_type: str) -> AircraftParameters:
        """Load aircraft parameters from configuration file"""
        try:
            config_file = Path("configs/aircraft_parameters.yaml")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if aircraft_type in config:
                    params = config[aircraft_type]
                    return AircraftParameters(
                        wingspan=params['wingspan'],
                        wing_area=params['wing_area'],
                        length=params['length'],
                        mass_empty=params['mass_empty'],
                        mass_max_takeoff=params['mass_max_takeoff'],
                        cl_alpha=params['cl_alpha'],
                        cd0=params['cd0'],
                        oswald_efficiency=params['oswald_efficiency'],
                        aspect_ratio=params['aspect_ratio'],
                        thrust_max=params['thrust_max'],
                        specific_fuel_consumption=params['specific_fuel_consumption'],
                        propeller_efficiency=params['propeller_efficiency'],
                        elevator_effectiveness=params['elevator_effectiveness'],
                        aileron_effectiveness=params['aileron_effectiveness'],
                        rudder_effectiveness=params['rudder_effectiveness'],
                        stall_speed=params['stall_speed'],
                        max_speed=params['max_speed'],
                        service_ceiling=params['service_ceiling'],
                        range_max=params['range_max'],
                        endurance_max=params['endurance_max']
                    )
                else:
                    logging.warning(f"Aircraft type '{aircraft_type}' not found, using default")
                    return self._get_default_parameters()
            else:
                logging.warning("Aircraft parameters file not found, using default parameters")
                return self._get_default_parameters()
        except Exception as e:
            logging.error(f"Failed to load aircraft parameters: {e}")
            return self._get_default_parameters()
            
    def _get_default_parameters(self) -> AircraftParameters:
        """Get default aircraft parameters"""
        return AircraftParameters(
            wingspan=25.0,
            wing_area=35.0,
            length=12.0,
            mass_empty=1500.0,
            mass_max_takeoff=3000.0,
            cl_alpha=5.5,
            cd0=0.02,
            oswald_efficiency=0.85,
            aspect_ratio=18.0,
            thrust_max=6000.0,
            specific_fuel_consumption=0.0002,
            propeller_efficiency=0.8,
            elevator_effectiveness=0.12,
            aileron_effectiveness=0.10,
            rudder_effectiveness=0.08,
            stall_speed=25.0,
            max_speed=120.0,
            service_ceiling=20000.0,
            range_max=500000.0,
            endurance_max=86400.0
        )
        
    def _setup_ros_interface(self):
        """Setup ROS2 publishers and subscribers"""
        # Publishers
        self.publishers['model_state'] = self.create_publisher(
            ModelState, '/gazebo/set_model_state', 10
        )
        
        self.publishers['control'] = self.create_publisher(
            Twist, f'/{self.model_name}/cmd_vel', 10
        )
        
        # Subscribers
        self.subscribers['imu'] = self.create_subscription(
            Imu, f'/{self.model_name}/imu', self._imu_callback, 10
        )
        
        self.subscribers['gps'] = self.create_subscription(
            NavSatFix, f'/{self.model_name}/gps', self._gps_callback, 10
        )
        
        self.subscribers['air_data'] = self.create_subscription(
            FluidPressure, f'/{self.model_name}/air_data', self._air_data_callback, 10
        )
        
    def _imu_callback(self, msg):
        """Callback for IMU data"""
        self.imu_data = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'linear_acceleration': [msg.linear_acceleration.x, 
                                  msg.linear_acceleration.y, 
                                  msg.linear_acceleration.z],
            'angular_velocity': [msg.angular_velocity.x,
                               msg.angular_velocity.y,
                               msg.angular_velocity.z],
            'orientation': [msg.orientation.x,
                          msg.orientation.y,
                          msg.orientation.z,
                          msg.orientation.w]
        }
        
    def _gps_callback(self, msg):
        """Callback for GPS data"""
        self.gps_data = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
            'status': msg.status.status,
            'service': msg.status.service
        }
        
    def _air_data_callback(self, msg):
        """Callback for air data"""
        self.air_data = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'pressure': msg.fluid_pressure,
            'variance': msg.variance
        }
        
    def load_model(self) -> bool:
        """
        Load the HALE drone model into Gazebo
        
        Returns:
            True if model loaded successfully
        """
        if not GAZEBO_AVAILABLE:
            self.model_loaded = True
            logging.info("Mock model loaded (Gazebo not available)")
            return True
            
        try:
            # In a real implementation, this would spawn the model in Gazebo
            # using the SDF file and initial pose
            logging.info(f"Loading model {self.model_name} from {self.model_config.sdf_file}")
            
            # Set initial pose
            self.set_model_pose(*self.model_config.initial_pose)
            
            self.model_loaded = True
            logging.info(f"Model {self.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
            
    def set_model_pose(self, x: float, y: float, z: float, 
                      roll: float, pitch: float, yaw: float):
        """Set model pose in Gazebo"""
        if not GAZEBO_AVAILABLE:
            # Mock implementation
            self.current_pose.position.x = x
            self.current_pose.position.y = y
            self.current_pose.position.z = z
            self.current_pose.orientation.x = roll
            self.current_pose.orientation.y = pitch
            self.current_pose.orientation.z = yaw
            return
            
        try:
            # Create model state message
            model_state = ModelState()
            model_state.model_name = self.model_name
            model_state.pose.position.x = x
            model_state.pose.position.y = y
            model_state.pose.position.z = z
            
            # Convert Euler angles to quaternion
            quat = self._euler_to_quaternion(roll, pitch, yaw)
            model_state.pose.orientation.x = quat[0]
            model_state.pose.orientation.y = quat[1]
            model_state.pose.orientation.z = quat[2]
            model_state.pose.orientation.w = quat[3]
            
            # Publish model state
            self.publishers['model_state'].publish(model_state)
            
        except Exception as e:
            logging.error(f"Failed to set model pose: {e}")
            
    def set_control_inputs(self, throttle: float, elevator: float, 
                          aileron: float, rudder: float):
        """Set control inputs for the aircraft"""
        self.control_inputs['throttle'] = np.clip(throttle, 0.0, 1.0)
        self.control_inputs['elevator'] = np.clip(elevator, -0.5, 0.5)
        self.control_inputs['aileron'] = np.clip(aileron, -0.5, 0.5)
        self.control_inputs['rudder'] = np.clip(rudder, -0.3, 0.3)
        
        # Update flight dynamics
        self.flight_dynamics.set_controls(
            self.control_inputs['throttle'],
            self.control_inputs['elevator'],
            self.control_inputs['aileron'],
            self.control_inputs['rudder']
        )
        
        if not GAZEBO_AVAILABLE:
            # Mock physics update
            self._update_mock_physics()
        else:
            # Send control commands to Gazebo
            self._send_control_commands()
            
    def _send_control_commands(self):
        """Send control commands to Gazebo"""
        try:
            twist = Twist()
            # Convert control inputs to velocity commands
            # This is a simplified conversion - in reality would be more complex
            twist.linear.x = self.control_inputs['throttle'] * 50.0  # Forward velocity
            twist.angular.x = self.control_inputs['aileron'] * 2.0   # Roll rate
            twist.angular.y = self.control_inputs['elevator'] * 2.0  # Pitch rate
            twist.angular.z = self.control_inputs['rudder'] * 1.0    # Yaw rate
            
            self.publishers['control'].publish(twist)
            
        except Exception as e:
            logging.error(f"Failed to send control commands: {e}")
            
    def _update_mock_physics(self):
        """Update mock physics simulation"""
        # Step the flight dynamics
        self.flight_dynamics.step()
        
        # Update simulation time
        self.simulation_time += self.dt
        
        # Update mock sensor data
        telemetry = self.flight_dynamics.get_telemetry()
        if telemetry:
            # Update IMU data
            self.imu_data = {
                'timestamp': self.simulation_time,
                'linear_acceleration': [0.0, 0.0, -9.81],  # Mock acceleration
                'angular_velocity': [0.0, 0.0, 0.0],      # Mock angular velocity
                'orientation': [0.0, 0.0, 0.0, 1.0]       # Mock quaternion
            }
            
            # Update GPS data
            pos = telemetry.get('position', {})
            self.gps_data = {
                'timestamp': self.simulation_time,
                'latitude': pos.get('latitude', 0.0),
                'longitude': pos.get('longitude', 0.0),
                'altitude': pos.get('altitude', 0.0),
                'status': 0,
                'service': 1
            }
            
            # Update air data
            env = telemetry.get('environment', {})
            self.air_data = {
                'timestamp': self.simulation_time,
                'pressure': env.get('pressure', 101325.0),
                'variance': 0.0
            }
            
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state"""
        if not GAZEBO_AVAILABLE:
            # Return mock state from flight dynamics
            telemetry = self.flight_dynamics.get_telemetry()
            if telemetry:
                return {
                    'position': telemetry.get('position', {}),
                    'velocity': telemetry.get('velocity', {}),
                    'attitude': telemetry.get('attitude', {}),
                    'flight_phase': telemetry.get('flight_phase', 'unknown'),
                    'energy': telemetry.get('energy', {}),
                    'controls': telemetry.get('controls', {}),
                    'performance': telemetry.get('performance', {}),
                    'environment': telemetry.get('environment', {})
                }
            return {}
            
        try:
            # In real implementation, get state from Gazebo
            return {
                'position': {
                    'x': self.current_pose.position.x,
                    'y': self.current_pose.position.y,
                    'z': self.current_pose.position.z
                },
                'orientation': {
                    'roll': self.current_pose.orientation.x,
                    'pitch': self.current_pose.orientation.y,
                    'yaw': self.current_pose.orientation.z
                }
            }
        except Exception as e:
            logging.error(f"Failed to get model state: {e}")
            return {}
            
    def get_sensor_data(self) -> Dict[str, Any]:
        """Get current sensor data"""
        return {
            'imu': self.imu_data,
            'gps': self.gps_data,
            'air_data': self.air_data
        }
        
    def set_wind_conditions(self, wind_velocity: Tuple[float, float, float]):
        """Set wind conditions in the simulation"""
        self.model_config.wind_velocity = wind_velocity
        logging.info(f"Wind conditions set to: {wind_velocity}")
        
    def set_gravity(self, enabled: bool):
        """Enable or disable gravity"""
        self.model_config.gravity = enabled
        logging.info(f"Gravity {'enabled' if enabled else 'disabled'}")
        
    def pause_simulation(self):
        """Pause the simulation"""
        self.paused = True
        logging.info("Simulation paused")
        
    def resume_simulation(self):
        """Resume the simulation"""
        self.paused = False
        logging.info("Simulation resumed")
        
    def reset_simulation(self):
        """Reset the simulation"""
        self.simulation_time = 0.0
        self.paused = False
        
        # Reset flight dynamics
        if self.flight_dynamics.state:
            initial_state = AircraftState(
                latitude=np.radians(0.0),
                longitude=np.radians(0.0),
                altitude=20000.0,
                velocity_north=50.0,
                velocity_east=0.0,
                velocity_down=0.0,
                roll=0.0, pitch=0.0, yaw=0.0,
                roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
                airspeed=50.0, ground_speed=50.0, heading=0.0, flight_path_angle=0.0,
                total_energy=0.0, potential_energy=0.0, kinetic_energy=0.0,
                fuel_remaining=100000.0, fuel_consumption_rate=0.0,
                air_density=0.0889, temperature=216.65, pressure=5474.9,
                wind_north=0.0, wind_east=0.0, wind_up=0.0
            )
            self.flight_dynamics.initialize_state(initial_state)
            
        logging.info("Simulation reset")
        
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> List[float]:
        """Convert Euler angles to quaternion"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        q = [0] * 4
        q[0] = cy * cp * cr + sy * sp * sr
        q[1] = cy * cp * sr - sy * sp * cr
        q[2] = sy * cp * sr + cy * sp * cr
        q[3] = sy * cp * cr - cy * sp * sr
        
        return q
        
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get simulation status"""
        return {
            'simulation_time': self.simulation_time,
            'model_loaded': self.model_loaded,
            'simulation_running': self.simulation_running,
            'paused': self.paused,
            'flight_mode': self.flight_mode,
            'aircraft_type': self.model_config.aircraft_type,
            'model_state': self.get_model_state(),
            'sensor_data': self.get_sensor_data(),
            'flight_controller_status': self.flight_controller.get_controller_status()
        }
        
    def set_flight_mode(self, mode: str):
        """Set flight control mode"""
        self.flight_mode = mode
        self.flight_controller.set_control_mode(mode)
        logging.info(f"Flight mode set to: {mode}")
        
    def step(self):
        """Execute one simulation step"""
        if self.paused:
            return
            
        # Update flight controller if in automatic mode
        if self.flight_mode != "manual":
            state_dict = self.get_model_state()
            if state_dict:
                controls = self.flight_controller.update(state_dict, self.dt)
                self.set_control_inputs(
                    controls['throttle'],
                    controls['elevator'],
                    controls['aileron'],
                    controls['rudder']
                )
        
        # Update physics
        if not GAZEBO_AVAILABLE:
            self._update_mock_physics()
            
        self.simulation_time += self.dt 