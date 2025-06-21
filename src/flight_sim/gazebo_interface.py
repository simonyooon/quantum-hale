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


class GazeboInterface(Node):
    """
    Gazebo simulation interface for HALE drone
    
    Provides methods to control and monitor the drone in Gazebo,
    including physics simulation, sensor data, and visualization.
    """
    
    def __init__(self, model_config: GazeboModelConfig):
        if GAZEBO_AVAILABLE:
            super().__init__('gazebo_interface')
        else:
            super().__init__('gazebo_interface')
            
        self.model_config = model_config
        self.model_name = model_config.model_name
        
        # Simulation state
        self.simulation_time = 0.0
        self.simulation_running = False
        self.paused = False
        
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
        
        # Publishers and subscribers
        self.publishers = {}
        self.subscribers = {}
        
        # Initialize ROS2/Gazebo interface
        if GAZEBO_AVAILABLE:
            self._setup_ros_interface()
        else:
            logging.warning("Gazebo/ROS2 not available - using mock interface")
            
        logging.info(f"Gazebo Interface initialized for model: {model_config.model_name}")
        
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
            q = self._euler_to_quaternion(roll, pitch, yaw)
            model_state.pose.orientation.x = q[0]
            model_state.pose.orientation.y = q[1]
            model_state.pose.orientation.z = q[2]
            model_state.pose.orientation.w = q[3]
            
            # Publish model state
            self.publishers['model_state'].publish(model_state)
            
        except Exception as e:
            logging.error(f"Failed to set model pose: {e}")
            
    def set_control_inputs(self, throttle: float, elevator: float, 
                          aileron: float, rudder: float):
        """Set control inputs for the drone"""
        self.control_inputs = {
            'throttle': np.clip(throttle, 0.0, 1.0),
            'elevator': np.clip(elevator, -0.5, 0.5),
            'aileron': np.clip(aileron, -0.5, 0.5),
            'rudder': np.clip(rudder, -0.5, 0.5)
        }
        
        if not GAZEBO_AVAILABLE:
            # Mock control response
            self._update_mock_physics()
            return
            
        try:
            # Convert control inputs to velocity commands
            # This is a simplified conversion - real implementation would be more complex
            linear_x = throttle * 50.0  # Forward velocity based on throttle
            angular_z = rudder * 0.5    # Yaw rate based on rudder
            
            # Create twist message
            twist = Twist()
            twist.linear.x = linear_x
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = angular_z
            
            # Publish control command
            self.publishers['control'].publish(twist)
            
        except Exception as e:
            logging.error(f"Failed to set control inputs: {e}")
            
    def _update_mock_physics(self):
        """Update mock physics simulation"""
        # Simple physics update for development without Gazebo
        dt = 0.01
        
        # Update position based on velocity
        self.current_pose.position.x += self.current_twist.linear.x * dt
        self.current_pose.position.y += self.current_twist.linear.y * dt
        self.current_pose.position.z += self.current_twist.linear.z * dt
        
        # Update orientation based on angular velocity
        self.current_pose.orientation.x += self.current_twist.angular.x * dt
        self.current_pose.orientation.y += self.current_twist.angular.y * dt
        self.current_pose.orientation.z += self.current_twist.angular.z * dt
        
        # Update velocity based on control inputs
        self.current_twist.linear.x = self.control_inputs['throttle'] * 50.0
        self.current_twist.angular.z = self.control_inputs['rudder'] * 0.5
        
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state from Gazebo"""
        if not GAZEBO_AVAILABLE:
            # Return mock state
            return {
                'timestamp': time.time(),
                'position': {
                    'x': self.current_pose.position.x,
                    'y': self.current_pose.position.y,
                    'z': self.current_pose.position.z
                },
                'orientation': {
                    'roll': self.current_pose.orientation.x,
                    'pitch': self.current_pose.orientation.y,
                    'yaw': self.current_pose.orientation.z
                },
                'velocity': {
                    'linear': {
                        'x': self.current_twist.linear.x,
                        'y': self.current_twist.linear.y,
                        'z': self.current_twist.linear.z
                    },
                    'angular': {
                        'x': self.current_twist.angular.x,
                        'y': self.current_twist.angular.y,
                        'z': self.current_twist.angular.z
                    }
                }
            }
            
        try:
            # In real implementation, this would query Gazebo for model state
            # For now, return a placeholder
            return {
                'timestamp': time.time(),
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'orientation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'velocity': {
                    'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to get model state: {e}")
            return {}
            
    def get_sensor_data(self) -> Dict[str, Any]:
        """Get sensor data from Gazebo"""
        sensor_data = {}
        
        if self.imu_data:
            sensor_data['imu'] = self.imu_data
            
        if self.gps_data:
            sensor_data['gps'] = self.gps_data
            
        if self.air_data:
            sensor_data['air_data'] = self.air_data
            
        return sensor_data
        
    def set_wind_conditions(self, wind_velocity: Tuple[float, float, float]):
        """Set wind conditions in Gazebo"""
        if not GAZEBO_AVAILABLE:
            logging.info(f"Mock wind conditions set: {wind_velocity}")
            return
            
        try:
            # In real implementation, this would set wind parameters in Gazebo
            logging.info(f"Setting wind conditions: {wind_velocity}")
            
        except Exception as e:
            logging.error(f"Failed to set wind conditions: {e}")
            
    def set_gravity(self, enabled: bool):
        """Enable or disable gravity in Gazebo"""
        if not GAZEBO_AVAILABLE:
            logging.info(f"Mock gravity {'enabled' if enabled else 'disabled'}")
            return
            
        try:
            # In real implementation, this would configure gravity in Gazebo
            logging.info(f"Gravity {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            logging.error(f"Failed to set gravity: {e}")
            
    def pause_simulation(self):
        """Pause Gazebo simulation"""
        self.paused = True
        logging.info("Simulation paused")
        
    def resume_simulation(self):
        """Resume Gazebo simulation"""
        self.paused = False
        logging.info("Simulation resumed")
        
    def reset_simulation(self):
        """Reset Gazebo simulation"""
        if not GAZEBO_AVAILABLE:
            # Reset mock state
            self.current_pose = Pose()
            self.current_twist = Twist()
            self.control_inputs = {'throttle': 0.0, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
            logging.info("Mock simulation reset")
            return
            
        try:
            # In real implementation, this would reset the Gazebo simulation
            logging.info("Simulation reset")
            
        except Exception as e:
            logging.error(f"Failed to reset simulation: {e}")
            
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> List[float]:
        """Convert Euler angles to quaternion"""
        # Simple conversion - in production use proper quaternion math
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = sr * cp * cy - cr * sp * sy
        q[1] = cr * sp * cy + sr * cp * sy
        q[2] = cr * cp * sy - sr * sp * cy
        q[3] = cr * cp * cy + sr * sp * sy
        
        return q
        
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get simulation status"""
        return {
            'model_loaded': self.model_loaded,
            'simulation_running': self.simulation_running,
            'paused': self.paused,
            'simulation_time': self.simulation_time,
            'gazebo_available': GAZEBO_AVAILABLE,
            'control_inputs': self.control_inputs.copy()
        } 