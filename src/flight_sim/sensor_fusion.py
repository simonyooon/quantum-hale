"""
HALE Drone Sensor Fusion
=======================

This module implements multi-sensor data fusion and processing for
High-Altitude Long-Endurance drones, including GPS, IMU, air data,
and environmental sensors.

Author: Quantum HALE Development Team
License: MIT
"""

import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import math
from scipy.spatial.transform import Rotation
from scipy.linalg import inv, cholesky


class SensorType(Enum):
    """Sensor types for HALE drone"""
    GPS = "gps"
    IMU = "imu"
    AIR_DATA = "air_data"
    MAGNETOMETER = "magnetometer"
    BAROMETER = "barometer"
    TEMPERATURE = "temperature"
    WIND = "wind"
    CAMERA = "camera"
    RADAR = "radar"


@dataclass
class SensorReading:
    """Individual sensor reading"""
    sensor_type: SensorType
    timestamp: float
    data: Dict[str, Any]
    uncertainty: Dict[str, float]
    valid: bool = True


@dataclass
class FusedState:
    """Fused sensor state estimate"""
    timestamp: float
    
    # Position (ECEF)
    latitude: float  # radians
    longitude: float  # radians
    altitude: float  # meters MSL
    
    # Velocity (NED frame)
    velocity_north: float  # m/s
    velocity_east: float   # m/s
    velocity_down: float   # m/s
    
    # Attitude (Euler angles)
    roll: float    # radians
    pitch: float   # radians
    yaw: float     # radians
    
    # Angular rates (body frame)
    roll_rate: float   # rad/s
    pitch_rate: float  # rad/s
    yaw_rate: float    # rad/s
    
    # Environmental
    airspeed: float    # m/s
    wind_north: float  # m/s
    wind_east: float   # m/s
    wind_up: float     # m/s
    
    # Uncertainty (covariance matrix diagonal)
    position_uncertainty: float  # meters
    velocity_uncertainty: float  # m/s
    attitude_uncertainty: float  # radians


class KalmanFilter:
    """Extended Kalman Filter for sensor fusion"""
    
    def __init__(self, initial_state: FusedState):
        self.state = initial_state
        
        # State vector: [lat, lon, alt, vn, ve, vd, roll, pitch, yaw, wr, wp, wy]
        self.x = np.zeros(12)
        self.x[0] = initial_state.latitude
        self.x[1] = initial_state.longitude
        self.x[2] = initial_state.altitude
        self.x[3] = initial_state.velocity_north
        self.x[4] = initial_state.velocity_east
        self.x[5] = initial_state.velocity_down
        self.x[6] = initial_state.roll
        self.x[7] = initial_state.pitch
        self.x[8] = initial_state.yaw
        self.x[9] = initial_state.roll_rate
        self.x[10] = initial_state.pitch_rate
        self.x[11] = initial_state.yaw_rate
        
        # State covariance matrix
        self.P = np.eye(12) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(12) * 0.01
        
        # Measurement noise covariance (will be updated per sensor)
        self.R = np.eye(6) * 1.0
        
        # Time step
        self.dt = 0.01
        
    def predict(self):
        """Predict step of EKF"""
        # State transition (simplified - in production use proper dynamics)
        F = np.eye(12)
        F[0, 3] = self.dt  # lat += vn * dt
        F[1, 4] = self.dt  # lon += ve * dt
        F[2, 5] = self.dt  # alt += vd * dt
        F[6, 9] = self.dt  # roll += wr * dt
        F[7, 10] = self.dt  # pitch += wp * dt
        F[8, 11] = self.dt  # yaw += wy * dt
        
        # Predict state
        self.x = F @ self.x
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, measurement: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Update step of EKF"""
        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ inv(S)
        
        # Update state
        y = measurement - H @ self.x  # Innovation
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(12)
        self.P = (I - K @ H) @ self.P
        
    def get_state(self) -> FusedState:
        """Get current fused state"""
        return FusedState(
            timestamp=time.time(),
            latitude=self.x[0],
            longitude=self.x[1],
            altitude=self.x[2],
            velocity_north=self.x[3],
            velocity_east=self.x[4],
            velocity_down=self.x[5],
            roll=self.x[6],
            pitch=self.x[7],
            yaw=self.x[8],
            roll_rate=self.x[9],
            pitch_rate=self.x[10],
            yaw_rate=self.x[11],
            airspeed=np.sqrt(self.x[3]**2 + self.x[4]**2 + self.x[5]**2),
            wind_north=0.0,  # Would be estimated from air data
            wind_east=0.0,
            wind_up=0.0,
            position_uncertainty=np.sqrt(self.P[0, 0] + self.P[1, 1] + self.P[2, 2]),
            velocity_uncertainty=np.sqrt(self.P[3, 3] + self.P[4, 4] + self.P[5, 5]),
            attitude_uncertainty=np.sqrt(self.P[6, 6] + self.P[7, 7] + self.P[8, 8])
        )


class GPSSensor:
    """GPS sensor simulation"""
    
    def __init__(self, update_rate: float = 1.0):
        self.update_rate = update_rate
        self.last_update = 0.0
        
        # GPS characteristics
        self.position_accuracy = 3.0  # meters (typical GPS accuracy)
        self.velocity_accuracy = 0.1  # m/s
        self.fix_quality = 1.0  # 0-1, quality of GPS fix
        
    def get_reading(self, true_position: Tuple[float, float, float],
                   true_velocity: Tuple[float, float, float]) -> SensorReading:
        """Get GPS reading with simulated noise"""
        current_time = time.time()
        
        if current_time - self.last_update < 1.0 / self.update_rate:
            return None
            
        self.last_update = current_time
        
        # Add noise to true values
        lat_noise = np.random.normal(0, self.position_accuracy / 111000)  # Convert to degrees
        lon_noise = np.random.normal(0, self.position_accuracy / (111000 * np.cos(np.radians(true_position[0]))))
        alt_noise = np.random.normal(0, self.position_accuracy * 1.5)
        
        vn_noise = np.random.normal(0, self.velocity_accuracy)
        ve_noise = np.random.normal(0, self.velocity_accuracy)
        vd_noise = np.random.normal(0, self.velocity_accuracy)
        
        data = {
            'latitude': true_position[0] + lat_noise,
            'longitude': true_position[1] + lon_noise,
            'altitude': true_position[2] + alt_noise,
            'velocity_north': true_velocity[0] + vn_noise,
            'velocity_east': true_velocity[1] + ve_noise,
            'velocity_down': true_velocity[2] + vd_noise,
            'fix_quality': self.fix_quality,
            'satellites': np.random.randint(8, 15)
        }
        
        uncertainty = {
            'position': self.position_accuracy,
            'velocity': self.velocity_accuracy
        }
        
        return SensorReading(
            sensor_type=SensorType.GPS,
            timestamp=current_time,
            data=data,
            uncertainty=uncertainty,
            valid=self.fix_quality > 0.5
        )


class IMUSensor:
    """Inertial Measurement Unit simulation"""
    
    def __init__(self, update_rate: float = 100.0):
        self.update_rate = update_rate
        self.last_update = 0.0
        
        # IMU characteristics
        self.accelerometer_bias = np.random.normal(0, 0.01, 3)  # m/s²
        self.gyroscope_bias = np.random.normal(0, 0.001, 3)    # rad/s
        self.accelerometer_noise = 0.05  # m/s²
        self.gyroscope_noise = 0.001     # rad/s
        
    def get_reading(self, true_acceleration: Tuple[float, float, float],
                   true_angular_rate: Tuple[float, float, float]) -> SensorReading:
        """Get IMU reading with simulated noise and bias"""
        current_time = time.time()
        
        if current_time - self.last_update < 1.0 / self.update_rate:
            return None
            
        self.last_update = current_time
        
        # Add bias and noise
        accel_noise = np.random.normal(0, self.accelerometer_noise, 3)
        gyro_noise = np.random.normal(0, self.gyroscope_noise, 3)
        
        data = {
            'acceleration_x': true_acceleration[0] + self.accelerometer_bias[0] + accel_noise[0],
            'acceleration_y': true_acceleration[1] + self.accelerometer_bias[1] + accel_noise[1],
            'acceleration_z': true_acceleration[2] + self.accelerometer_bias[2] + accel_noise[2],
            'angular_rate_x': true_angular_rate[0] + self.gyroscope_bias[0] + gyro_noise[0],
            'angular_rate_y': true_angular_rate[1] + self.gyroscope_bias[1] + gyro_noise[1],
            'angular_rate_z': true_angular_rate[2] + self.gyroscope_bias[2] + gyro_noise[2]
        }
        
        uncertainty = {
            'acceleration': self.accelerometer_noise,
            'angular_rate': self.gyroscope_noise
        }
        
        return SensorReading(
            sensor_type=SensorType.IMU,
            timestamp=current_time,
            data=data,
            uncertainty=uncertainty,
            valid=True
        )


class AirDataSensor:
    """Air data sensor simulation"""
    
    def __init__(self, update_rate: float = 10.0):
        self.update_rate = update_rate
        self.last_update = 0.0
        
        # Air data characteristics
        self.pitot_noise = 0.5  # m/s
        self.static_pressure_noise = 100  # Pa
        self.temperature_noise = 0.5  # K
        
    def get_reading(self, true_airspeed: float, true_pressure: float,
                   true_temperature: float) -> SensorReading:
        """Get air data reading with simulated noise"""
        current_time = time.time()
        
        if current_time - self.last_update < 1.0 / self.update_rate:
            return None
            
        self.last_update = current_time
        
        # Add noise
        airspeed_noise = np.random.normal(0, self.pitot_noise)
        pressure_noise = np.random.normal(0, self.static_pressure_noise)
        temperature_noise = np.random.normal(0, self.temperature_noise)
        
        data = {
            'airspeed': true_airspeed + airspeed_noise,
            'static_pressure': true_pressure + pressure_noise,
            'temperature': true_temperature + temperature_noise,
            'dynamic_pressure': 0.5 * 1.225 * (true_airspeed + airspeed_noise)**2
        }
        
        uncertainty = {
            'airspeed': self.pitot_noise,
            'pressure': self.static_pressure_noise,
            'temperature': self.temperature_noise
        }
        
        return SensorReading(
            sensor_type=SensorType.AIR_DATA,
            timestamp=current_time,
            data=data,
            uncertainty=uncertainty,
            valid=True
        )


class SensorFusion:
    """
    Multi-sensor data fusion system for HALE drone
    
    Combines GPS, IMU, air data, and other sensors to provide
    accurate state estimation and environmental data.
    """
    
    def __init__(self):
        self.kalman_filter = None
        self.sensors = {}
        self.sensor_readings = {}
        self.fused_state = None
        
        # Initialize sensors
        self.sensors[SensorType.GPS] = GPSSensor()
        self.sensors[SensorType.IMU] = IMUSensor()
        self.sensors[SensorType.AIR_DATA] = AirDataSensor()
        
        # Sensor fusion parameters
        self.gps_weight = 0.7
        self.imu_weight = 0.2
        self.air_data_weight = 0.1
        
        # Performance monitoring
        self.fusion_accuracy = 1.0
        self.last_fusion_time = 0.0
        
        logging.info("Sensor Fusion system initialized")
        
    def initialize(self, initial_state: FusedState):
        """Initialize sensor fusion with initial state"""
        self.kalman_filter = KalmanFilter(initial_state)
        self.fused_state = initial_state
        logging.info("Sensor Fusion initialized with initial state")
        
    def update_sensors(self, true_state: Dict[str, Any]):
        """Update all sensors with true state data"""
        current_time = time.time()
        
        # Update GPS
        gps_reading = self.sensors[SensorType.GPS].get_reading(
            (true_state['position']['latitude'], 
             true_state['position']['longitude'], 
             true_state['position']['altitude']),
            (true_state['velocity']['airspeed'], 0, 0)  # Simplified
        )
        if gps_reading:
            self.sensor_readings[SensorType.GPS] = gps_reading
            
        # Update IMU
        imu_reading = self.sensors[SensorType.IMU].get_reading(
            (0, 0, -9.81),  # Gravity
            (true_state.get('roll_rate', 0), 
             true_state.get('pitch_rate', 0), 
             true_state.get('yaw_rate', 0))
        )
        if imu_reading:
            self.sensor_readings[SensorType.IMU] = imu_reading
            
        # Update air data
        air_data_reading = self.sensors[SensorType.AIR_DATA].get_reading(
            true_state['velocity']['airspeed'],
            101325,  # Standard atmospheric pressure
            288.15   # Standard temperature
        )
        if air_data_reading:
            self.sensor_readings[SensorType.AIR_DATA] = air_data_reading
            
    def fuse_sensors(self) -> FusedState:
        """Fuse sensor data to get best state estimate"""
        if not self.kalman_filter:
            return None
            
        current_time = time.time()
        
        # Predict step
        self.kalman_filter.predict()
        
        # Update with available sensors
        if SensorType.GPS in self.sensor_readings:
            gps_reading = self.sensor_readings[SensorType.GPS]
            if gps_reading.valid:
                # GPS measurement matrix
                H_gps = np.zeros((6, 12))
                H_gps[0, 0] = 1  # latitude
                H_gps[1, 1] = 1  # longitude
                H_gps[2, 2] = 1  # altitude
                H_gps[3, 3] = 1  # velocity_north
                H_gps[4, 4] = 1  # velocity_east
                H_gps[5, 5] = 1  # velocity_down
                
                # GPS measurement
                z_gps = np.array([
                    gps_reading.data['latitude'],
                    gps_reading.data['longitude'],
                    gps_reading.data['altitude'],
                    gps_reading.data['velocity_north'],
                    gps_reading.data['velocity_east'],
                    gps_reading.data['velocity_down']
                ])
                
                # GPS measurement noise
                R_gps = np.eye(6) * gps_reading.uncertainty['position']**2
                R_gps[3:, 3:] = np.eye(3) * gps_reading.uncertainty['velocity']**2
                
                self.kalman_filter.update(z_gps, H_gps, R_gps)
                
        if SensorType.IMU in self.sensor_readings:
            imu_reading = self.sensor_readings[SensorType.IMU]
            if imu_reading.valid:
                # IMU measurement matrix (simplified)
                H_imu = np.zeros((6, 12))
                H_imu[0, 9] = 1   # roll_rate
                H_imu[1, 10] = 1  # pitch_rate
                H_imu[2, 11] = 1  # yaw_rate
                # Accelerometer measurements would require more complex integration
                
                # IMU measurement
                z_imu = np.array([
                    imu_reading.data['angular_rate_x'],
                    imu_reading.data['angular_rate_y'],
                    imu_reading.data['angular_rate_z'],
                    0, 0, 0  # Placeholder for accelerometer
                ])
                
                # IMU measurement noise
                R_imu = np.eye(6) * imu_reading.uncertainty['angular_rate']**2
                
                self.kalman_filter.update(z_imu, H_imu, R_imu)
                
        # Get fused state
        self.fused_state = self.kalman_filter.get_state()
        self.last_fusion_time = current_time
        
        # Update fusion accuracy
        self._update_fusion_accuracy()
        
        return self.fused_state
        
    def _update_fusion_accuracy(self):
        """Update fusion accuracy metric"""
        if not self.fused_state:
            return
            
        # Calculate accuracy based on uncertainty
        total_uncertainty = (self.fused_state.position_uncertainty + 
                           self.fused_state.velocity_uncertainty + 
                           self.fused_state.attitude_uncertainty)
        
        # Normalize to 0-1 scale
        self.fusion_accuracy = max(0, 1 - total_uncertainty / 10.0)
        
    def get_sensor_status(self) -> Dict[str, Any]:
        """Get status of all sensors"""
        status = {}
        
        for sensor_type, sensor in self.sensors.items():
            last_reading = self.sensor_readings.get(sensor_type)
            status[sensor_type.value] = {
                'active': last_reading is not None,
                'last_update': last_reading.timestamp if last_reading else 0,
                'valid': last_reading.valid if last_reading else False,
                'uncertainty': last_reading.uncertainty if last_reading else {}
            }
            
        return status
        
    def get_fusion_status(self) -> Dict[str, Any]:
        """Get sensor fusion status"""
        return {
            'fused_state_available': self.fused_state is not None,
            'fusion_accuracy': self.fusion_accuracy,
            'last_fusion_time': self.last_fusion_time,
            'active_sensors': len([r for r in self.sensor_readings.values() if r.valid]),
            'total_sensors': len(self.sensors)
        } 