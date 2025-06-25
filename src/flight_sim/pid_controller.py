"""
PID Controller for HALE Drone Flight Control
===========================================

This module implements PID controllers for various flight control functions
including attitude control, altitude hold, speed control, and navigation.

Author: Quantum HALE Development Team
License: MIT
"""

import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum


class ControlMode(Enum):
    """PID control modes"""
    POSITION = "position"
    VELOCITY = "velocity"
    ATTITUDE = "attitude"
    RATE = "rate"
    ALTITUDE = "altitude"
    SPEED = "speed"
    HEADING = "heading"


@dataclass
class PIDGains:
    """PID controller gains"""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    kf: float = 0.0  # Feedforward gain
    output_limit: float = 1.0  # Output saturation limit
    integral_limit: float = 0.5  # Integral anti-windup limit
    deadband: float = 0.0  # Deadband around setpoint


@dataclass
class PIDState:
    """Internal PID controller state"""
    setpoint: float = 0.0
    measurement: float = 0.0
    error: float = 0.0
    error_prev: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    output: float = 0.0
    output_filtered: float = 0.0
    timestamp: float = 0.0
    dt: float = 0.01


class PIDController:
    """
    Advanced PID Controller with anti-windup, filtering, and multiple modes
    
    Features:
    - Anti-windup protection
    - Output filtering
    - Deadband handling
    - Feedforward control
    - Multiple control modes
    - Bumpless transfer
    """
    
    def __init__(self, gains: PIDGains, mode: ControlMode = ControlMode.POSITION):
        self.gains = gains
        self.mode = mode
        self.state = PIDState()
        
        # Filtering parameters
        self.derivative_filter_alpha = 0.1  # Low-pass filter for derivative
        self.output_filter_alpha = 0.3  # Low-pass filter for output
        
        # Anti-windup
        self.integral_windup = 0.0
        self.back_calculation = True
        
        # Performance tracking
        self.performance_metrics = {
            'total_error': 0.0,
            'max_error': 0.0,
            'settling_time': 0.0,
            'overshoot': 0.0,
            'rise_time': 0.0
        }
        
        # Control history
        self.history = {
            'time': [],
            'setpoint': [],
            'measurement': [],
            'error': [],
            'output': []
        }
        
        logging.info(f"PID Controller initialized: {mode.value} mode, Kp={gains.kp}, Ki={gains.ki}, Kd={gains.kd}")
        
    def set_setpoint(self, setpoint: float):
        """Set the target value for the controller"""
        self.state.setpoint = setpoint
        
    def update(self, measurement: float, dt: Optional[float] = None) -> float:
        """
        Update the PID controller with new measurement
        
        Args:
            measurement: Current measured value
            dt: Time step (uses previous if None)
            
        Returns:
            Control output
        """
        current_time = time.time()
        
        if dt is None:
            dt = current_time - self.state.timestamp if self.state.timestamp > 0 else 0.01
        else:
            dt = max(dt, 0.001)  # Minimum time step
            
        # Update state
        self.state.measurement = measurement
        self.state.error_prev = self.state.error
        self.state.error = self.state.setpoint - measurement
        self.state.dt = dt
        self.state.timestamp = current_time
        
        # Apply deadband
        if abs(self.state.error) < self.gains.deadband:
            self.state.error = 0.0
            
        # Calculate PID terms
        proportional = self.gains.kp * self.state.error
        
        # Integral term with anti-windup
        if self.gains.ki > 0:
            self.state.integral += self.state.error * dt
            
            # Anti-windup: limit integral term
            integral_limit = self.gains.integral_limit / self.gains.ki
            self.state.integral = np.clip(self.state.integral, -integral_limit, integral_limit)
            
            integral = self.gains.ki * self.state.integral
        else:
            integral = 0.0
            
        # Derivative term with filtering
        if dt > 0:
            derivative_raw = (self.state.error - self.state.error_prev) / dt
            self.state.derivative = (self.derivative_filter_alpha * derivative_raw + 
                                   (1 - self.derivative_filter_alpha) * self.state.derivative)
            derivative = self.gains.kd * self.state.derivative
        else:
            derivative = 0.0
            
        # Feedforward term
        feedforward = self.gains.kf * self.state.setpoint
        
        # Calculate output
        output = proportional + integral + derivative + feedforward
        
        # Apply output limits
        output = np.clip(output, -self.gains.output_limit, self.gains.output_limit)
        
        # Output filtering
        self.state.output_filtered = (self.output_filter_alpha * output + 
                                    (1 - self.output_filter_alpha) * self.state.output_filtered)
        
        self.state.output = self.state.output_filtered
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Store history
        self._store_history()
        
        return self.state.output
        
    def reset(self):
        """Reset the controller state"""
        self.state = PIDState()
        self.integral_windup = 0.0
        
    def set_gains(self, gains: PIDGains):
        """Update controller gains"""
        self.gains = gains
        logging.info(f"PID gains updated: Kp={gains.kp}, Ki={gains.ki}, Kd={gains.kd}")
        
    def get_state(self) -> Dict[str, Any]:
        """Get current controller state"""
        return {
            'setpoint': self.state.setpoint,
            'measurement': self.state.measurement,
            'error': self.state.error,
            'integral': self.state.integral,
            'derivative': self.state.derivative,
            'output': self.state.output,
            'gains': {
                'kp': self.gains.kp,
                'ki': self.gains.ki,
                'kd': self.gains.kd,
                'kf': self.gains.kf
            },
            'performance': self.performance_metrics
        }
        
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        self.performance_metrics['total_error'] += abs(self.state.error) * self.state.dt
        self.performance_metrics['max_error'] = max(self.performance_metrics['max_error'], abs(self.state.error))
        
    def _store_history(self):
        """Store control history for analysis"""
        if len(self.history['time']) > 1000:  # Limit history size
            for key in self.history:
                self.history[key] = self.history[key][-500:]
                
        self.history['time'].append(self.state.timestamp)
        self.history['setpoint'].append(self.state.setpoint)
        self.history['measurement'].append(self.state.measurement)
        self.history['error'].append(self.state.error)
        self.history['output'].append(self.state.output)


class FlightController:
    """
    Multi-loop flight controller for HALE drone
    
    Implements cascaded control loops for:
    - Altitude hold
    - Speed control
    - Heading control
    - Attitude stabilization
    """
    
    def __init__(self):
        # Altitude controller (outer loop)
        self.altitude_controller = PIDController(
            PIDGains(kp=0.1, ki=0.01, kd=0.05, output_limit=0.5),
            ControlMode.ALTITUDE
        )
        
        # Speed controller
        self.speed_controller = PIDController(
            PIDGains(kp=0.2, ki=0.02, kd=0.1, output_limit=1.0),
            ControlMode.SPEED
        )
        
        # Heading controller (outer loop)
        self.heading_controller = PIDController(
            PIDGains(kp=0.5, ki=0.01, kd=0.1, output_limit=0.3),
            ControlMode.HEADING
        )
        
        # Pitch rate controller (inner loop)
        self.pitch_rate_controller = PIDController(
            PIDGains(kp=2.0, ki=0.1, kd=0.5, output_limit=0.5),
            ControlMode.RATE
        )
        
        # Roll rate controller (inner loop)
        self.roll_rate_controller = PIDController(
            PIDGains(kp=2.0, ki=0.1, kd=0.5, output_limit=0.5),
            ControlMode.RATE
        )
        
        # Yaw rate controller (inner loop)
        self.yaw_rate_controller = PIDController(
            PIDGains(kp=1.5, ki=0.05, kd=0.3, output_limit=0.3),
            ControlMode.RATE
        )
        
        # Control mode
        self.control_mode = "manual"  # manual, altitude_hold, heading_hold, auto
        
        # Control outputs
        self.controls = {
            'throttle': 0.0,
            'elevator': 0.0,
            'aileron': 0.0,
            'rudder': 0.0
        }
        
        logging.info("Flight Controller initialized")
        
    def update(self, state: Dict[str, Any], dt: float) -> Dict[str, float]:
        """
        Update flight controller with current state
        
        Args:
            state: Current aircraft state
            dt: Time step
            
        Returns:
            Control outputs
        """
        if self.control_mode == "manual":
            return self.controls
            
        # Extract state values from the dictionary structure
        position = state.get('position', {})
        velocity = state.get('velocity', {})
        attitude = state.get('attitude', {})
        
        current_altitude = position.get('altitude', 0.0)
        current_airspeed = velocity.get('airspeed', 0.0)
        current_heading = velocity.get('heading', 0.0)
        current_pitch_rate = attitude.get('pitch_rate', 0.0)
        current_roll_rate = attitude.get('roll_rate', 0.0)
        current_yaw_rate = attitude.get('yaw_rate', 0.0)
        
        # Altitude hold
        if self.control_mode in ["altitude_hold", "auto"]:
            altitude_setpoint = state.get('altitude_setpoint', current_altitude)
            pitch_setpoint = self.altitude_controller.update(current_altitude, dt)
            
            # Pitch rate control
            pitch_rate_error = pitch_setpoint - current_pitch_rate
            self.controls['elevator'] = self.pitch_rate_controller.update(current_pitch_rate, dt)
            
        # Speed control
        if self.control_mode in ["speed_hold", "auto"]:
            speed_setpoint = state.get('speed_setpoint', current_airspeed)
            self.controls['throttle'] = self.speed_controller.update(current_airspeed, dt)
            
        # Heading control
        if self.control_mode in ["heading_hold", "auto"]:
            heading_setpoint = state.get('heading_setpoint', current_heading)
            roll_setpoint = self.heading_controller.update(current_heading, dt)
            
            # Roll rate control
            roll_rate_error = roll_setpoint - current_roll_rate
            self.controls['aileron'] = self.roll_rate_controller.update(current_roll_rate, dt)
            
        # Apply control limits
        self.controls['throttle'] = np.clip(self.controls['throttle'], 0.0, 1.0)
        self.controls['elevator'] = np.clip(self.controls['elevator'], -0.5, 0.5)
        self.controls['aileron'] = np.clip(self.controls['aileron'], -0.5, 0.5)
        self.controls['rudder'] = np.clip(self.controls['rudder'], -0.3, 0.3)
        
        return self.controls
        
    def set_control_mode(self, mode: str):
        """Set control mode"""
        self.control_mode = mode
        logging.info(f"Flight control mode set to: {mode}")
        
    def set_manual_controls(self, throttle: float, elevator: float, 
                          aileron: float, rudder: float):
        """Set manual control inputs"""
        self.controls['throttle'] = np.clip(throttle, 0.0, 1.0)
        self.controls['elevator'] = np.clip(elevator, -0.5, 0.5)
        self.controls['aileron'] = np.clip(aileron, -0.5, 0.5)
        self.controls['rudder'] = np.clip(rudder, -0.3, 0.3)
        
    def get_controller_status(self) -> Dict[str, Any]:
        """Get status of all controllers"""
        return {
            'control_mode': self.control_mode,
            'controls': self.controls,
            'altitude_controller': self.altitude_controller.get_state(),
            'speed_controller': self.speed_controller.get_state(),
            'heading_controller': self.heading_controller.get_state()
        } 