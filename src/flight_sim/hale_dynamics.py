"""
HALE Drone Flight Dynamics Simulation
====================================

This module implements the flight dynamics and control systems for
High-Altitude Long-Endurance (HALE) drones, including aerodynamic models,
propulsion systems, and flight control algorithms.

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

# Import PID controller
from .pid_controller import PIDController, PIDGains, ControlMode, FlightController

# Flight dynamics constants
GRAVITY = 9.81  # m/s²
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³
TROPOSPHERE_HEIGHT = 11000  # meters
STRATOSPHERE_HEIGHT = 50000  # meters
EARTH_RADIUS = 6371000  # meters


class FlightPhase(Enum):
    """Flight phases for HALE drone"""
    GROUND = "ground"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    LANDING = "landing"
    HOLDING = "holding"


@dataclass
class AircraftState:
    """Complete aircraft state vector"""
    # Position (ECEF coordinates)
    latitude: float  # radians
    longitude: float  # radians
    altitude: float  # meters MSL
    
    # Velocity (body frame)
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
    
    # Flight parameters
    airspeed: float    # m/s
    ground_speed: float  # m/s
    heading: float     # radians
    flight_path_angle: float  # radians
    
    # Energy state
    total_energy: float  # J
    potential_energy: float  # J
    kinetic_energy: float    # J
    
    # Fuel/battery state
    fuel_remaining: float  # kg or Wh
    fuel_consumption_rate: float  # kg/s or W
    
    # Environmental
    air_density: float  # kg/m³
    temperature: float  # K
    pressure: float     # Pa
    wind_north: float   # m/s
    wind_east: float    # m/s
    wind_up: float      # m/s


@dataclass
class AircraftParameters:
    """Aircraft physical and aerodynamic parameters"""
    # Physical dimensions
    wingspan: float  # meters
    wing_area: float  # m²
    length: float  # meters
    mass_empty: float  # kg
    mass_max_takeoff: float  # kg
    
    # Aerodynamic coefficients
    cl_alpha: float  # lift curve slope
    cd0: float  # zero-lift drag coefficient
    oswald_efficiency: float  # Oswald efficiency factor
    aspect_ratio: float  # wing aspect ratio
    
    # Propulsion
    thrust_max: float  # N
    specific_fuel_consumption: float  # kg/N/s
    propeller_efficiency: float
    
    # Control surfaces
    elevator_effectiveness: float
    aileron_effectiveness: float
    rudder_effectiveness: float
    
    # Performance
    stall_speed: float  # m/s
    max_speed: float    # m/s
    service_ceiling: float  # meters
    range_max: float    # meters
    endurance_max: float  # seconds


class AtmosphericModel:
    """Enhanced atmospheric model for high-altitude operations"""
    
    def __init__(self):
        self.temperature_lapse_rate = -0.0065  # K/m (troposphere)
        self.temperature_sea_level = 288.15  # K
        self.pressure_sea_level = 101325  # Pa
        
    def get_atmospheric_conditions(self, altitude: float) -> Dict[str, float]:
        """
        Calculate atmospheric conditions at given altitude
        
        Args:
            altitude: Altitude in meters MSL
            
        Returns:
            Dictionary with temperature, pressure, density, speed_of_sound
        """
        if altitude <= TROPOSPHERE_HEIGHT:
            # Troposphere (0-11km)
            temperature = self.temperature_sea_level + self.temperature_lapse_rate * altitude
            pressure = self.pressure_sea_level * (temperature / self.temperature_sea_level) ** 5.256
        elif altitude <= STRATOSPHERE_HEIGHT:
            # Stratosphere (11-50km) - more accurate model
            temperature = 216.65  # Constant temperature in lower stratosphere
            pressure = 22632 * np.exp(-(altitude - TROPOSPHERE_HEIGHT) / 6341.6)
        else:
            # Upper atmosphere (simplified)
            temperature = 270.65 - 0.001 * (altitude - STRATOSPHERE_HEIGHT)
            pressure = 5474.9 * np.exp(-(altitude - STRATOSPHERE_HEIGHT) / 7922.0)
            
        # Calculate density using ideal gas law
        density = pressure / (287.05 * temperature)
        
        # Calculate speed of sound
        speed_of_sound = np.sqrt(1.4 * 287.05 * temperature)
        
        return {
            'temperature': temperature,
            'pressure': pressure,
            'density': density,
            'speed_of_sound': speed_of_sound
        }
        
    def get_wind_model(self, altitude: float, latitude: float, longitude: float) -> Tuple[float, float, float]:
        """
        Calculate wind conditions at given position
        
        Args:
            altitude: Altitude in meters
            latitude: Latitude in radians
            longitude: Longitude in radians
            
        Returns:
            Wind velocity components (north, east, up) in m/s
        """
        # Simplified wind model - in reality this would use weather data
        if altitude < 1000:
            # Low altitude: light winds
            wind_speed = 2.0 + 3.0 * np.sin(time.time() * 0.001)
            wind_direction = np.sin(time.time() * 0.0005)
        elif altitude < 10000:
            # Mid altitude: moderate winds
            wind_speed = 5.0 + 8.0 * np.sin(time.time() * 0.0008)
            wind_direction = np.sin(time.time() * 0.0003)
        else:
            # High altitude: jet stream effects
            wind_speed = 15.0 + 20.0 * np.sin(time.time() * 0.0006)
            wind_direction = np.sin(time.time() * 0.0002)
            
        wind_north = wind_speed * np.cos(wind_direction)
        wind_east = wind_speed * np.sin(wind_direction)
        wind_up = 0.0  # Vertical wind component (usually small)
        
        return wind_north, wind_east, wind_up


class HALEDynamics:
    """
    Enhanced HALE Drone Flight Dynamics Simulation
    
    Implements 6-DOF flight dynamics with realistic atmospheric modeling,
    propulsion systems, and advanced flight control.
    """
    
    def __init__(self, aircraft_params: AircraftParameters):
        self.params = aircraft_params
        self.atmosphere = AtmosphericModel()
        self.state = None
        self.flight_phase = FlightPhase.GROUND
        self.time = 0.0
        self.dt = 0.01  # Integration time step
        
        # Flight controller
        self.flight_controller = FlightController()
        
        # Control inputs
        self.throttle = 0.0  # 0-1
        self.elevator = 0.0  # radians
        self.aileron = 0.0   # radians
        self.rudder = 0.0    # radians
        
        # Performance tracking
        self.flight_time = 0.0
        self.distance_traveled = 0.0
        self.fuel_consumed = 0.0
        
        # Flight envelope protection
        self.envelope_limits = {
            'max_altitude': aircraft_params.service_ceiling,
            'min_altitude': 100.0,  # Minimum safe altitude
            'max_airspeed': aircraft_params.max_speed,
            'min_airspeed': aircraft_params.stall_speed * 1.2,  # 20% above stall
            'max_load_factor': 2.5,
            'max_angle_of_attack': np.radians(15.0)
        }
        
        # State history for analysis
        self.state_history = []
        
        logging.info(f"HALE Dynamics initialized for aircraft: {aircraft_params.mass_max_takeoff}kg MTOW")
        
    def initialize_state(self, initial_state: AircraftState):
        """Initialize aircraft state"""
        self.state = initial_state
        self.time = 0.0
        self.flight_time = 0.0
        self.distance_traveled = 0.0
        self.fuel_consumed = 0.0
        
        # Update atmospheric conditions
        self._update_atmospheric_conditions()
        
        logging.info(f"Aircraft initialized at {initial_state.altitude}m altitude")
        
    def set_controls(self, throttle: float, elevator: float, 
                    aileron: float, rudder: float):
        """Set control inputs with envelope protection"""
        self.throttle = np.clip(throttle, 0.0, 1.0)
        self.elevator = np.clip(elevator, -0.5, 0.5)
        self.aileron = np.clip(aileron, -0.5, 0.5)
        self.rudder = np.clip(rudder, -0.5, 0.5)
        
    def calculate_aerodynamic_forces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate aerodynamic forces and moments
        
        Returns:
            Tuple of (forces, moments) in body frame
        """
        if not self.state:
            return np.zeros(3), np.zeros(3)
            
        # Get atmospheric conditions
        atm_conditions = self.atmosphere.get_atmospheric_conditions(self.state.altitude)
        rho = atm_conditions['density']
        
        # Calculate dynamic pressure
        q = 0.5 * rho * self.state.airspeed**2
        
        # Calculate angle of attack and sideslip
        # Use a small minimum angle of attack to ensure lift generation
        alpha = np.arctan2(self.state.velocity_down, max(self.state.velocity_north, 1.0))
        beta = np.arcsin(self.state.velocity_east / max(self.state.airspeed, 1.0))
        
        # Ensure minimum angle of attack for lift generation
        if abs(alpha) < 0.01:  # Less than ~0.6 degrees
            alpha = 0.01 if alpha >= 0 else -0.01
            
        # Lift coefficient (simplified)
        cl = self.params.cl_alpha * alpha
        
        # Drag coefficient (parabolic drag polar)
        cd = self.params.cd0 + cl**2 / (np.pi * self.params.aspect_ratio * self.params.oswald_efficiency)
        
        # Side force coefficient
        cy = -0.1 * beta  # Simplified side force model
        
        # Calculate forces in body frame
        lift = q * self.params.wing_area * cl
        drag = q * self.params.wing_area * cd
        side_force = q * self.params.wing_area * cy
        
        # Convert to body frame
        forces = np.array([
            -drag,  # X-axis (forward)
            side_force,  # Y-axis (right)
            -lift  # Z-axis (down)
        ])
        
        # Calculate moments (simplified)
        # Roll moment due to aileron
        roll_moment = q * self.params.wing_area * self.params.wingspan * 0.1 * self.aileron
        
        # Pitch moment due to elevator and angle of attack
        pitch_moment = q * self.params.wing_area * self.params.wingspan * (
            0.05 * alpha + 0.1 * self.elevator
        )
        
        # Yaw moment due to rudder and sideslip
        yaw_moment = q * self.params.wing_area * self.params.wingspan * (
            -0.05 * beta + 0.1 * self.rudder
        )
        
        moments = np.array([roll_moment, pitch_moment, yaw_moment])
        
        return forces, moments
        
    def calculate_propulsion_forces(self) -> np.ndarray:
        """Calculate propulsion forces"""
        if not self.state:
            return np.zeros(3)
            
        # Get atmospheric conditions for thrust variation with altitude
        atm_conditions = self.atmosphere.get_atmospheric_conditions(self.state.altitude)
        density_ratio = atm_conditions['density'] / AIR_DENSITY_SEA_LEVEL
        
        # Thrust varies with altitude and airspeed
        thrust = self.params.thrust_max * self.throttle * density_ratio
        
        # Thrust vector in body frame (assumed aligned with X-axis)
        thrust_vector = np.array([thrust, 0.0, 0.0])
        
        # Calculate fuel consumption
        if self.throttle > 0:
            self.state.fuel_consumption_rate = (
                self.params.specific_fuel_consumption * thrust * self.throttle
            )
        else:
            self.state.fuel_consumption_rate = 0.0
            
        return thrust_vector
        
    def calculate_gravity_forces(self) -> np.ndarray:
        """Calculate gravitational forces"""
        if not self.state:
            return np.zeros(3)
            
        # Current mass (varies with fuel consumption)
        current_mass = self.params.mass_empty + self.state.fuel_remaining
        
        # Gravity vector in body frame
        gravity_body = np.array([
            -GRAVITY * current_mass * np.sin(self.state.pitch),
            GRAVITY * current_mass * np.sin(self.state.roll) * np.cos(self.state.pitch),
            GRAVITY * current_mass * np.cos(self.state.roll) * np.cos(self.state.pitch)
        ])
        
        return gravity_body
        
    def update_state(self):
        """Update aircraft state using 6-DOF equations of motion"""
        if not self.state:
            return
            
        # Calculate forces and moments
        aero_forces, aero_moments = self.calculate_aerodynamic_forces()
        prop_forces = self.calculate_propulsion_forces()
        gravity_forces = self.calculate_gravity_forces()
        
        total_forces = aero_forces + prop_forces + gravity_forces
        total_moments = aero_moments
        
        # Current mass
        current_mass = self.params.mass_empty + self.state.fuel_remaining
        
        # Inertia matrix (simplified)
        Ixx = current_mass * (self.params.wingspan / 2)**2
        Iyy = current_mass * (self.params.length / 2)**2
        Izz = Ixx + Iyy
        I = np.diag([Ixx, Iyy, Izz])
        
        # Update velocities (F = ma)
        acceleration = total_forces / current_mass
        self.state.velocity_north += acceleration[0] * self.dt
        self.state.velocity_east += acceleration[1] * self.dt
        self.state.velocity_down += acceleration[2] * self.dt
        
        # Update angular rates (M = I * alpha)
        angular_acceleration = np.linalg.solve(I, total_moments)
        self.state.roll_rate += angular_acceleration[0] * self.dt
        self.state.pitch_rate += angular_acceleration[1] * self.dt
        self.state.yaw_rate += angular_acceleration[2] * self.dt
        
        # Update attitude (Euler integration)
        self.state.roll += self.state.roll_rate * self.dt
        self.state.pitch += self.state.pitch_rate * self.dt
        self.state.yaw += self.state.yaw_rate * self.dt
        
        # Update position
        # Convert body velocities to NED frame
        cos_roll = np.cos(self.state.roll)
        sin_roll = np.sin(self.state.roll)
        cos_pitch = np.cos(self.state.pitch)
        sin_pitch = np.sin(self.state.pitch)
        cos_yaw = np.cos(self.state.yaw)
        sin_yaw = np.sin(self.state.yaw)
        
        # Rotation matrix from body to NED
        R = np.array([
            [cos_pitch * cos_yaw, sin_roll * sin_pitch * cos_yaw - cos_roll * sin_yaw, cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw],
            [cos_pitch * sin_yaw, sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw, cos_roll * sin_pitch * sin_yaw - sin_roll * cos_yaw],
            [-sin_pitch, sin_roll * cos_pitch, cos_roll * cos_pitch]
        ])
        
        # Transform velocities
        ned_velocities = R @ np.array([self.state.velocity_north, self.state.velocity_east, self.state.velocity_down])
        
        # Update position (simplified - assumes flat Earth)
        self.state.latitude += ned_velocities[0] / EARTH_RADIUS * self.dt
        self.state.longitude += ned_velocities[1] / (EARTH_RADIUS * np.cos(self.state.latitude)) * self.dt
        self.state.altitude -= ned_velocities[2] * self.dt
        
        # Update flight parameters
        self.state.airspeed = np.sqrt(self.state.velocity_north**2 + self.state.velocity_east**2 + self.state.velocity_down**2)
        self.state.ground_speed = np.sqrt(ned_velocities[0]**2 + ned_velocities[1]**2)
        self.state.heading = np.arctan2(ned_velocities[1], ned_velocities[0])
        self.state.flight_path_angle = np.arctan2(-ned_velocities[2], self.state.ground_speed)
        
        # Update energy state
        self.state.potential_energy = current_mass * GRAVITY * self.state.altitude
        self.state.kinetic_energy = 0.5 * current_mass * self.state.airspeed**2
        self.state.total_energy = self.state.potential_energy + self.state.kinetic_energy
        
        # Update fuel
        self.state.fuel_remaining -= self.state.fuel_consumption_rate * self.dt
        self.state.fuel_remaining = max(0.0, self.state.fuel_remaining)
        
        # Update atmospheric conditions
        self._update_atmospheric_conditions()
        
        # Update flight phase
        self._update_flight_phase()
        
        # Flight envelope protection
        self._apply_envelope_protection()
        
        # Update performance tracking
        self.flight_time += self.dt
        self.distance_traveled += self.state.ground_speed * self.dt
        self.fuel_consumed += self.state.fuel_consumption_rate * self.dt
        
    def _update_atmospheric_conditions(self):
        """Update atmospheric conditions at current position"""
        if not self.state:
            return
            
        atm_conditions = self.atmosphere.get_atmospheric_conditions(self.state.altitude)
        self.state.air_density = atm_conditions['density']
        self.state.temperature = atm_conditions['temperature']
        self.state.pressure = atm_conditions['pressure']
        
        # Update wind conditions
        wind_north, wind_east, wind_up = self.atmosphere.get_wind_model(
            self.state.altitude, self.state.latitude, self.state.longitude
        )
        self.state.wind_north = wind_north
        self.state.wind_east = wind_east
        self.state.wind_up = wind_up
        
    def _update_flight_phase(self):
        """Update flight phase based on current state"""
        if not self.state:
            return
            
        if self.state.altitude < 10:
            self.flight_phase = FlightPhase.GROUND
        elif self.state.altitude < 100 and self.state.velocity_down > 0:
            self.flight_phase = FlightPhase.LANDING
        elif self.state.altitude < 100 and self.state.velocity_down < 0:
            self.flight_phase = FlightPhase.TAKEOFF
        elif self.state.velocity_down < -2:
            self.flight_phase = FlightPhase.CLIMB
        elif self.state.velocity_down > 2:
            self.flight_phase = FlightPhase.DESCENT
        else:
            self.flight_phase = FlightPhase.CRUISE
            
    def _apply_envelope_protection(self):
        """Apply flight envelope protection"""
        if not self.state:
            return
            
        # Altitude limits
        if self.state.altitude > self.envelope_limits['max_altitude']:
            self.state.altitude = self.envelope_limits['max_altitude']
            self.state.velocity_down = max(0.0, self.state.velocity_down)
            
        if self.state.altitude < self.envelope_limits['min_altitude']:
            self.state.altitude = self.envelope_limits['min_altitude']
            self.state.velocity_down = min(0.0, self.state.velocity_down)
            
        # Airspeed limits
        if self.state.airspeed > self.envelope_limits['max_airspeed']:
            # Reduce throttle
            self.throttle = max(0.0, self.throttle - 0.1)
            
        if self.state.airspeed < self.envelope_limits['min_airspeed']:
            # Increase throttle
            self.throttle = min(1.0, self.throttle + 0.1)
            
    def step(self):
        """Execute one simulation step"""
        # Update flight controller if in automatic mode
        if self.flight_controller.control_mode != "manual":
            state_dict = self.get_telemetry()
            controls = self.flight_controller.update(state_dict, self.dt)
            self.set_controls(controls['throttle'], controls['elevator'], 
                            controls['aileron'], controls['rudder'])
        
        # Update aircraft state
        self.update_state()
        
        # Store state history
        self._store_state_history()
        
        self.time += self.dt
        
    def _store_state_history(self):
        """Store state history for analysis"""
        if len(self.state_history) > 1000:  # Limit history size
            self.state_history = self.state_history[-500:]
            
        self.state_history.append({
            'time': self.time,
            'altitude': self.state.altitude if self.state else 0.0,
            'airspeed': self.state.airspeed if self.state else 0.0,
            'fuel_remaining': self.state.fuel_remaining if self.state else 0.0,
            'flight_phase': self.flight_phase.value
        })
        
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry data"""
        if not self.state:
            return {}
            
        return {
            'timestamp': self.time,
            'position': {
                'latitude': np.degrees(self.state.latitude),
                'longitude': np.degrees(self.state.longitude),
                'altitude': self.state.altitude
            },
            'velocity': {
                'airspeed': self.state.airspeed,
                'ground_speed': self.state.ground_speed,
                'heading': np.degrees(self.state.heading)
            },
            'attitude': {
                'roll': np.degrees(self.state.roll),
                'pitch': np.degrees(self.state.pitch),
                'yaw': np.degrees(self.state.yaw)
            },
            'energy': {
                'total_energy': self.state.total_energy,
                'fuel_remaining': self.state.fuel_remaining,
                'fuel_consumption_rate': self.state.fuel_consumption_rate
            },
            'flight_phase': self.flight_phase.value,
            'controls': {
                'throttle': self.throttle,
                'elevator': self.elevator,
                'aileron': self.aileron,
                'rudder': self.rudder
            },
            'performance': {
                'flight_time': self.flight_time,
                'distance_traveled': self.distance_traveled,
                'fuel_consumed': self.fuel_consumed
            },
            'environment': {
                'air_density': self.state.air_density,
                'temperature': self.state.temperature,
                'pressure': self.state.pressure,
                'wind_north': self.state.wind_north,
                'wind_east': self.state.wind_east,
                'wind_up': self.state.wind_up
            }
        }
        
    def check_flight_envelope(self) -> Dict[str, bool]:
        """Check if aircraft is within flight envelope"""
        if not self.state:
            return {'within_envelope': False, 'warnings': ['No state available']}
            
        envelope_ok = True
        warnings = []
        
        # Check altitude
        if self.state.altitude > self.envelope_limits['max_altitude']:
            envelope_ok = False
            warnings.append('Above service ceiling')
            
        if self.state.altitude < self.envelope_limits['min_altitude']:
            envelope_ok = False
            warnings.append('Below minimum safe altitude')
            
        # Check airspeed
        if self.state.airspeed > self.envelope_limits['max_airspeed']:
            envelope_ok = False
            warnings.append('Above maximum airspeed')
            
        if self.state.airspeed < self.envelope_limits['min_airspeed']:
            envelope_ok = False
            warnings.append('Below minimum airspeed (stall risk)')
            
        # Check fuel
        if self.state.fuel_remaining < 100:  # 100kg or Wh remaining
            warnings.append('Low fuel warning')
            
        return {
            'within_envelope': envelope_ok,
            'warnings': warnings
        }
        
    def set_flight_controller_mode(self, mode: str):
        """Set flight controller mode"""
        self.flight_controller.set_control_mode(mode)
        
    def get_flight_controller_status(self) -> Dict[str, Any]:
        """Get flight controller status"""
        return self.flight_controller.get_controller_status() 