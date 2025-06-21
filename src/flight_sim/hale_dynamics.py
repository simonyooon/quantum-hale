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

# Flight dynamics constants
GRAVITY = 9.81  # m/s²
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³
TROPOSPHERE_HEIGHT = 11000  # meters
STRATOSPHERE_HEIGHT = 50000  # meters


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
    """Atmospheric model for high-altitude operations"""
    
    def __init__(self):
        self.temperature_lapse_rate = -0.0065  # K/m (troposphere)
        self.temperature_sea_level = 288.15  # K
        
    def get_atmospheric_conditions(self, altitude: float) -> Dict[str, float]:
        """
        Calculate atmospheric conditions at given altitude
        
        Args:
            altitude: Altitude in meters MSL
            
        Returns:
            Dictionary with temperature, pressure, density
        """
        if altitude <= TROPOSPHERE_HEIGHT:
            # Troposphere (0-11km)
            temperature = self.temperature_sea_level + self.temperature_lapse_rate * altitude
            pressure = 101325 * (temperature / self.temperature_sea_level) ** 5.256
        else:
            # Stratosphere (11-50km) - simplified model
            temperature = 216.65  # Constant temperature in lower stratosphere
            pressure = 22632 * np.exp(-(altitude - TROPOSPHERE_HEIGHT) / 6341.6)
            
        # Calculate density using ideal gas law
        density = pressure / (287.05 * temperature)
        
        return {
            'temperature': temperature,
            'pressure': pressure,
            'density': density
        }


class HALEDynamics:
    """
    HALE Drone Flight Dynamics Simulation
    
    Implements 6-DOF flight dynamics with atmospheric modeling,
    propulsion systems, and basic flight control.
    """
    
    def __init__(self, aircraft_params: AircraftParameters):
        self.params = aircraft_params
        self.atmosphere = AtmosphericModel()
        self.state = None
        self.flight_phase = FlightPhase.GROUND
        self.time = 0.0
        self.dt = 0.01  # Integration time step
        
        # Control inputs
        self.throttle = 0.0  # 0-1
        self.elevator = 0.0  # radians
        self.aileron = 0.0   # radians
        self.rudder = 0.0    # radians
        
        # Performance tracking
        self.flight_time = 0.0
        self.distance_traveled = 0.0
        self.fuel_consumed = 0.0
        
        logging.info(f"HALE Dynamics initialized for aircraft: {aircraft_params.mass_max_takeoff}kg MTOW")
        
    def initialize_state(self, initial_state: AircraftState):
        """Initialize aircraft state"""
        self.state = initial_state
        self.time = 0.0
        logging.info(f"Aircraft initialized at {initial_state.altitude}m altitude")
        
    def set_controls(self, throttle: float, elevator: float, 
                    aileron: float, rudder: float):
        """Set control inputs"""
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
        if self.state is None:
            return np.zeros(3), np.zeros(3)
            
        # Get atmospheric conditions
        atm = self.atmosphere.get_atmospheric_conditions(self.state.altitude)
        rho = atm['density']
        
        # Calculate dynamic pressure
        q = 0.5 * rho * self.state.airspeed ** 2
        
        # Calculate angle of attack and sideslip
        alpha = np.arctan2(self.state.velocity_down, self.state.velocity_north)
        beta = np.arcsin(self.state.velocity_east / self.state.airspeed)
        
        # Lift coefficient
        cl = self.params.cl_alpha * alpha
        
        # Drag coefficient (simplified)
        cd = self.params.cd0 + (cl ** 2) / (np.pi * self.params.aspect_ratio * self.params.oswald_efficiency)
        
        # Aerodynamic forces
        lift = q * self.params.wing_area * cl
        drag = q * self.params.wing_area * cd
        
        # Convert to body frame
        forces = np.array([
            -drag,  # X-axis (forward)
            0.0,    # Y-axis (side)
            -lift   # Z-axis (down)
        ])
        
        # Aerodynamic moments (simplified)
        moments = np.array([
            q * self.params.wing_area * self.params.aileron_effectiveness * self.aileron,  # Roll
            q * self.params.wing_area * self.params.elevator_effectiveness * self.elevator,  # Pitch
            q * self.params.wing_area * self.params.rudder_effectiveness * self.rudder   # Yaw
        ])
        
        return forces, moments
        
    def calculate_propulsion_forces(self) -> np.ndarray:
        """Calculate propulsion forces"""
        if self.state is None:
            return np.zeros(3)
            
        # Get atmospheric conditions for thrust variation
        atm = self.atmosphere.get_atmospheric_conditions(self.state.altitude)
        rho_ratio = atm['density'] / AIR_DENSITY_SEA_LEVEL
        
        # Thrust varies with altitude
        thrust = self.params.thrust_max * self.throttle * rho_ratio ** 0.7
        
        # Fuel consumption
        fuel_consumption = thrust * self.params.specific_fuel_consumption * self.dt
        self.state.fuel_remaining -= fuel_consumption
        self.fuel_consumed += fuel_consumption
        
        return np.array([thrust, 0.0, 0.0])  # Thrust in forward direction
        
    def calculate_gravity_forces(self) -> np.ndarray:
        """Calculate gravitational forces"""
        if self.state is None:
            return np.zeros(3)
            
        # Gravitational acceleration varies with altitude
        g = GRAVITY * (6371000 / (6371000 + self.state.altitude)) ** 2
        mass = self.params.mass_empty + self.state.fuel_remaining
        
        return np.array([0.0, 0.0, mass * g])
        
    def update_state(self):
        """Update aircraft state using 6-DOF dynamics"""
        if self.state is None:
            return
            
        # Calculate forces and moments
        aero_forces, aero_moments = self.calculate_aerodynamic_forces()
        prop_forces = self.calculate_propulsion_forces()
        grav_forces = self.calculate_gravity_forces()
        
        total_forces = aero_forces + prop_forces + grav_forces
        
        # Simple Euler integration (in production, use RK4)
        mass = self.params.mass_empty + self.state.fuel_remaining
        
        # Update velocities
        self.state.velocity_north += total_forces[0] / mass * self.dt
        self.state.velocity_east += total_forces[1] / mass * self.dt
        self.state.velocity_down += total_forces[2] / mass * self.dt
        
        # Update angular rates
        self.state.roll_rate += aero_moments[0] / mass * self.dt
        self.state.pitch_rate += aero_moments[1] / mass * self.dt
        self.state.yaw_rate += aero_moments[2] / mass * self.dt
        
        # Update position (simplified - should use proper coordinate transformation)
        self.state.latitude += self.state.velocity_north / 6371000 * self.dt
        self.state.longitude += self.state.velocity_east / (6371000 * np.cos(self.state.latitude)) * self.dt
        self.state.altitude -= self.state.velocity_down * self.dt
        
        # Update attitude
        self.state.roll += self.state.roll_rate * self.dt
        self.state.pitch += self.state.pitch_rate * self.dt
        self.state.yaw += self.state.yaw_rate * self.dt
        
        # Update derived quantities
        self.state.airspeed = np.sqrt(
            self.state.velocity_north**2 + 
            self.state.velocity_east**2 + 
            self.state.velocity_down**2
        )
        
        self.state.ground_speed = np.sqrt(
            (self.state.velocity_north + self.state.wind_north)**2 +
            (self.state.velocity_east + self.state.wind_east)**2
        )
        
        self.state.heading = np.arctan2(self.state.velocity_east, self.state.velocity_north)
        self.state.flight_path_angle = np.arctan2(-self.state.velocity_down, 
                                                 np.sqrt(self.state.velocity_north**2 + self.state.velocity_east**2))
        
        # Update energy state
        self.state.kinetic_energy = 0.5 * mass * self.state.airspeed**2
        self.state.potential_energy = mass * GRAVITY * self.state.altitude
        self.state.total_energy = self.state.kinetic_energy + self.state.potential_energy
        
        # Update atmospheric conditions
        atm = self.atmosphere.get_atmospheric_conditions(self.state.altitude)
        self.state.air_density = atm['density']
        self.state.temperature = atm['temperature']
        self.state.pressure = atm['pressure']
        
        # Update time and distance
        self.time += self.dt
        self.flight_time += self.dt
        self.distance_traveled += self.state.ground_speed * self.dt
        
    def step(self):
        """Perform one simulation step"""
        self.update_state()
        
        # Update flight phase based on conditions
        self._update_flight_phase()
        
    def _update_flight_phase(self):
        """Update flight phase based on current state"""
        if self.state is None:
            return
            
        if self.state.altitude < 10:
            self.flight_phase = FlightPhase.GROUND
        elif self.state.velocity_down < -2 and self.state.altitude < 1000:
            self.flight_phase = FlightPhase.TAKEOFF
        elif self.state.velocity_down < -1:
            self.flight_phase = FlightPhase.CLIMB
        elif abs(self.state.velocity_down) < 1:
            self.flight_phase = FlightPhase.CRUISE
        elif self.state.velocity_down > 1:
            self.flight_phase = FlightPhase.DESCENT
            
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry data"""
        if self.state is None:
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
                'elevator': np.degrees(self.elevator),
                'aileron': np.degrees(self.aileron),
                'rudder': np.degrees(self.rudder)
            },
            'performance': {
                'flight_time': self.flight_time,
                'distance_traveled': self.distance_traveled,
                'fuel_consumed': self.fuel_consumed
            }
        }
        
    def check_flight_envelope(self) -> Dict[str, bool]:
        """Check if aircraft is within flight envelope"""
        if self.state is None:
            return {'valid': False}
            
        warnings = {
            'stall_warning': self.state.airspeed < self.params.stall_speed * 1.1,
            'overspeed_warning': self.state.airspeed > self.params.max_speed * 0.95,
            'altitude_warning': self.state.altitude > self.params.service_ceiling * 0.95,
            'fuel_warning': self.state.fuel_remaining < 0.1 * self.params.mass_max_takeoff,
            'valid': True
        }
        
        return warnings 