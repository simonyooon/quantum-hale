"""
HALE Drone Autonomy Engine
=========================

This module implements autonomous mission planning and execution for
High-Altitude Long-Endurance drones, including waypoint navigation,
mission planning, and autonomous decision making.

Author: Quantum HALE Development Team
License: MIT
"""

import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import math
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class MissionType(Enum):
    """Mission types for HALE drones"""
    ISR_PATROL = "isr_patrol"
    COMMUNICATION_RELAY = "communication_relay"
    WEATHER_MONITORING = "weather_monitoring"
    EMERGENCY_RESPONSE = "emergency_response"
    SURVEILLANCE = "surveillance"
    CARGO_TRANSPORT = "cargo_transport"


class MissionState(Enum):
    """Mission execution states"""
    PLANNING = "planning"
    EXECUTING = "executing"
    HOLDING = "holding"
    ABORTING = "aborting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Waypoint:
    """Mission waypoint definition"""
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters MSL
    speed: float  # m/s
    heading: Optional[float] = None  # degrees, None for auto-calculate
    action: str = "fly_to"  # fly_to, loiter, land, takeoff
    duration: float = 0.0  # seconds for loiter
    tolerance: float = 100.0  # meters position tolerance
    altitude_tolerance: float = 50.0  # meters altitude tolerance


@dataclass
class Mission:
    """Mission definition"""
    id: str
    type: MissionType
    waypoints: List[Waypoint]
    priority: int = 1  # 1-10, higher is more important
    fuel_reserve: float = 0.2  # 20% fuel reserve
    max_wind_speed: float = 20.0  # m/s
    max_turbulence: float = 5.0  # m/s
    emergency_landing_sites: List[Tuple[float, float]] = None  # lat, lon


class NavigationController:
    """Navigation and guidance controller"""
    
    def __init__(self):
        self.current_waypoint_index = 0
        self.waypoints = []
        self.navigation_mode = "waypoint"
        
        # Navigation parameters
        self.cross_track_tolerance = 50.0  # meters
        self.altitude_tolerance = 25.0  # meters
        self.speed_tolerance = 2.0  # m/s
        
        # PID controller gains
        self.heading_kp = 0.5
        self.heading_ki = 0.01
        self.heading_kd = 0.1
        
        self.altitude_kp = 0.3
        self.altitude_ki = 0.005
        self.altitude_kd = 0.05
        
        # Control history for PID
        self.heading_error_integral = 0.0
        self.altitude_error_integral = 0.0
        self.last_heading_error = 0.0
        self.last_altitude_error = 0.0
        
    def set_waypoints(self, waypoints: List[Waypoint]):
        """Set mission waypoints"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        
    def calculate_navigation_commands(self, current_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate navigation commands based on current state and waypoints
        
        Args:
            current_state: Current aircraft state from dynamics
            
        Returns:
            Dictionary with control commands
        """
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return {'throttle': 0.0, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
            
        current_wp = self.waypoints[self.current_waypoint_index]
        
        # Calculate distance to waypoint
        distance = self._calculate_distance(
            current_state['position']['latitude'],
            current_state['position']['longitude'],
            current_wp.latitude,
            current_wp.longitude
        )
        
        # Check if waypoint reached
        if distance < current_wp.tolerance:
            if current_wp.action == "loiter":
                # Continue loitering
                pass
            else:
                # Move to next waypoint
                self.current_waypoint_index += 1
                if self.current_waypoint_index >= len(self.waypoints):
                    return {'throttle': 0.0, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
                current_wp = self.waypoints[self.current_waypoint_index]
        
        # Calculate desired heading
        desired_heading = self._calculate_bearing(
            current_state['position']['latitude'],
            current_state['position']['longitude'],
            current_wp.latitude,
            current_wp.longitude
        )
        
        if current_wp.heading is not None:
            desired_heading = current_wp.heading
            
        # Calculate heading error
        current_heading = current_state['velocity']['heading']
        heading_error = self._normalize_angle(desired_heading - current_heading)
        
        # PID control for heading
        self.heading_error_integral += heading_error
        heading_derivative = heading_error - self.last_heading_error
        heading_command = (self.heading_kp * heading_error + 
                          self.heading_ki * self.heading_error_integral +
                          self.heading_kd * heading_derivative)
        
        self.last_heading_error = heading_error
        
        # Calculate altitude error
        altitude_error = current_wp.altitude - current_state['position']['altitude']
        
        # PID control for altitude
        self.altitude_error_integral += altitude_error
        altitude_derivative = altitude_error - self.last_altitude_error
        altitude_command = (self.altitude_kp * altitude_error +
                           self.altitude_ki * self.altitude_error_integral +
                           self.altitude_kd * altitude_derivative)
        
        self.last_altitude_error = altitude_error
        
        # Calculate speed error
        speed_error = current_wp.speed - current_state['velocity']['airspeed']
        
        # Convert to control commands
        throttle = 0.5 + 0.3 * speed_error  # Base throttle + speed correction
        throttle = np.clip(throttle, 0.0, 1.0)
        
        # Convert heading command to aileron/rudder
        aileron = np.clip(heading_command * 0.1, -0.5, 0.5)
        rudder = np.clip(heading_command * 0.05, -0.3, 0.3)
        
        # Convert altitude command to elevator
        elevator = np.clip(altitude_command * 0.1, -0.5, 0.5)
        
        return {
            'throttle': throttle,
            'elevator': elevator,
            'aileron': aileron,
            'rudder': rudder
        }
        
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
        
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = (np.cos(lat1_rad) * np.sin(lat2_rad) - 
             np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon))
        
        bearing = np.degrees(np.arctan2(y, x))
        return (bearing + 360) % 360
        
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-180, 180] degrees"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle


class MissionPlanner:
    """Mission planning and optimization"""
    
    def __init__(self):
        self.weather_data = {}
        self.airspace_restrictions = []
        self.performance_model = None
        
    def plan_mission(self, mission: Mission, 
                    current_position: Tuple[float, float, float],
                    aircraft_performance: Dict[str, Any]) -> List[Waypoint]:
        """
        Plan optimal mission route
        
        Args:
            mission: Mission definition
            current_position: Current (lat, lon, alt) position
            aircraft_performance: Aircraft performance parameters
            
        Returns:
            Optimized waypoint list
        """
        waypoints = []
        
        # Add takeoff waypoint if needed
        if current_position[2] < 100:  # Below 100m altitude
            waypoints.append(Waypoint(
                latitude=current_position[0],
                longitude=current_position[1],
                altitude=mission.waypoints[0].altitude,
                speed=mission.waypoints[0].speed,
                action="takeoff"
            ))
        
        # Add mission waypoints
        for wp in mission.waypoints:
            waypoints.append(wp)
            
        # Add landing waypoint if needed
        if mission.emergency_landing_sites:
            # Find nearest emergency landing site
            nearest_site = min(mission.emergency_landing_sites,
                             key=lambda site: self._calculate_distance(
                                 waypoints[-1].latitude, waypoints[-1].longitude,
                                 site[0], site[1]
                             ))
            
            waypoints.append(Waypoint(
                latitude=nearest_site[0],
                longitude=nearest_site[1],
                altitude=0,
                speed=15.0,  # Approach speed
                action="land"
            ))
        
        return waypoints
        
    def optimize_route(self, waypoints: List[Waypoint], 
                      constraints: Dict[str, Any]) -> List[Waypoint]:
        """
        Optimize route considering constraints
        
        Args:
            waypoints: Initial waypoint list
            constraints: Route constraints (weather, airspace, etc.)
            
        Returns:
            Optimized waypoint list
        """
        # Simple optimization - in production, use more sophisticated algorithms
        optimized_waypoints = []
        
        for i, wp in enumerate(waypoints):
            # Check weather constraints
            if self._check_weather_constraints(wp, constraints.get('weather', {})):
                optimized_waypoints.append(wp)
            else:
                # Find alternative route around weather
                alt_wp = self._find_weather_avoidance_route(wp, constraints.get('weather', {}))
                if alt_wp:
                    optimized_waypoints.extend(alt_wp)
                    
        return optimized_waypoints
        
    def _check_weather_constraints(self, waypoint: Waypoint, weather: Dict[str, Any]) -> bool:
        """Check if waypoint satisfies weather constraints"""
        # Simplified weather check
        return True  # Placeholder
        
    def _find_weather_avoidance_route(self, waypoint: Waypoint, weather: Dict[str, Any]) -> List[Waypoint]:
        """Find route to avoid weather"""
        # Simplified weather avoidance
        return [waypoint]  # Placeholder


class AutonomyEngine:
    """
    Main autonomy engine for HALE drone
    
    Coordinates mission planning, navigation, and autonomous decision making
    """
    
    def __init__(self):
        self.navigation_controller = NavigationController()
        self.mission_planner = MissionPlanner()
        self.current_mission = None
        self.mission_state = MissionState.PLANNING
        
        # Autonomous decision making
        self.emergency_handlers = {}
        self.mission_handlers = {}
        
        # Performance monitoring
        self.mission_start_time = None
        self.mission_progress = 0.0
        self.fuel_consumption_rate = 0.0
        
        logging.info("Autonomy Engine initialized")
        
    def load_mission(self, mission: Mission):
        """Load and plan mission"""
        self.current_mission = mission
        self.mission_state = MissionState.PLANNING
        
        # Plan mission route
        current_position = (0.0, 0.0, 0.0)  # Should get from aircraft state
        aircraft_performance = {}  # Should get from aircraft parameters
        
        waypoints = self.mission_planner.plan_mission(
            mission, current_position, aircraft_performance
        )
        
        # Optimize route
        constraints = {
            'weather': self.mission_planner.weather_data,
            'airspace': self.mission_planner.airspace_restrictions
        }
        
        optimized_waypoints = self.mission_planner.optimize_route(waypoints, constraints)
        
        # Set waypoints for navigation
        self.navigation_controller.set_waypoints(optimized_waypoints)
        
        self.mission_state = MissionState.EXECUTING
        self.mission_start_time = time.time()
        
        logging.info(f"Mission {mission.id} loaded with {len(optimized_waypoints)} waypoints")
        
    def update(self, current_state: Dict[str, Any], 
               environment_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Update autonomy engine and return control commands
        
        Args:
            current_state: Current aircraft state
            environment_data: Environmental data (weather, threats, etc.)
            
        Returns:
            Control commands for aircraft
        """
        if self.mission_state != MissionState.EXECUTING:
            return {'throttle': 0.0, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
            
        # Check for emergency conditions
        if self._check_emergency_conditions(current_state, environment_data):
            self._handle_emergency(current_state, environment_data)
            return self._get_emergency_commands(current_state)
            
        # Update mission progress
        self._update_mission_progress(current_state)
        
        # Get navigation commands
        commands = self.navigation_controller.calculate_navigation_commands(current_state)
        
        # Apply mission-specific modifications
        commands = self._apply_mission_modifications(commands, current_state)
        
        return commands
        
    def _check_emergency_conditions(self, current_state: Dict[str, Any], 
                                  environment_data: Dict[str, Any]) -> bool:
        """Check for emergency conditions requiring immediate action"""
        emergency_reasons = []
        
        # Check fuel level
        fuel_remaining = current_state.get('energy', {}).get('fuel_remaining', 100.0)
        if fuel_remaining < 0.1:  # Less than 10%
            emergency_reasons.append(f"Low fuel: {fuel_remaining:.2f}% remaining")
            
        # Check flight envelope violations
        warnings = current_state.get('warnings', {})
        if warnings.get('stall_warning', False):
            emergency_reasons.append("Stall warning detected")
        if warnings.get('overspeed_warning', False):
            emergency_reasons.append("Overspeed warning detected")
            
        # Check weather conditions
        wind_speed = environment_data.get('wind_speed', 0)
        if wind_speed > 25.0:  # High winds
            emergency_reasons.append(f"High winds: {wind_speed:.1f} m/s")
            
        # Check for threats
        if environment_data.get('threats', {}).get('jamming_active', False):
            emergency_reasons.append("Jamming threat detected")
            
        # Store emergency reasons for detailed reporting
        if emergency_reasons:
            self.emergency_reasons = emergency_reasons
            return True
            
        return False
        
    def _handle_emergency(self, current_state: Dict[str, Any], 
                         environment_data: Dict[str, Any]):
        """Handle emergency conditions"""
        self.mission_state = MissionState.ABORTING
        
        # Log detailed emergency information
        if hasattr(self, 'emergency_reasons'):
            for reason in self.emergency_reasons:
                logging.warning(f"EMERGENCY: {reason}")
            logging.warning(f"Mission aborting due to {len(self.emergency_reasons)} emergency condition(s)")
        else:
            logging.warning("Emergency condition detected - aborting mission (details unavailable)")
        
        # Find nearest safe landing site
        if self.current_mission and self.current_mission.emergency_landing_sites:
            nearest_site = min(self.current_mission.emergency_landing_sites,
                             key=lambda site: self._calculate_distance(
                                 current_state['position']['latitude'],
                                 current_state['position']['longitude'],
                                 site[0], site[1]
                             ))
            
            # Create emergency landing waypoint
            emergency_wp = Waypoint(
                latitude=nearest_site[0],
                longitude=nearest_site[1],
                altitude=0,
                speed=15.0,
                action="land"
            )
            
            self.navigation_controller.set_waypoints([emergency_wp])
            logging.info(f"Emergency landing waypoint set: {nearest_site}")
        else:
            logging.warning("No emergency landing sites available")
        
    def _get_emergency_commands(self, current_state: Dict[str, Any]) -> Dict[str, float]:
        """Get control commands for emergency situations"""
        # Conservative emergency commands
        return {
            'throttle': 0.3,  # Reduced throttle
            'elevator': 0.0,  # Level flight
            'aileron': 0.0,   # No roll
            'rudder': 0.0     # No yaw
        }
        
    def _update_mission_progress(self, current_state: Dict[str, Any]):
        """Update mission progress tracking"""
        if not self.current_mission or not self.mission_start_time:
            return
            
        total_waypoints = len(self.current_mission.waypoints)
        completed_waypoints = self.navigation_controller.current_waypoint_index
        
        self.mission_progress = completed_waypoints / total_waypoints
        
        # Update fuel consumption
        self.fuel_consumption_rate = current_state['energy'].get('fuel_consumption_rate', 0.0)
        
    def _apply_mission_modifications(self, commands: Dict[str, float], 
                                   current_state: Dict[str, Any]) -> Dict[str, float]:
        """Apply mission-specific modifications to control commands"""
        if not self.current_mission:
            return commands
            
        # Mission-specific modifications
        if self.current_mission.type == MissionType.ISR_PATROL:
            # Optimize for surveillance - slower, more stable flight
            commands['throttle'] *= 0.8
            
        elif self.current_mission.type == MissionType.COMMUNICATION_RELAY:
            # Optimize for communication - maintain altitude and position
            commands['elevator'] *= 0.5  # Reduced pitch changes
            
        return commands
        
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
        
    def get_mission_status(self) -> Dict[str, Any]:
        """Get current mission status"""
        return {
            'mission_id': self.current_mission.id if self.current_mission else None,
            'mission_type': self.current_mission.type.value if self.current_mission else None,
            'mission_state': self.mission_state.value,
            'progress': self.mission_progress,
            'waypoint_index': self.navigation_controller.current_waypoint_index,
            'total_waypoints': len(self.navigation_controller.waypoints),
            'fuel_consumption_rate': self.fuel_consumption_rate,
            'mission_time': time.time() - self.mission_start_time if self.mission_start_time else 0.0
        } 