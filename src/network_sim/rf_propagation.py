"""
RF Propagation Models for HALE Drone Network Simulation

This module provides various radio frequency propagation models
for simulating wireless communication between HALE drones.
"""

import math
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class PropagationModel(Enum):
    """Types of RF propagation models"""
    FREE_SPACE = "free_space"
    TWO_RAY = "two_ray"
    ITU_R = "itu_r"
    HATA = "hata"
    COST_231 = "cost_231"


@dataclass
class PropagationResult:
    """Results from propagation calculation"""
    path_loss: float  # dB
    received_power: float  # dBm
    signal_strength: float  # dBm
    snr: float  # dB
    link_budget: float  # dB


class RFPropagation:
    """
    Radio Frequency propagation modeling for HALE drone communications
    """
    
    def __init__(self, model: PropagationModel = PropagationModel.FREE_SPACE):
        self.model = model
        self.frequency = 2.4e9  # Hz (2.4 GHz)
        self.wavelength = 3e8 / self.frequency  # meters
        
        # Atmospheric parameters
        self.temperature = 288.15  # Kelvin (15°C)
        self.pressure = 101325  # Pa (1 atm)
        self.humidity = 50  # % relative humidity
        
        logging.info(f"RF Propagation initialized with {model.value} model")
    
    def calculate_path_loss(self, distance: float, tx_height: float, 
                           rx_height: float, **kwargs) -> float:
        """
        Calculate path loss using the selected propagation model
        
        Args:
            distance: Distance between transmitter and receiver (meters)
            tx_height: Transmitter height above ground (meters)
            rx_height: Receiver height above ground (meters)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Path loss in dB
        """
        if self.model == PropagationModel.FREE_SPACE:
            return self._free_space_path_loss(distance)
        elif self.model == PropagationModel.TWO_RAY:
            return self._two_ray_path_loss(distance, tx_height, rx_height)
        elif self.model == PropagationModel.ITU_R:
            return self._itu_r_path_loss(distance, tx_height, rx_height, **kwargs)
        elif self.model == PropagationModel.HATA:
            return self._hata_path_loss(distance, tx_height, rx_height, **kwargs)
        else:
            raise ValueError(f"Unsupported propagation model: {self.model}")
    
    def _free_space_path_loss(self, distance: float) -> float:
        """Free space path loss model"""
        # FSPL = 20 * log10(4π * d / λ)
        path_loss = 20 * math.log10(4 * math.pi * distance / self.wavelength)
        return path_loss
    
    def _two_ray_path_loss(self, distance: float, tx_height: float, 
                          rx_height: float) -> float:
        """Two-ray ground reflection model"""
        # Calculate direct and reflected path distances
        d_direct = math.sqrt(distance**2 + (tx_height - rx_height)**2)
        d_reflected = math.sqrt(distance**2 + (tx_height + rx_height)**2)
        
        # Path loss calculation
        if distance < 4 * math.pi * tx_height * rx_height / self.wavelength:
            # Use free space model for short distances
            return self._free_space_path_loss(distance)
        else:
            # Use two-ray model for longer distances
            path_loss = 40 * math.log10(distance) - 20 * math.log10(tx_height * rx_height)
            return path_loss
    
    def _itu_r_path_loss(self, distance: float, tx_height: float, 
                        rx_height: float, **kwargs) -> float:
        """ITU-R P.525-4 path loss model for line-of-sight"""
        # Get additional parameters
        terrain_type = kwargs.get('terrain_type', 'average')
        
        # Base path loss
        base_loss = self._free_space_path_loss(distance)
        
        # Terrain factor
        terrain_factors = {
            'smooth': 0,
            'average': 3,
            'rough': 6,
            'very_rough': 9
        }
        terrain_factor = terrain_factors.get(terrain_type, 3)
        
        # Atmospheric absorption (simplified)
        atmospheric_loss = self._calculate_atmospheric_loss(distance)
        
        total_loss = base_loss + terrain_factor + atmospheric_loss
        return total_loss
    
    def _hata_path_loss(self, distance: float, tx_height: float, 
                       rx_height: float, **kwargs) -> float:
        """Hata model for urban/suburban areas"""
        # Hata model parameters
        frequency_mhz = self.frequency / 1e6
        
        # Base path loss for urban area
        base_loss = 69.55 + 26.16 * math.log10(frequency_mhz) - 13.82 * math.log10(tx_height)
        base_loss += (44.9 - 6.55 * math.log10(tx_height)) * math.log10(distance / 1000)
        
        # Correction factor for receiver height
        if rx_height <= 3:
            correction = (1.1 * math.log10(frequency_mhz) - 0.7) * rx_height
            correction -= (1.56 * math.log10(frequency_mhz) - 0.8)
        else:
            correction = 3.2 * (math.log10(11.75 * rx_height))**2 - 4.97
        
        # Environment correction
        environment = kwargs.get('environment', 'urban')
        if environment == 'suburban':
            correction += -2 * (math.log10(frequency_mhz / 28))**2 + 5.4
        elif environment == 'rural':
            correction += -4.78 * (math.log10(frequency_mhz))**2 + 18.33 * math.log10(frequency_mhz) - 40.94
        
        return base_loss + correction
    
    def _calculate_atmospheric_loss(self, distance: float) -> float:
        """Calculate atmospheric absorption loss"""
        # Simplified atmospheric absorption model
        # Based on ITU-R P.676-12 for 2.4 GHz
        absorption_rate = 0.0001  # dB/km (very low at 2.4 GHz)
        return absorption_rate * distance / 1000
    
    def calculate_link_budget(self, tx_power: float, tx_gain: float, 
                            rx_gain: float, distance: float, tx_height: float,
                            rx_height: float, **kwargs) -> PropagationResult:
        """
        Calculate complete link budget
        
        Args:
            tx_power: Transmitter power (dBm)
            tx_gain: Transmitter antenna gain (dBi)
            rx_gain: Receiver antenna gain (dBi)
            distance: Distance between antennas (meters)
            tx_height: Transmitter height (meters)
            rx_height: Receiver height (meters)
            **kwargs: Additional parameters
            
        Returns:
            PropagationResult with link budget analysis
        """
        # Calculate path loss
        path_loss = self.calculate_path_loss(distance, tx_height, rx_height, **kwargs)
        
        # Calculate received power
        received_power = tx_power + tx_gain + rx_gain - path_loss
        
        # Calculate signal strength (same as received power for this model)
        signal_strength = received_power
        
        # Calculate SNR (assuming thermal noise)
        noise_power = self._calculate_thermal_noise()
        snr = received_power - noise_power
        
        # Calculate link budget margin
        link_budget = received_power - noise_power
        
        return PropagationResult(
            path_loss=path_loss,
            received_power=received_power,
            signal_strength=signal_strength,
            snr=snr,
            link_budget=link_budget
        )
    
    def _calculate_thermal_noise(self, bandwidth: float = 20e6) -> float:
        """Calculate thermal noise power"""
        # Thermal noise = k * T * B
        # k = Boltzmann constant, T = temperature, B = bandwidth
        k = 1.380649e-23  # J/K
        noise_power_watts = k * self.temperature * bandwidth
        noise_power_dbm = 10 * math.log10(noise_power_watts * 1000)
        return noise_power_dbm
    
    def calculate_coverage_area(self, tx_power: float, tx_gain: float,
                              rx_gain: float, tx_height: float, rx_height: float,
                              min_snr: float = 10.0, **kwargs) -> float:
        """
        Calculate coverage area for given parameters
        
        Args:
            tx_power: Transmitter power (dBm)
            tx_gain: Transmitter antenna gain (dBi)
            rx_gain: Receiver antenna gain (dBi)
            tx_height: Transmitter height (meters)
            rx_height: Receiver height (meters)
            min_snr: Minimum required SNR (dB)
            **kwargs: Additional parameters
            
        Returns:
            Coverage radius in meters
        """
        # Calculate required received power for minimum SNR
        noise_power = self._calculate_thermal_noise()
        required_rx_power = noise_power + min_snr
        
        # Calculate maximum allowed path loss
        max_path_loss = tx_power + tx_gain + rx_gain - required_rx_power
        
        # Find distance that gives this path loss (iterative)
        max_distance = self._find_distance_for_path_loss(max_path_loss, tx_height, rx_height, **kwargs)
        
        return max_distance
    
    def _find_distance_for_path_loss(self, target_path_loss: float, tx_height: float,
                                   rx_height: float, **kwargs) -> float:
        """Find distance that gives the target path loss"""
        # Binary search for distance
        min_distance = 1.0  # meters
        max_distance = 100000.0  # meters (100 km)
        tolerance = 0.1  # meters
        
        while max_distance - min_distance > tolerance:
            mid_distance = (min_distance + max_distance) / 2
            path_loss = self.calculate_path_loss(mid_distance, tx_height, rx_height, **kwargs)
            
            if path_loss < target_path_loss:
                min_distance = mid_distance
            else:
                max_distance = mid_distance
        
        return mid_distance
    
    def calculate_fresnel_zone(self, distance: float, frequency: Optional[float] = None) -> float:
        """
        Calculate first Fresnel zone radius
        
        Args:
            distance: Total path distance (meters)
            frequency: Frequency (Hz), uses default if None
            
        Returns:
            First Fresnel zone radius (meters)
        """
        if frequency is None:
            frequency = self.frequency
        
        wavelength = 3e8 / frequency
        fresnel_radius = math.sqrt(wavelength * distance / 4)
        return fresnel_radius
    
    def calculate_multipath_effects(self, distance: float, tx_height: float,
                                  rx_height: float, **kwargs) -> Dict[str, float]:
        """
        Calculate multipath effects
        
        Args:
            distance: Distance between antennas (meters)
            tx_height: Transmitter height (meters)
            rx_height: Receiver height (meters)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with multipath parameters
        """
        # Calculate direct and reflected paths
        d_direct = math.sqrt(distance**2 + (tx_height - rx_height)**2)
        d_reflected = math.sqrt(distance**2 + (tx_height + rx_height)**2)
        
        # Path difference
        path_difference = d_reflected - d_direct
        
        # Phase difference
        phase_difference = 2 * math.pi * path_difference / self.wavelength
        
        # Reflection coefficient (simplified)
        reflection_coefficient = 0.3  # Typical for ground reflection
        
        # Multipath fading factor
        multipath_factor = 1 + reflection_coefficient * math.cos(phase_difference)
        
        # Convert to dB
        multipath_loss_db = 20 * math.log10(multipath_factor)
        
        return {
            'path_difference': path_difference,
            'phase_difference': phase_difference,
            'multipath_factor': multipath_factor,
            'multipath_loss_db': multipath_loss_db
        }
    
    def set_frequency(self, frequency: float):
        """Set operating frequency"""
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        logging.info(f"Frequency set to {frequency/1e9:.2f} GHz")
    
    def set_atmospheric_conditions(self, temperature: float, pressure: float, humidity: float):
        """Set atmospheric conditions for propagation calculations"""
        self.temperature = temperature
        self.pressure = pressure
        self.humidity = humidity
        logging.info(f"Atmospheric conditions updated: T={temperature}K, P={pressure}Pa, H={humidity}%")
    
    def update_position(self, position: Tuple[float, float, float]):
        """Update the current position for propagation calculations"""
        self.current_position = position
        logging.debug(f"RF Propagation position updated to {position}")
    
    def get_current_position(self) -> Optional[Tuple[float, float, float]]:
        """Get the current position"""
        return getattr(self, 'current_position', None)
    
    def get_current_conditions(self) -> Dict[str, Any]:
        """Get current atmospheric and propagation conditions"""
        return {
            'frequency': self.frequency,
            'wavelength': self.wavelength,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'humidity': self.humidity,
            'position': self.get_current_position(),
            'model': self.model.value,
            'atmospheric_loss_rate': self._calculate_atmospheric_loss(1000) / 1000  # dB/m
        } 