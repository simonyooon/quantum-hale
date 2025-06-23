"""
Enhanced RF Propagation Models for HALE Drone Network Simulation

This module provides advanced radio frequency propagation models
for simulating wireless communication between HALE drones with features:
- High-altitude atmospheric modeling
- Multi-frequency band support (2.4GHz, 5GHz, satellite bands)
- Advanced propagation models (ITU-R, COST-231, Hata, etc.)
- Atmospheric absorption and scattering effects
- Tropospheric and ionospheric effects
- Fresnel zone calculations
- Multipath and fading models
"""

import math
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class PropagationModel(Enum):
    """Types of RF propagation models"""
    FREE_SPACE = "free_space"
    TWO_RAY = "two_ray"
    ITU_R = "itu_r"
    HATA = "hata"
    COST_231 = "cost_231"
    ITM = "itm"  # Irregular Terrain Model
    TIREM = "tirem"  # Terrain Integrated Rough Earth Model
    ADVANCED_ITU_R = "advanced_itu_r"


class FrequencyBand(Enum):
    """Frequency bands for HALE drone communications"""
    L_BAND = (1.5e9, 1.6e9)  # Satellite communications
    S_BAND = (2.0e9, 4.0e9)  # Weather radar, satellite
    C_BAND = (4.0e9, 8.0e9)  # Satellite communications
    X_BAND = (8.0e9, 12.0e9)  # Military, weather radar
    KU_BAND = (12.0e9, 18.0e9)  # Satellite communications
    KA_BAND = (26.5e9, 40.0e9)  # Satellite communications
    WIFI_2_4 = (2.4e9, 2.5e9)  # WiFi 2.4 GHz
    WIFI_5 = (5.15e9, 5.85e9)  # WiFi 5 GHz


class AtmosphericCondition(Enum):
    """Atmospheric conditions affecting propagation"""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"
    DUST = "dust"
    TURBULENCE = "turbulence"


@dataclass
class AtmosphericParams:
    """Atmospheric parameters for propagation modeling"""
    temperature: float = 288.15  # Kelvin (15°C)
    pressure: float = 101325  # Pa (1 atm)
    humidity: float = 50  # % relative humidity
    rain_rate: float = 0.0  # mm/h
    visibility: float = 10000  # meters
    wind_speed: float = 0.0  # m/s
    turbulence_strength: float = 0.0  # Cn2 parameter
    condition: AtmosphericCondition = AtmosphericCondition.CLEAR


@dataclass
class PropagationResult:
    """Enhanced results from propagation calculation"""
    path_loss: float  # dB
    received_power: float  # dBm
    signal_strength: float  # dBm
    snr: float  # dB
    link_budget: float  # dB
    atmospheric_loss: float  # dB
    multipath_loss: float  # dB
    fading_margin: float  # dB
    fresnel_clearance: float  # meters
    availability: float  # 0.0 to 1.0


@dataclass
class FadingModel:
    """Fading model parameters"""
    rician_k: float = 10.0  # Rician K-factor (dB)
    rayleigh_probability: float = 0.1  # Probability of Rayleigh fading
    shadowing_std: float = 4.0  # Shadowing standard deviation (dB)
    doppler_shift: float = 0.0  # Hz


class EnhancedRFPropagation:
    """
    Enhanced Radio Frequency propagation modeling for HALE drone communications
    """
    
    def __init__(self, model: PropagationModel = PropagationModel.ADVANCED_ITU_R,
                 frequency_band: FrequencyBand = FrequencyBand.WIFI_2_4):
        self.model = model
        self.frequency_band = frequency_band
        self.frequency = (frequency_band.value[0] + frequency_band.value[1]) / 2
        self.wavelength = 3e8 / self.frequency  # meters
        
        # Atmospheric parameters
        self.atmospheric = AtmosphericParams()
        
        # Fading model
        self.fading = FadingModel()
        
        # Earth parameters
        self.earth_radius = 6371000  # meters
        self.effective_earth_radius = self.earth_radius * 4/3  # Standard atmosphere
        
        # Cache for calculations
        self._path_loss_cache: Dict[Tuple, float] = {}
        self._atmospheric_loss_cache: Dict[Tuple, float] = {}
        
        logging.info(f"Enhanced RF Propagation initialized with {model.value} model at {self.frequency/1e9:.1f} GHz")
    
    def set_frequency_band(self, frequency_band: FrequencyBand):
        """Set frequency band for propagation calculations"""
        self.frequency_band = frequency_band
        self.frequency = (frequency_band.value[0] + frequency_band.value[1]) / 2
        self.wavelength = 3e8 / self.frequency
        self._clear_cache()
        logging.info(f"Frequency band set to {frequency_band.name} ({self.frequency/1e9:.1f} GHz)")
    
    def set_atmospheric_conditions(self, atmospheric: AtmosphericParams):
        """Set atmospheric conditions"""
        self.atmospheric = atmospheric
        self._clear_cache()
        logging.info(f"Atmospheric conditions set: {atmospheric.condition.value}")
    
    def _clear_cache(self):
        """Clear calculation caches"""
        self._path_loss_cache.clear()
        self._atmospheric_loss_cache.clear()
    
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
        # Check cache first
        cache_key = (distance, tx_height, rx_height, self.frequency, self.atmospheric.condition.value)
        if cache_key in self._path_loss_cache:
            return self._path_loss_cache[cache_key]
        
        if self.model == PropagationModel.FREE_SPACE:
            path_loss = self._free_space_path_loss(distance)
        elif self.model == PropagationModel.TWO_RAY:
            path_loss = self._two_ray_path_loss(distance, tx_height, rx_height)
        elif self.model == PropagationModel.ITU_R:
            path_loss = self._itu_r_path_loss(distance, tx_height, rx_height, **kwargs)
        elif self.model == PropagationModel.HATA:
            path_loss = self._hata_path_loss(distance, tx_height, rx_height, **kwargs)
        elif self.model == PropagationModel.COST_231:
            path_loss = self._cost_231_path_loss(distance, tx_height, rx_height, **kwargs)
        elif self.model == PropagationModel.ITM:
            path_loss = self._itm_path_loss(distance, tx_height, rx_height, **kwargs)
        elif self.model == PropagationModel.TIREM:
            path_loss = self._tirem_path_loss(distance, tx_height, rx_height, **kwargs)
        elif self.model == PropagationModel.ADVANCED_ITU_R:
            path_loss = self._advanced_itu_r_path_loss(distance, tx_height, rx_height, **kwargs)
        else:
            raise ValueError(f"Unsupported propagation model: {self.model}")
        
        # Cache result
        self._path_loss_cache[cache_key] = path_loss
        
        return path_loss
    
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
        
        # Atmospheric absorption
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
    
    def _cost_231_path_loss(self, distance: float, tx_height: float, 
                           rx_height: float, **kwargs) -> float:
        """COST-231 Hata model for urban areas"""
        frequency_mhz = self.frequency / 1e6
        
        # Base path loss
        base_loss = 46.3 + 33.9 * math.log10(frequency_mhz) - 13.82 * math.log10(tx_height)
        base_loss += (44.9 - 6.55 * math.log10(tx_height)) * math.log10(distance / 1000)
        
        # Receiver height correction
        if rx_height <= 3:
            correction = (1.1 * math.log10(frequency_mhz) - 0.7) * rx_height
            correction -= (1.56 * math.log10(frequency_mhz) - 0.8)
        else:
            correction = 3.2 * (math.log10(11.75 * rx_height))**2 - 4.97
        
        # City size correction
        city_size = kwargs.get('city_size', 'medium')
        if city_size == 'small':
            correction += -2 * (math.log10(frequency_mhz / 28))**2 + 5.4
        elif city_size == 'large':
            correction += 3
        
        return base_loss + correction
    
    def _itm_path_loss(self, distance: float, tx_height: float, 
                      rx_height: float, **kwargs) -> float:
        """Irregular Terrain Model (Longley-Rice)"""
        # Simplified ITM implementation
        # In practice, this would use the full ITM algorithm
        
        # Base free space loss
        base_loss = self._free_space_path_loss(distance)
        
        # Terrain roughness factor
        terrain_roughness = kwargs.get('terrain_roughness', 50)  # meters
        roughness_factor = 10 * math.log10(1 + terrain_roughness / 100)
        
        # Height factor
        height_factor = 20 * math.log10((tx_height + rx_height) / 2)
        
        return base_loss + roughness_factor + height_factor
    
    def _tirem_path_loss(self, distance: float, tx_height: float, 
                        rx_height: float, **kwargs) -> float:
        """Terrain Integrated Rough Earth Model"""
        # Simplified TIREM implementation
        # Similar to ITM but with different terrain modeling
        
        base_loss = self._free_space_path_loss(distance)
        
        # Terrain factor
        terrain_type = kwargs.get('terrain_type', 'average')
        terrain_factors = {
            'smooth': 2,
            'average': 5,
            'rough': 8,
            'very_rough': 12
        }
        terrain_factor = terrain_factors.get(terrain_type, 5)
        
        return base_loss + terrain_factor
    
    def _advanced_itu_r_path_loss(self, distance: float, tx_height: float, 
                                 rx_height: float, **kwargs) -> float:
        """Advanced ITU-R model with comprehensive atmospheric effects"""
        # Base free space loss
        base_loss = self._free_space_path_loss(distance)
        
        # Atmospheric absorption loss
        atmospheric_loss = self._calculate_atmospheric_loss(distance)
        
        # Tropospheric effects
        tropo_loss = self._calculate_tropospheric_loss(distance, tx_height, rx_height)
        
        # Ionospheric effects (for higher frequencies)
        iono_loss = self._calculate_ionospheric_loss(distance)
        
        # Multipath effects
        multipath_loss = self._calculate_multipath_loss(distance, tx_height, rx_height)
        
        # Terrain effects
        terrain_loss = self._calculate_terrain_loss(distance, tx_height, rx_height, **kwargs)
        
        total_loss = (base_loss + atmospheric_loss + tropo_loss + 
                     iono_loss + multipath_loss + terrain_loss)
        
        return total_loss
    
    def _calculate_atmospheric_loss(self, distance: float) -> float:
        """Calculate atmospheric absorption loss"""
        # Check cache
        cache_key = (distance, self.frequency, self.atmospheric.condition.value)
        if cache_key in self._atmospheric_loss_cache:
            return self._atmospheric_loss_cache[cache_key]
        
        # Atmospheric absorption based on ITU-R P.676-12
        frequency_ghz = self.frequency / 1e9
        
        # Oxygen absorption
        if frequency_ghz < 60:
            # Simplified oxygen absorption model
            oxygen_absorption = 0.0001 * frequency_ghz**2  # dB/km
        else:
            oxygen_absorption = 0.1  # dB/km
        
        # Water vapor absorption
        if frequency_ghz < 350:
            # Simplified water vapor absorption model
            water_vapor_absorption = 0.00005 * frequency_ghz**2 * (self.atmospheric.humidity / 100)  # dB/km
        else:
            water_vapor_absorption = 0.05  # dB/km
        
        # Rain absorption (if applicable)
        rain_absorption = 0.0
        if self.atmospheric.condition == AtmosphericCondition.RAIN:
            # Simplified rain absorption model
            rain_absorption = 0.001 * frequency_ghz * (self.atmospheric.rain_rate / 10)  # dB/km
        
        # Total atmospheric absorption
        total_absorption = oxygen_absorption + water_vapor_absorption + rain_absorption
        
        # Convert to total loss over distance
        atmospheric_loss = total_absorption * distance / 1000  # dB
        
        # Cache result
        self._atmospheric_loss_cache[cache_key] = atmospheric_loss
        
        return atmospheric_loss
    
    def _calculate_tropospheric_loss(self, distance: float, tx_height: float, 
                                   rx_height: float) -> float:
        """Calculate tropospheric effects"""
        # Tropospheric scintillation and ducting effects
        # Simplified model based on height and distance
        
        avg_height = (tx_height + rx_height) / 2
        
        # Scintillation loss (increases with height and frequency)
        frequency_ghz = self.frequency / 1e9
        scintillation_loss = 0.01 * frequency_ghz * (avg_height / 10000) * (distance / 1000)
        
        # Ducting effects (simplified)
        ducting_loss = 0.0
        if avg_height < 1000:  # Low altitude ducting
            ducting_loss = 0.1 * (distance / 1000)
        
        return scintillation_loss + ducting_loss
    
    def _calculate_ionospheric_loss(self, distance: float) -> float:
        """Calculate ionospheric effects"""
        # Ionospheric effects are minimal for frequencies above ~30 MHz
        # but can be significant for lower frequencies
        
        frequency_mhz = self.frequency / 1e6
        
        if frequency_mhz < 30:
            # Significant ionospheric effects
            iono_loss = 5.0 * (30 / frequency_mhz) * (distance / 1000)
        else:
            # Negligible ionospheric effects
            iono_loss = 0.0
        
        return iono_loss
    
    def _calculate_multipath_loss(self, distance: float, tx_height: float, 
                                rx_height: float) -> float:
        """Calculate multipath effects"""
        # Multipath fading effects
        
        # Frequency-dependent multipath
        frequency_ghz = self.frequency / 1e9
        multipath_factor = 0.1 * frequency_ghz * (distance / 1000)
        
        # Height-dependent multipath
        height_factor = 1.0 / (1.0 + (tx_height + rx_height) / 10000)
        
        return multipath_factor * height_factor
    
    def _calculate_terrain_loss(self, distance: float, tx_height: float, 
                              rx_height: float, **kwargs) -> float:
        """Calculate terrain effects"""
        terrain_type = kwargs.get('terrain_type', 'average')
        
        terrain_factors = {
            'smooth': 0,
            'average': 2,
            'rough': 5,
            'very_rough': 8,
            'mountainous': 12
        }
        
        base_terrain_loss = terrain_factors.get(terrain_type, 2)
        
        # Distance-dependent terrain loss
        distance_factor = math.log10(distance / 1000 + 1)
        
        return base_terrain_loss * distance_factor
    
    def calculate_link_budget(self, tx_power: float, tx_gain: float, 
                            rx_gain: float, distance: float, tx_height: float,
                            rx_height: float, **kwargs) -> PropagationResult:
        """
        Calculate complete link budget with enhanced analysis
        
        Args:
            tx_power: Transmitter power (dBm)
            tx_gain: Transmitter antenna gain (dBi)
            rx_gain: Receiver antenna gain (dBi)
            distance: Distance between antennas (meters)
            tx_height: Transmitter height (meters)
            rx_height: Receiver height (meters)
            **kwargs: Additional parameters
            
        Returns:
            PropagationResult with comprehensive link budget analysis
        """
        # Calculate path loss
        path_loss = self.calculate_path_loss(distance, tx_height, rx_height, **kwargs)
        
        # Calculate atmospheric loss
        atmospheric_loss = self._calculate_atmospheric_loss(distance)
        
        # Calculate multipath loss
        multipath_loss = self._calculate_multipath_loss(distance, tx_height, rx_height)
        
        # Calculate received power
        received_power = tx_power + tx_gain + rx_gain - path_loss
        
        # Calculate signal strength (same as received power for this model)
        signal_strength = received_power
        
        # Calculate SNR
        noise_power = self._calculate_thermal_noise()
        snr = received_power - noise_power
        
        # Calculate fading margin
        fading_margin = self._calculate_fading_margin()
        
        # Calculate link budget margin
        link_budget = received_power - noise_power - fading_margin
        
        # Calculate Fresnel zone clearance
        fresnel_clearance = self.calculate_fresnel_zone(distance)
        
        # Calculate availability
        availability = self._calculate_availability(snr, fading_margin)
        
        return PropagationResult(
            path_loss=path_loss,
            received_power=received_power,
            signal_strength=signal_strength,
            snr=snr,
            link_budget=link_budget,
            atmospheric_loss=atmospheric_loss,
            multipath_loss=multipath_loss,
            fading_margin=fading_margin,
            fresnel_clearance=fresnel_clearance,
            availability=availability
        )
    
    def _calculate_thermal_noise(self, bandwidth: float = 20e6) -> float:
        """Calculate thermal noise power"""
        # Thermal noise = k * T * B
        k = 1.380649e-23  # Boltzmann constant (J/K)
        T = self.atmospheric.temperature  # Temperature in Kelvin
        B = bandwidth  # Bandwidth in Hz
        
        noise_power_watts = k * T * B
        noise_power_dbm = 10 * math.log10(noise_power_watts * 1000)
        
        return noise_power_dbm
    
    def _calculate_fading_margin(self) -> float:
        """Calculate fading margin based on fading model"""
        # Rician fading margin
        rician_margin = 10 * math.log10(1 + self.fading.rician_k / 10)
        
        # Rayleigh fading margin
        rayleigh_margin = -10 * math.log10(-math.log(1 - self.fading.rayleigh_probability))
        
        # Shadowing margin
        shadowing_margin = self.fading.shadowing_std * 2.0  # 95% confidence
        
        # Total fading margin
        total_margin = rician_margin + rayleigh_margin + shadowing_margin
        
        return total_margin
    
    def _calculate_availability(self, snr: float, fading_margin: float) -> float:
        """Calculate link availability based on SNR and fading"""
        # Simplified availability calculation
        # In practice, this would use more sophisticated models
        
        if snr < 10:  # Poor SNR
            availability = 0.5
        elif snr < 15:  # Fair SNR
            availability = 0.8
        elif snr < 20:  # Good SNR
            availability = 0.95
        else:  # Excellent SNR
            availability = 0.99
        
        # Reduce availability based on fading margin
        if fading_margin > 10:
            availability *= 0.9
        elif fading_margin > 5:
            availability *= 0.95
        
        return availability
    
    def calculate_coverage_area(self, tx_power: float, tx_gain: float,
                              rx_gain: float, tx_height: float, rx_height: float,
                              min_snr: float = 10.0, **kwargs) -> float:
        """
        Calculate coverage area for a given transmitter
        
        Args:
            tx_power: Transmitter power (dBm)
            tx_gain: Transmitter antenna gain (dBi)
            rx_gain: Receiver antenna gain (dBi)
            tx_height: Transmitter height (meters)
            rx_height: Receiver height (meters)
            min_snr: Minimum required SNR (dB)
            **kwargs: Additional parameters
            
        Returns:
            Coverage area in square meters
        """
        # Find maximum distance for minimum SNR
        max_distance = self._find_distance_for_snr(tx_power, tx_gain, rx_gain, 
                                                 tx_height, rx_height, min_snr, **kwargs)
        
        # Calculate coverage area (circular)
        coverage_area = math.pi * max_distance * max_distance
        
        return coverage_area
    
    def _find_distance_for_snr(self, tx_power: float, tx_gain: float, rx_gain: float,
                              tx_height: float, rx_height: float, min_snr: float,
                              **kwargs) -> float:
        """Find maximum distance for given SNR requirement"""
        # Binary search for maximum distance
        min_distance = 100  # meters
        max_distance = 1000000  # meters (1000 km)
        
        while max_distance - min_distance > 100:  # 100m precision
            test_distance = (min_distance + max_distance) / 2
            
            # Calculate SNR at test distance
            link_budget = self.calculate_link_budget(tx_power, tx_gain, rx_gain,
                                                   test_distance, tx_height, rx_height, **kwargs)
            
            if link_budget.snr >= min_snr:
                min_distance = test_distance
            else:
                max_distance = test_distance
        
        return min_distance
    
    def calculate_fresnel_zone(self, distance: float, frequency: Optional[float] = None) -> float:
        """
        Calculate Fresnel zone radius
        
        Args:
            distance: Distance between antennas (meters)
            frequency: Frequency (Hz), uses default if None
            
        Returns:
            Fresnel zone radius (meters)
        """
        if frequency is None:
            frequency = self.frequency
        
        wavelength = 3e8 / frequency
        
        # First Fresnel zone radius
        fresnel_radius = math.sqrt(wavelength * distance / 4)
        
        return fresnel_radius
    
    def calculate_multipath_effects(self, distance: float, tx_height: float,
                                  rx_height: float, **kwargs) -> Dict[str, float]:
        """
        Calculate comprehensive multipath effects
        
        Args:
            distance: Distance between antennas (meters)
            tx_height: Transmitter height (meters)
            rx_height: Receiver height (meters)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of multipath effects
        """
        # Ground reflection
        ground_reflection_loss = self._calculate_ground_reflection_loss(distance, tx_height, rx_height)
        
        # Diffraction loss
        diffraction_loss = self._calculate_diffraction_loss(distance, tx_height, rx_height, **kwargs)
        
        # Scattering loss
        scattering_loss = self._calculate_scattering_loss(distance, **kwargs)
        
        # Doppler effects
        doppler_shift = self._calculate_doppler_shift(distance, **kwargs)
        
        return {
            'ground_reflection_loss': ground_reflection_loss,
            'diffraction_loss': diffraction_loss,
            'scattering_loss': scattering_loss,
            'doppler_shift': doppler_shift,
            'total_multipath_loss': ground_reflection_loss + diffraction_loss + scattering_loss
        }
    
    def _calculate_ground_reflection_loss(self, distance: float, tx_height: float, 
                                        rx_height: float) -> float:
        """Calculate ground reflection loss"""
        # Simplified ground reflection model
        reflection_coefficient = 0.3  # Typical for ground reflection
        
        # Path difference
        d_direct = math.sqrt(distance**2 + (tx_height - rx_height)**2)
        d_reflected = math.sqrt(distance**2 + (tx_height + rx_height)**2)
        path_difference = d_reflected - d_direct
        
        # Phase difference
        phase_difference = 2 * math.pi * path_difference / self.wavelength
        
        # Reflection loss
        reflection_loss = 20 * math.log10(abs(1 + reflection_coefficient * 
                                            math.cos(phase_difference)))
        
        return abs(reflection_loss)
    
    def _calculate_diffraction_loss(self, distance: float, tx_height: float, 
                                  rx_height: float, **kwargs) -> float:
        """Calculate diffraction loss"""
        # Simplified knife-edge diffraction model
        obstacle_height = kwargs.get('obstacle_height', 0)
        
        if obstacle_height <= 0:
            return 0.0
        
        # Fresnel parameter
        lambda_wavelength = self.wavelength
        d1 = distance / 2  # Distance to obstacle
        d2 = distance / 2  # Distance from obstacle
        
        v = obstacle_height * math.sqrt(2 * (d1 + d2) / (lambda_wavelength * d1 * d2))
        
        # Diffraction loss (simplified)
        if v < -0.8:
            diffraction_loss = 0.0
        elif v < 0:
            diffraction_loss = 6 + 9 * v
        elif v < 1:
            diffraction_loss = 6 + 9 * v + 20 * math.log10(v)
        else:
            diffraction_loss = 6 + 9 * v + 20 * math.log10(v + math.sqrt(v**2 + 1))
        
        return diffraction_loss
    
    def _calculate_scattering_loss(self, distance: float, **kwargs) -> float:
        """Calculate scattering loss"""
        # Simplified scattering model
        frequency_ghz = self.frequency / 1e9
        
        # Atmospheric scattering
        atmospheric_scattering = 0.01 * frequency_ghz * (distance / 1000)
        
        # Terrain scattering
        terrain_roughness = kwargs.get('terrain_roughness', 50)
        terrain_scattering = 0.001 * terrain_roughness * (distance / 1000)
        
        return atmospheric_scattering + terrain_scattering
    
    def _calculate_doppler_shift(self, distance: float, **kwargs) -> float:
        """Calculate Doppler shift"""
        # Relative velocity between transmitter and receiver
        relative_velocity = kwargs.get('relative_velocity', 0.0)  # m/s
        
        # Doppler shift
        doppler_shift = relative_velocity * self.frequency / 3e8
        
        return doppler_shift
    
    def get_propagation_statistics(self, distances: List[float], tx_height: float,
                                 rx_height: float, **kwargs) -> Dict[str, Any]:
        """
        Calculate propagation statistics over a range of distances
        
        Args:
            distances: List of distances to analyze
            tx_height: Transmitter height (meters)
            rx_height: Receiver height (meters)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of propagation statistics
        """
        path_losses = []
        snr_values = []
        availability_values = []
        
        for distance in distances:
            # Calculate path loss
            path_loss = self.calculate_path_loss(distance, tx_height, rx_height, **kwargs)
            path_losses.append(path_loss)
            
            # Calculate link budget (assuming standard parameters)
            link_budget = self.calculate_link_budget(20, 0, 0, distance, tx_height, rx_height, **kwargs)
            snr_values.append(link_budget.snr)
            availability_values.append(link_budget.availability)
        
        return {
            'distances': distances,
            'path_losses': path_losses,
            'snr_values': snr_values,
            'availability_values': availability_values,
            'average_path_loss': np.mean(path_losses),
            'average_snr': np.mean(snr_values),
            'average_availability': np.mean(availability_values),
            'max_coverage_distance': distances[np.argmax(np.array(snr_values) >= 10)]
        }
    
    def set_frequency(self, frequency: float):
        """Set operating frequency"""
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        self._clear_cache()
        logging.info(f"Frequency set to {frequency/1e9:.1f} GHz")
    
    def update_position(self, position: Tuple[float, float, float]):
        """Update current position (for mobile scenarios)"""
        # This could be used for dynamic propagation modeling
        pass
    
    def get_current_position(self) -> Optional[Tuple[float, float, float]]:
        """Get current position"""
        return None
    
    def get_current_conditions(self) -> Dict[str, Any]:
        """Get current propagation conditions"""
        return {
            'frequency': self.frequency,
            'wavelength': self.wavelength,
            'model': self.model.value,
            'frequency_band': self.frequency_band.name,
            'atmospheric_condition': self.atmospheric.condition.value,
            'temperature': self.atmospheric.temperature,
            'humidity': self.atmospheric.humidity,
            'rain_rate': self.atmospheric.rain_rate,
            'fading_model': {
                'rician_k': self.fading.rician_k,
                'rayleigh_probability': self.fading.rayleigh_probability,
                'shadowing_std': self.fading.shadowing_std
            }
        } 