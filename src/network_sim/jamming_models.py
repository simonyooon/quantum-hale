"""
Jamming and Interference Models for HALE Drone Network Simulation

This module provides models for simulating jamming attacks and interference
effects on HALE drone communication networks.
"""

import math
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class JammingType(Enum):
    """Types of jamming attacks"""
    CONTINUOUS = "continuous"
    PULSE = "pulse"
    SWEEP = "sweep"
    REACTIVE = "reactive"
    SELECTIVE = "selective"


@dataclass
class JammingSource:
    """Configuration for a jamming source"""
    jammer_id: str
    position: Tuple[float, float, float]  # x, y, z in meters
    power: float  # dBm
    frequency_range: Tuple[float, float]  # Hz (start, end)
    jamming_type: JammingType
    duty_cycle: float = 1.0  # 0.0 to 1.0
    direction: Optional[Tuple[float, float]] = None  # azimuth, elevation in degrees


@dataclass
class InterferenceResult:
    """Results from interference calculation"""
    interference_power: float  # dBm
    snr_degradation: float  # dB
    link_quality: float  # 0.0 to 1.0
    jamming_effectiveness: float  # 0.0 to 1.0


class JammingModels:
    """
    Jamming and interference modeling for HALE drone communications
    """
    
    def __init__(self):
        self.jammers: Dict[str, JammingSource] = {}
        self.frequency_bands = {
            '2.4ghz': (2.4e9, 2.5e9),
            '5ghz': (5.15e9, 5.85e9),
            'satellite': (1.5e9, 1.6e9)
        }
        
        logging.info("Jamming Models initialized")
    
    def add_jammer(self, jammer: JammingSource):
        """Add a jamming source to the simulation"""
        self.jammers[jammer.jammer_id] = jammer
        logging.info(f"Jammer added: {jammer.jammer_id} at {jammer.position}")
    
    def add_jamming_source(self, jammer: JammingSource):
        """Alias for add_jammer method for compatibility"""
        self.add_jammer(jammer)
    
    def remove_jammer(self, jammer_id: str):
        """Remove a jamming source"""
        if jammer_id in self.jammers:
            del self.jammers[jammer_id]
            logging.info(f"Jammer removed: {jammer_id}")
    
    def calculate_interference(self, tx_position: Tuple[float, float, float],
                             rx_position: Tuple[float, float, float],
                             frequency: float, tx_power: float,
                             time: float = 0.0) -> InterferenceResult:
        """
        Calculate interference from all jamming sources
        
        Args:
            tx_position: Transmitter position (x, y, z)
            rx_position: Receiver position (x, y, z)
            frequency: Operating frequency (Hz)
            tx_power: Transmitter power (dBm)
            time: Simulation time (seconds)
            
        Returns:
            InterferenceResult with interference analysis
        """
        total_interference = 0.0
        jamming_effectiveness = 0.0
        
        for jammer_id, jammer in self.jammers.items():
            # Check if frequency is in jammer's range
            if not (jammer.frequency_range[0] <= frequency <= jammer.frequency_range[1]):
                continue
            
            # Calculate distance from jammer to receiver
            distance = self._calculate_distance(jammer.position, rx_position)
            
            # Calculate jamming power at receiver
            jamming_power = self._calculate_jamming_power(jammer, distance, frequency, time)
            
            # Add to total interference
            total_interference = self._add_powers(total_interference, jamming_power)
            
            # Calculate jamming effectiveness
            effectiveness = self._calculate_jamming_effectiveness(jammer, distance, frequency, time)
            jamming_effectiveness = max(jamming_effectiveness, effectiveness)
        
        # Calculate SNR degradation
        signal_power = tx_power - self._calculate_path_loss(tx_position, rx_position, frequency)
        snr_without_jamming = signal_power - self._thermal_noise()
        snr_with_jamming = signal_power - self._thermal_noise() - total_interference
        snr_degradation = snr_without_jamming - snr_with_jamming
        
        # Calculate link quality (0.0 to 1.0)
        link_quality = self._calculate_link_quality(snr_with_jamming)
        
        return InterferenceResult(
            interference_power=total_interference,
            snr_degradation=snr_degradation,
            link_quality=link_quality,
            jamming_effectiveness=jamming_effectiveness
        )
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    def _calculate_jamming_power(self, jammer: JammingSource, distance: float,
                               frequency: float, time: float) -> float:
        """Calculate jamming power at receiver"""
        # Free space path loss
        wavelength = 3e8 / frequency
        path_loss = 20 * math.log10(4 * math.pi * distance / wavelength)
        
        # Base jamming power
        jamming_power = jammer.power - path_loss
        
        # Apply jamming type effects
        if jammer.jamming_type == JammingType.CONTINUOUS:
            # Full power continuous jamming
            power_factor = 1.0
        elif jammer.jamming_type == JammingType.PULSE:
            # Pulsed jamming with duty cycle
            pulse_frequency = 1000  # Hz
            pulse_duty_cycle = jammer.duty_cycle
            if (time * pulse_frequency) % 1 < pulse_duty_cycle:
                power_factor = 1.0
            else:
                power_factor = 0.0
        elif jammer.jamming_type == JammingType.SWEEP:
            # Frequency sweep jamming
            sweep_rate = 1e6  # Hz/s
            sweep_bandwidth = jammer.frequency_range[1] - jammer.frequency_range[0]
            sweep_position = (time * sweep_rate) % sweep_bandwidth
            if abs(frequency - (jammer.frequency_range[0] + sweep_position)) < 1e6:
                power_factor = 1.0
            else:
                power_factor = 0.1
        elif jammer.jamming_type == JammingType.REACTIVE:
            # Reactive jamming (responds to detected signals)
            # Simplified model - assume 50% effectiveness
            power_factor = 0.5
        else:  # SELECTIVE
            # Selective jamming (targets specific frequencies)
            frequency_tolerance = 1e6  # Hz
            if abs(frequency - (jammer.frequency_range[0] + jammer.frequency_range[1]) / 2) < frequency_tolerance:
                power_factor = 1.0
            else:
                power_factor = 0.0
        
        return jamming_power + 10 * math.log10(power_factor) if power_factor > 0 else -1000
    
    def _calculate_jamming_effectiveness(self, jammer: JammingSource, distance: float,
                                       frequency: float, time: float) -> float:
        """Calculate jamming effectiveness (0.0 to 1.0)"""
        # Base effectiveness based on distance
        max_range = 10000  # meters
        distance_factor = max(0, 1 - distance / max_range)
        
        # Frequency match factor
        center_freq = (jammer.frequency_range[0] + jammer.frequency_range[1]) / 2
        bandwidth = jammer.frequency_range[1] - jammer.frequency_range[0]
        freq_match = max(0, 1 - abs(frequency - center_freq) / bandwidth)
        
        # Jamming type effectiveness
        type_effectiveness = {
            JammingType.CONTINUOUS: 1.0,
            JammingType.PULSE: 0.7,
            JammingType.SWEEP: 0.6,
            JammingType.REACTIVE: 0.8,
            JammingType.SELECTIVE: 0.9
        }
        
        effectiveness = distance_factor * freq_match * type_effectiveness[jammer.jamming_type]
        return min(1.0, effectiveness)
    
    def _calculate_path_loss(self, tx_pos: Tuple[float, float, float],
                           rx_pos: Tuple[float, float, float], frequency: float) -> float:
        """Calculate path loss between transmitter and receiver"""
        distance = self._calculate_distance(tx_pos, rx_pos)
        wavelength = 3e8 / frequency
        return 20 * math.log10(4 * math.pi * distance / wavelength)
    
    def _thermal_noise(self, bandwidth: float = 20e6) -> float:
        """Calculate thermal noise power"""
        k = 1.380649e-23  # Boltzmann constant
        T = 290  # Temperature in Kelvin
        noise_power_watts = k * T * bandwidth
        return 10 * math.log10(noise_power_watts * 1000)  # Convert to dBm
    
    def _add_powers(self, power1: float, power2: float) -> float:
        """Add two powers in dBm"""
        if power1 == -1000:
            return power2
        if power2 == -1000:
            return power1
        
        # Convert to linear scale, add, convert back to dB
        p1_linear = 10**(power1 / 10)
        p2_linear = 10**(power2 / 10)
        total_linear = p1_linear + p2_linear
        return 10 * math.log10(total_linear)
    
    def _calculate_link_quality(self, snr: float) -> float:
        """Calculate link quality based on SNR"""
        # Simplified model: quality = 1 / (1 + exp(-(SNR - 10) / 5))
        # This gives quality of 0.5 at SNR = 10 dB, approaches 1.0 at high SNR
        return 1 / (1 + math.exp(-(snr - 10) / 5))
    
    def create_continuous_jammer(self, jammer_id: str, position: Tuple[float, float, float],
                               power: float, frequency_range: Tuple[float, float]) -> JammingSource:
        """Create a continuous jamming source"""
        return JammingSource(
            jammer_id=jammer_id,
            position=position,
            power=power,
            frequency_range=frequency_range,
            jamming_type=JammingType.CONTINUOUS
        )
    
    def create_pulse_jammer(self, jammer_id: str, position: Tuple[float, float, float],
                          power: float, frequency_range: Tuple[float, float],
                          duty_cycle: float = 0.5) -> JammingSource:
        """Create a pulse jamming source"""
        return JammingSource(
            jammer_id=jammer_id,
            position=position,
            power=power,
            frequency_range=frequency_range,
            jamming_type=JammingType.PULSE,
            duty_cycle=duty_cycle
        )
    
    def create_sweep_jammer(self, jammer_id: str, position: Tuple[float, float, float],
                           power: float, frequency_range: Tuple[float, float]) -> JammingSource:
        """Create a frequency sweep jamming source"""
        return JammingSource(
            jammer_id=jammer_id,
            position=position,
            power=power,
            frequency_range=frequency_range,
            jamming_type=JammingType.SWEEP
        )
    
    def analyze_jamming_scenario(self, network_nodes: Dict[str, Tuple[float, float, float]],
                               operating_frequency: float, tx_power: float) -> Dict[str, Any]:
        """
        Analyze jamming effects on entire network
        
        Args:
            network_nodes: Dictionary of node_id -> position
            operating_frequency: Network operating frequency (Hz)
            tx_power: Typical transmitter power (dBm)
            
        Returns:
            Analysis results
        """
        results = {
            'affected_nodes': [],
            'average_link_quality': 0.0,
            'worst_case_snr': float('inf'),
            'jamming_coverage': 0.0
        }
        
        total_link_quality = 0.0
        link_count = 0
        affected_count = 0
        
        # Analyze each potential link
        node_ids = list(network_nodes.keys())
        for i, node1_id in enumerate(node_ids):
            for j, node2_id in enumerate(node_ids[i+1:], i+1):
                tx_pos = network_nodes[node1_id]
                rx_pos = network_nodes[node2_id]
                
                # Calculate interference
                interference = self.calculate_interference(
                    tx_pos, rx_pos, operating_frequency, tx_power
                )
                
                total_link_quality += interference.link_quality
                link_count += 1
                
                if interference.link_quality < 0.5:
                    affected_count += 1
                    results['affected_nodes'].append((node1_id, node2_id))
                
                results['worst_case_snr'] = min(results['worst_case_snr'], 
                                               interference.snr_degradation)
        
        if link_count > 0:
            results['average_link_quality'] = total_link_quality / link_count
            results['jamming_coverage'] = affected_count / link_count
        
        return results
    
    def get_jamming_statistics(self) -> Dict[str, Any]:
        """Get statistics about active jammers"""
        stats = {
            'total_jammers': len(self.jammers),
            'jamming_types': {},
            'total_power': 0.0,
            'frequency_coverage': set()
        }
        
        for jammer in self.jammers.values():
            # Count jamming types
            jam_type = jammer.jamming_type.value
            stats['jamming_types'][jam_type] = stats['jamming_types'].get(jam_type, 0) + 1
            
            # Sum total power
            stats['total_power'] += jammer.power
            
            # Track frequency coverage
            stats['frequency_coverage'].add(jammer.frequency_range)
        
        return stats
    
    def clear_all_jammers(self):
        """Remove all jamming sources"""
        self.jammers.clear()
        logging.info("All jammers cleared")
    
    def is_jamming_active(self) -> bool:
        """Check if any jamming is currently active"""
        return len(self.jammers) > 0
    
    def step(self, simulation_time: float):
        """Update jamming simulation for current time step"""
        # Update jamming effects based on time
        for jammer_id, jammer in self.jammers.items():
            # For now, just log that jamming is active
            if simulation_time % 10 < 1:  # Log every 10 seconds
                logging.debug(f"Jammer {jammer_id} active at time {simulation_time}")
    
    def get_jamming_status(self) -> Dict[str, Any]:
        """Get current jamming status and statistics"""
        return {
            'num_jammers': len(self.jammers),
            'jammers_active': len(self.jammers) > 0,
            'jammer_ids': list(self.jammers.keys()),
            'total_interference_power': sum(jammer.power for jammer in self.jammers.values()),
            'frequency_coverage': self._get_frequency_coverage(),
            'threat_level': self._calculate_threat_level()
        }
    
    def _get_frequency_coverage(self) -> Dict[str, float]:
        """Get frequency coverage of all jammers"""
        if not self.jammers:
            return {}
        
        min_freq = min(float(jammer.frequency_range[0]) for jammer in self.jammers.values())
        max_freq = max(float(jammer.frequency_range[1]) for jammer in self.jammers.values())
        
        return {
            'min_frequency': min_freq,
            'max_frequency': max_freq,
            'bandwidth': max_freq - min_freq
        }
    
    def _calculate_threat_level(self) -> str:
        """Calculate overall threat level"""
        if not self.jammers:
            return "none"
        
        total_power = sum(jammer.power for jammer in self.jammers.values())
        num_jammers = len(self.jammers)
        
        if total_power > 100 or num_jammers > 3:
            return "high"
        elif total_power > 50 or num_jammers > 1:
            return "medium"
        else:
            return "low" 