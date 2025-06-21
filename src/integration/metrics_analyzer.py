"""
Metrics Analyzer for Quantum HALE Drone System
============================================

This module analyzes performance and security metrics from simulation
data, providing insights into system behavior and optimization opportunities.

Author: Quantum HALE Development Team
License: MIT
"""

import time
import logging
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class MetricType(Enum):
    """Types of metrics to analyze"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    NETWORK = "network"
    FLIGHT = "flight"
    QUANTUM = "quantum"


@dataclass
class MetricResult:
    """Result of a metric analysis"""
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    description: str
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = None


class MetricsAnalyzer:
    """
    Metrics analysis system for Quantum HALE Drone simulation
    
    Analyzes performance, security, and operational metrics from
    collected simulation data to provide insights and recommendations.
    """
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = Path(data_directory)
        self.sqlite_path = self.data_directory / "simulation_data.db"
        
        # Analysis results
        self.analysis_results = []
        self.recommendations = []
        
        # Database connection
        self.db_conn = None
        
        # Initialize database connection
        self._initialize_database()
        
        logging.info("Metrics Analyzer initialized")
        
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            if self.sqlite_path.exists():
                self.db_conn = sqlite3.connect(str(self.sqlite_path))
                logging.info("Database connection established")
            else:
                logging.warning("Database file not found")
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}")
            
    def analyze_performance_metrics(self) -> List[MetricResult]:
        """Analyze performance metrics from simulation data"""
        results = []
        
        if not self.db_conn:
            return results
            
        try:
            # Query performance data
            query = """
                SELECT cpu_usage, memory_usage, simulation_rate, real_time_factor
                FROM performance_data
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, self.db_conn)
            
            if df.empty:
                logging.warning("No performance data found")
                return results
                
            # Calculate performance metrics
            results.extend(self._calculate_performance_metrics(df))
            
        except Exception as e:
            logging.error(f"Failed to analyze performance metrics: {e}")
            
        return results
        
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> List[MetricResult]:
        """Calculate performance metrics from dataframe"""
        results = []
        
        # Average CPU usage
        avg_cpu = df['cpu_usage'].mean()
        results.append(MetricResult(
            metric_name="Average CPU Usage",
            metric_type=MetricType.PERFORMANCE,
            value=avg_cpu,
            unit="%",
            description="Average CPU utilization during simulation",
            timestamp=time.time()
        ))
        
        # Peak CPU usage
        peak_cpu = df['cpu_usage'].max()
        results.append(MetricResult(
            metric_name="Peak CPU Usage",
            metric_type=MetricType.PERFORMANCE,
            value=peak_cpu,
            unit="%",
            description="Maximum CPU utilization during simulation",
            timestamp=time.time()
        ))
        
        # Average memory usage
        avg_memory = df['memory_usage'].mean()
        results.append(MetricResult(
            metric_name="Average Memory Usage",
            metric_type=MetricType.PERFORMANCE,
            value=avg_memory,
            unit="MB",
            description="Average memory consumption during simulation",
            timestamp=time.time()
        ))
        
        # Average simulation rate
        avg_rate = df['simulation_rate'].mean()
        results.append(MetricResult(
            metric_name="Average Simulation Rate",
            metric_type=MetricType.PERFORMANCE,
            value=avg_rate,
            unit="Hz",
            description="Average simulation update rate",
            timestamp=time.time()
        ))
        
        # Real-time factor
        avg_rtf = df['real_time_factor'].mean()
        results.append(MetricResult(
            metric_name="Average Real-Time Factor",
            metric_type=MetricType.PERFORMANCE,
            value=avg_rtf,
            unit="",
            description="Average real-time factor (1.0 = real-time)",
            timestamp=time.time()
        ))
        
        return results
        
    def analyze_flight_metrics(self) -> List[MetricResult]:
        """Analyze flight performance metrics"""
        results = []
        
        if not self.db_conn:
            return results
            
        try:
            # Query telemetry data
            query = """
                SELECT latitude, longitude, altitude, airspeed, ground_speed,
                       heading, fuel_remaining, flight_phase
                FROM telemetry
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, self.db_conn)
            
            if df.empty:
                logging.warning("No flight data found")
                return results
                
            # Calculate flight metrics
            results.extend(self._calculate_flight_metrics(df))
            
        except Exception as e:
            logging.error(f"Failed to analyze flight metrics: {e}")
            
        return results
        
    def _calculate_flight_metrics(self, df: pd.DataFrame) -> List[MetricResult]:
        """Calculate flight metrics from dataframe"""
        results = []
        
        # Average altitude
        avg_altitude = df['altitude'].mean()
        results.append(MetricResult(
            metric_name="Average Altitude",
            metric_type=MetricType.FLIGHT,
            value=avg_altitude,
            unit="m",
            description="Average flight altitude during simulation",
            timestamp=time.time()
        ))
        
        # Maximum altitude
        max_altitude = df['altitude'].max()
        results.append(MetricResult(
            metric_name="Maximum Altitude",
            metric_type=MetricType.FLIGHT,
            value=max_altitude,
            unit="m",
            description="Maximum altitude reached during simulation",
            timestamp=time.time()
        ))
        
        # Average airspeed
        avg_airspeed = df['airspeed'].mean()
        results.append(MetricResult(
            metric_name="Average Airspeed",
            metric_type=MetricType.FLIGHT,
            value=avg_airspeed,
            unit="m/s",
            description="Average airspeed during simulation",
            timestamp=time.time()
        ))
        
        # Fuel efficiency
        if len(df) > 1:
            initial_fuel = df['fuel_remaining'].iloc[0]
            final_fuel = df['fuel_remaining'].iloc[-1]
            fuel_consumed = initial_fuel - final_fuel
            flight_time = len(df) * 0.01  # Assuming 0.01s timestep
            
            if fuel_consumed > 0:
                fuel_efficiency = flight_time / fuel_consumed
                results.append(MetricResult(
                    metric_name="Fuel Efficiency",
                    metric_type=MetricType.FLIGHT,
                    value=fuel_efficiency,
                    unit="s/kg",
                    description="Flight time per unit fuel consumed",
                    timestamp=time.time()
                ))
        
        # Flight distance (simplified calculation)
        if len(df) > 1:
            # Calculate total distance using Haversine formula
            total_distance = 0.0
            for i in range(1, len(df)):
                lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
                lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
                distance = self._haversine_distance(lat1, lon1, lat2, lon2)
                total_distance += distance
                
            results.append(MetricResult(
                metric_name="Total Flight Distance",
                metric_type=MetricType.FLIGHT,
                value=total_distance,
                unit="m",
                description="Total distance traveled during simulation",
                timestamp=time.time()
            ))
        
        return results
        
    def analyze_network_metrics(self) -> List[MetricResult]:
        """Analyze network performance metrics"""
        results = []
        
        if not self.db_conn:
            return results
            
        try:
            # Query network data
            query = """
                SELECT signal_strength, packet_loss, latency, bandwidth, jamming_active
                FROM network_data
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, self.db_conn)
            
            if df.empty:
                logging.warning("No network data found")
                return results
                
            # Calculate network metrics
            results.extend(self._calculate_network_metrics(df))
            
        except Exception as e:
            logging.error(f"Failed to analyze network metrics: {e}")
            
        return results
        
    def _calculate_network_metrics(self, df: pd.DataFrame) -> List[MetricResult]:
        """Calculate network metrics from dataframe"""
        results = []
        
        # Average signal strength
        avg_signal = df['signal_strength'].mean()
        results.append(MetricResult(
            metric_name="Average Signal Strength",
            metric_type=MetricType.NETWORK,
            value=avg_signal,
            unit="dBm",
            description="Average signal strength during simulation",
            timestamp=time.time()
        ))
        
        # Average packet loss
        avg_packet_loss = df['packet_loss'].mean()
        results.append(MetricResult(
            metric_name="Average Packet Loss",
            metric_type=MetricType.NETWORK,
            value=avg_packet_loss,
            unit="%",
            description="Average packet loss rate during simulation",
            timestamp=time.time()
        ))
        
        # Average latency
        avg_latency = df['latency'].mean()
        results.append(MetricResult(
            metric_name="Average Latency",
            metric_type=MetricType.NETWORK,
            value=avg_latency,
            unit="ms",
            description="Average network latency during simulation",
            timestamp=time.time()
        ))
        
        # Average bandwidth
        avg_bandwidth = df['bandwidth'].mean()
        results.append(MetricResult(
            metric_name="Average Bandwidth",
            metric_type=MetricType.NETWORK,
            value=avg_bandwidth,
            unit="Mbps",
            description="Average network bandwidth during simulation",
            timestamp=time.time()
        ))
        
        # Jamming incidents
        jamming_incidents = df['jamming_active'].sum()
        results.append(MetricResult(
            metric_name="Jamming Incidents",
            metric_type=MetricType.NETWORK,
            value=jamming_incidents,
            unit="count",
            description="Number of jamming incidents detected",
            timestamp=time.time()
        ))
        
        return results
        
    def analyze_quantum_metrics(self) -> List[MetricResult]:
        """Analyze quantum communications metrics"""
        results = []
        
        if not self.db_conn:
            return results
            
        try:
            # Query quantum data
            query = """
                SELECT handshake_latency, key_generation_time, security_level, qkd_fidelity
                FROM quantum_data
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, self.db_conn)
            
            if df.empty:
                logging.warning("No quantum data found")
                return results
                
            # Calculate quantum metrics
            results.extend(self._calculate_quantum_metrics(df))
            
        except Exception as e:
            logging.error(f"Failed to analyze quantum metrics: {e}")
            
        return results
        
    def _calculate_quantum_metrics(self, df: pd.DataFrame) -> List[MetricResult]:
        """Calculate quantum metrics from dataframe"""
        results = []
        
        # Average handshake latency
        avg_handshake_latency = df['handshake_latency'].mean()
        results.append(MetricResult(
            metric_name="Average Handshake Latency",
            metric_type=MetricType.QUANTUM,
            value=avg_handshake_latency,
            unit="ms",
            description="Average PQC handshake latency",
            timestamp=time.time()
        ))
        
        # Average key generation time
        avg_key_gen_time = df['key_generation_time'].mean()
        results.append(MetricResult(
            metric_name="Average Key Generation Time",
            metric_type=MetricType.QUANTUM,
            value=avg_key_gen_time,
            unit="ms",
            description="Average quantum key generation time",
            timestamp=time.time()
        ))
        
        # Average QKD fidelity
        avg_fidelity = df['qkd_fidelity'].mean()
        results.append(MetricResult(
            metric_name="Average QKD Fidelity",
            metric_type=MetricType.QUANTUM,
            value=avg_fidelity,
            unit="",
            description="Average quantum key distribution fidelity",
            timestamp=time.time()
        ))
        
        # Security level distribution
        security_levels = df['security_level'].value_counts()
        for level, count in security_levels.items():
            results.append(MetricResult(
                metric_name=f"Security Level {level} Usage",
                metric_type=MetricType.QUANTUM,
                value=count,
                unit="count",
                description=f"Number of handshakes using security level {level}",
                timestamp=time.time()
            ))
        
        return results
        
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Performance recommendations
        performance_results = [r for r in self.analysis_results if r.metric_type == MetricType.PERFORMANCE]
        
        for result in performance_results:
            if result.metric_name == "Average CPU Usage" and result.value > 80:
                recommendations.append("High CPU usage detected. Consider optimizing simulation algorithms or using more powerful hardware.")
            elif result.metric_name == "Average Real-Time Factor" and result.value < 0.8:
                recommendations.append("Simulation running slower than real-time. Consider reducing simulation complexity or improving performance.")
            elif result.metric_name == "Peak Memory Usage" and result.value > 1000:
                recommendations.append("High memory usage detected. Consider implementing memory optimization or increasing available RAM.")
        
        # Network recommendations
        network_results = [r for r in self.analysis_results if r.metric_type == MetricType.NETWORK]
        
        for result in network_results:
            if result.metric_name == "Average Packet Loss" and result.value > 5:
                recommendations.append("High packet loss detected. Consider improving network topology or reducing interference.")
            elif result.metric_name == "Average Latency" and result.value > 100:
                recommendations.append("High network latency detected. Consider optimizing routing or reducing network load.")
            elif result.metric_name == "Jamming Incidents" and result.value > 0:
                recommendations.append("Jamming incidents detected. Consider implementing anti-jamming techniques or frequency hopping.")
        
        # Quantum recommendations
        quantum_results = [r for r in self.analysis_results if r.metric_type == MetricType.QUANTUM]
        
        for result in quantum_results:
            if result.metric_name == "Average Handshake Latency" and result.value > 1000:
                recommendations.append("High quantum handshake latency detected. Consider optimizing PQC algorithms or using faster hardware.")
            elif result.metric_name == "Average QKD Fidelity" and result.value < 0.9:
                recommendations.append("Low QKD fidelity detected. Consider improving quantum channel quality or error correction.")
        
        # Flight recommendations
        flight_results = [r for r in self.analysis_results if r.metric_type == MetricType.FLIGHT]
        
        for result in flight_results:
            if result.metric_name == "Fuel Efficiency" and result.value < 100:
                recommendations.append("Low fuel efficiency detected. Consider optimizing flight path or improving propulsion efficiency.")
        
        return recommendations
        
    def create_analysis_report(self, output_file: str = None) -> str:
        """Create a comprehensive analysis report"""
        if not output_file:
            output_file = self.data_directory / f"analysis_report_{int(time.time())}.json"
            
        # Collect all metrics
        self.analysis_results = []
        self.analysis_results.extend(self.analyze_performance_metrics())
        self.analysis_results.extend(self.analyze_flight_metrics())
        self.analysis_results.extend(self.analyze_network_metrics())
        self.analysis_results.extend(self.analyze_quantum_metrics())
        
        # Generate recommendations
        self.recommendations = self.generate_recommendations()
        
        # Create report
        report = {
            'timestamp': time.time(),
            'metrics': [asdict(result) for result in self.analysis_results],
            'recommendations': self.recommendations,
            'summary': self._create_summary()
        }
        
        # Save report
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logging.info(f"Analysis report saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save analysis report: {e}")
            
        return str(output_file)
        
    def _create_summary(self) -> Dict[str, Any]:
        """Create summary of analysis results"""
        summary = {
            'total_metrics': len(self.analysis_results),
            'performance_metrics': len([r for r in self.analysis_results if r.metric_type == MetricType.PERFORMANCE]),
            'flight_metrics': len([r for r in self.analysis_results if r.metric_type == MetricType.FLIGHT]),
            'network_metrics': len([r for r in self.analysis_results if r.metric_type == MetricType.NETWORK]),
            'quantum_metrics': len([r for r in self.analysis_results if r.metric_type == MetricType.QUANTUM]),
            'recommendations_count': len(self.recommendations)
        }
        
        return summary
        
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
        
    def create_visualizations(self, output_directory: str = None):
        """Create visualization plots of the analysis results"""
        if not PLOTTING_AVAILABLE:
            logging.warning("Matplotlib/Seaborn not available for visualizations")
            return
            
        if not output_directory:
            output_directory = self.data_directory / "visualizations"
            
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.db_conn:
            logging.warning("No database connection for visualizations")
            return
            
        try:
            # Performance over time
            self._plot_performance_over_time(output_dir)
            
            # Flight trajectory
            self._plot_flight_trajectory(output_dir)
            
            # Network performance
            self._plot_network_performance(output_dir)
            
            # Quantum metrics
            self._plot_quantum_metrics(output_dir)
            
            logging.info(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            logging.error(f"Failed to create visualizations: {e}")
            
    def _plot_performance_over_time(self, output_dir: Path):
        """Plot performance metrics over time"""
        df = pd.read_sql_query("SELECT * FROM performance_data ORDER BY timestamp", self.db_conn)
        
        if df.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # CPU usage
        axes[0, 0].plot(df['timestamp'], df['cpu_usage'])
        axes[0, 0].set_title('CPU Usage Over Time')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        
        # Memory usage
        axes[0, 1].plot(df['timestamp'], df['memory_usage'])
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        
        # Simulation rate
        axes[1, 0].plot(df['timestamp'], df['simulation_rate'])
        axes[1, 0].set_title('Simulation Rate Over Time')
        axes[1, 0].set_ylabel('Rate (Hz)')
        
        # Real-time factor
        axes[1, 1].plot(df['timestamp'], df['real_time_factor'])
        axes[1, 1].set_title('Real-Time Factor Over Time')
        axes[1, 1].set_ylabel('RTF')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_flight_trajectory(self, output_dir: Path):
        """Plot flight trajectory"""
        df = pd.read_sql_query("SELECT latitude, longitude, altitude FROM telemetry ORDER BY timestamp", self.db_conn)
        
        if df.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 2D trajectory
        ax1.plot(df['longitude'], df['latitude'])
        ax1.set_title('Flight Trajectory (2D)')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True)
        
        # Altitude over time
        ax2.plot(range(len(df)), df['altitude'])
        ax2.set_title('Altitude Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Altitude (m)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'flight_trajectory.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_network_performance(self, output_dir: Path):
        """Plot network performance metrics"""
        df = pd.read_sql_query("SELECT * FROM network_data ORDER BY timestamp", self.db_conn)
        
        if df.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Signal strength
        axes[0, 0].plot(df['timestamp'], df['signal_strength'])
        axes[0, 0].set_title('Signal Strength Over Time')
        axes[0, 0].set_ylabel('Signal Strength (dBm)')
        
        # Packet loss
        axes[0, 1].plot(df['timestamp'], df['packet_loss'])
        axes[0, 1].set_title('Packet Loss Over Time')
        axes[0, 1].set_ylabel('Packet Loss (%)')
        
        # Latency
        axes[1, 0].plot(df['timestamp'], df['latency'])
        axes[1, 0].set_title('Latency Over Time')
        axes[1, 0].set_ylabel('Latency (ms)')
        
        # Bandwidth
        axes[1, 1].plot(df['timestamp'], df['bandwidth'])
        axes[1, 1].set_title('Bandwidth Over Time')
        axes[1, 1].set_ylabel('Bandwidth (Mbps)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'network_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_quantum_metrics(self, output_dir: Path):
        """Plot quantum communications metrics"""
        df = pd.read_sql_query("SELECT * FROM quantum_data ORDER BY timestamp", self.db_conn)
        
        if df.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Handshake latency
        axes[0, 0].plot(df['timestamp'], df['handshake_latency'])
        axes[0, 0].set_title('Handshake Latency Over Time')
        axes[0, 0].set_ylabel('Latency (ms)')
        
        # Key generation time
        axes[0, 1].plot(df['timestamp'], df['key_generation_time'])
        axes[0, 1].set_title('Key Generation Time Over Time')
        axes[0, 1].set_ylabel('Time (ms)')
        
        # QKD fidelity
        axes[1, 0].plot(df['timestamp'], df['qkd_fidelity'])
        axes[1, 0].set_title('QKD Fidelity Over Time')
        axes[1, 0].set_ylabel('Fidelity')
        
        # Security level distribution
        security_counts = df['security_level'].value_counts()
        axes[1, 1].pie(security_counts.values, labels=security_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Security Level Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quantum_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def close(self):
        """Close the metrics analyzer"""
        if self.db_conn:
            self.db_conn.close()
        logging.info("Metrics Analyzer closed") 