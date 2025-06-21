"""
Data Collector for Quantum HALE Drone System
==========================================

This module handles collection and storage of simulation data from
all components, including telemetry, network data, and quantum metrics.

Author: Quantum HALE Development Team
License: MIT
"""

import time
import logging
import json
import csv
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False


class DataType(Enum):
    """Types of data being collected"""
    TELEMETRY = "telemetry"
    NETWORK = "network"
    QUANTUM = "quantum"
    PERFORMANCE = "performance"
    SENSOR = "sensor"
    CONTROL = "control"


@dataclass
class DataPoint:
    """Single data point with metadata"""
    timestamp: float
    data_type: DataType
    source: str
    data: Dict[str, Any]
    tags: Dict[str, str] = None


class DataCollector:
    """
    Data collection and storage system for Quantum HALE Drone simulation
    
    Collects data from all simulation components and stores it in various
    formats for analysis and visualization.
    """
    
    def __init__(self, output_directory: str = "data", 
                 database_url: str = None,
                 enable_influxdb: bool = False):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.data_buffer = []
        self.buffer_size = 1000
        self.flush_interval = 10.0  # seconds
        self.last_flush = time.time()
        
        # Database connections
        self.sqlite_conn = None
        self.influxdb_client = None
        self.write_api = None
        
        # Statistics
        self.collected_points = 0
        self.flushed_points = 0
        self.errors = 0
        
        # Initialize storage systems
        self._initialize_sqlite()
        if enable_influxdb and INFLUXDB_AVAILABLE:
            self._initialize_influxdb(database_url)
            
        logging.info(f"Data Collector initialized with output directory: {output_directory}")
        
    def _initialize_sqlite(self):
        """Initialize SQLite database"""
        try:
            db_path = self.output_directory / "simulation_data.db"
            self.sqlite_conn = sqlite3.connect(str(db_path))
            
            # Create tables
            self._create_sqlite_tables()
            logging.info("SQLite database initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize SQLite: {e}")
            
    def _create_sqlite_tables(self):
        """Create SQLite tables for different data types"""
        cursor = self.sqlite_conn.cursor()
        
        # Telemetry table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                latitude REAL,
                longitude REAL,
                altitude REAL,
                airspeed REAL,
                ground_speed REAL,
                heading REAL,
                roll REAL,
                pitch REAL,
                yaw REAL,
                fuel_remaining REAL,
                flight_phase TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Network data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                node_id TEXT,
                signal_strength REAL,
                packet_loss REAL,
                latency REAL,
                bandwidth REAL,
                jamming_active BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Quantum data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quantum_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                handshake_latency REAL,
                key_generation_time REAL,
                security_level TEXT,
                qkd_fidelity REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                cpu_usage REAL,
                memory_usage REAL,
                simulation_rate REAL,
                real_time_factor REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.sqlite_conn.commit()
        
    def _initialize_influxdb(self, database_url: str):
        """Initialize InfluxDB connection"""
        try:
            if not database_url:
                database_url = "http://localhost:8086"
                
            self.influxdb_client = InfluxDBClient(
                url=database_url,
                token="quantum-hale-token",
                org="quantum-hale",
                bucket="simulation-data"
            )
            
            self.write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
            logging.info("InfluxDB connection initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize InfluxDB: {e}")
            
    def collect_telemetry(self, telemetry_data: Dict[str, Any], source: str = "flight_dynamics"):
        """Collect telemetry data from flight dynamics"""
        data_point = DataPoint(
            timestamp=time.time(),
            data_type=DataType.TELEMETRY,
            source=source,
            data=telemetry_data,
            tags={'component': 'flight_dynamics'}
        )
        
        self._add_data_point(data_point)
        
    def collect_network_data(self, network_data: Dict[str, Any], source: str = "network_sim"):
        """Collect network simulation data"""
        data_point = DataPoint(
            timestamp=time.time(),
            data_type=DataType.NETWORK,
            source=source,
            data=network_data,
            tags={'component': 'network_simulation'}
        )
        
        self._add_data_point(data_point)
        
    def collect_quantum_data(self, quantum_data: Dict[str, Any], source: str = "quantum_comms"):
        """Collect quantum communications data"""
        data_point = DataPoint(
            timestamp=time.time(),
            data_type=DataType.QUANTUM,
            source=source,
            data=quantum_data,
            tags={'component': 'quantum_communications'}
        )
        
        self._add_data_point(data_point)
        
    def collect_performance_data(self, performance_data: Dict[str, Any], source: str = "orchestrator"):
        """Collect performance monitoring data"""
        data_point = DataPoint(
            timestamp=time.time(),
            data_type=DataType.PERFORMANCE,
            source=source,
            data=performance_data,
            tags={'component': 'performance_monitoring'}
        )
        
        self._add_data_point(data_point)
        
    def collect_sensor_data(self, sensor_data: Dict[str, Any], source: str = "sensor_fusion"):
        """Collect sensor fusion data"""
        data_point = DataPoint(
            timestamp=time.time(),
            data_type=DataType.SENSOR,
            source=source,
            data=sensor_data,
            tags={'component': 'sensor_fusion'}
        )
        
        self._add_data_point(data_point)
        
    def collect_control_data(self, control_data: Dict[str, Any], source: str = "autonomy_engine"):
        """Collect control system data"""
        data_point = DataPoint(
            timestamp=time.time(),
            data_type=DataType.CONTROL,
            source=source,
            data=control_data,
            tags={'component': 'autonomy_engine'}
        )
        
        self._add_data_point(data_point)
        
    def _add_data_point(self, data_point: DataPoint):
        """Add data point to buffer"""
        self.data_buffer.append(data_point)
        self.collected_points += 1
        
        # Check if buffer is full or flush interval has passed
        if (len(self.data_buffer) >= self.buffer_size or 
            time.time() - self.last_flush >= self.flush_interval):
            self.flush_buffer()
            
    def flush_buffer(self):
        """Flush data buffer to storage"""
        if not self.data_buffer:
            return
            
        try:
            # Store in SQLite
            self._store_in_sqlite(self.data_buffer)
            
            # Store in InfluxDB if available
            if self.influxdb_client:
                self._store_in_influxdb(self.data_buffer)
                
            # Store as JSON files
            self._store_as_json(self.data_buffer)
            
            self.flushed_points += len(self.data_buffer)
            self.data_buffer.clear()
            self.last_flush = time.time()
            
            logging.debug(f"Flushed {len(self.data_buffer)} data points")
            
        except Exception as e:
            logging.error(f"Failed to flush data buffer: {e}")
            self.errors += 1
            
    def _store_in_sqlite(self, data_points: List[DataPoint]):
        """Store data points in SQLite database"""
        if not self.sqlite_conn:
            return
            
        cursor = self.sqlite_conn.cursor()
        
        for point in data_points:
            try:
                if point.data_type == DataType.TELEMETRY:
                    self._insert_telemetry_sqlite(cursor, point)
                elif point.data_type == DataType.NETWORK:
                    self._insert_network_sqlite(cursor, point)
                elif point.data_type == DataType.QUANTUM:
                    self._insert_quantum_sqlite(cursor, point)
                elif point.data_type == DataType.PERFORMANCE:
                    self._insert_performance_sqlite(cursor, point)
                    
            except Exception as e:
                logging.error(f"Failed to insert data point in SQLite: {e}")
                
        self.sqlite_conn.commit()
        
    def _insert_telemetry_sqlite(self, cursor, point: DataPoint):
        """Insert telemetry data into SQLite"""
        data = point.data
        cursor.execute('''
            INSERT INTO telemetry (
                timestamp, latitude, longitude, altitude, airspeed, 
                ground_speed, heading, roll, pitch, yaw, fuel_remaining, flight_phase
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            point.timestamp,
            data.get('position', {}).get('latitude'),
            data.get('position', {}).get('longitude'),
            data.get('position', {}).get('altitude'),
            data.get('velocity', {}).get('airspeed'),
            data.get('velocity', {}).get('ground_speed'),
            data.get('velocity', {}).get('heading'),
            data.get('attitude', {}).get('roll'),
            data.get('attitude', {}).get('pitch'),
            data.get('attitude', {}).get('yaw'),
            data.get('energy', {}).get('fuel_remaining'),
            data.get('flight_phase')
        ))
        
    def _insert_network_sqlite(self, cursor, point: DataPoint):
        """Insert network data into SQLite"""
        data = point.data
        cursor.execute('''
            INSERT INTO network_data (
                timestamp, node_id, signal_strength, packet_loss, 
                latency, bandwidth, jamming_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            point.timestamp,
            data.get('node_id'),
            data.get('signal_strength'),
            data.get('packet_loss'),
            data.get('latency'),
            data.get('bandwidth'),
            data.get('jamming_active', False)
        ))
        
    def _insert_quantum_sqlite(self, cursor, point: DataPoint):
        """Insert quantum data into SQLite"""
        data = point.data
        cursor.execute('''
            INSERT INTO quantum_data (
                timestamp, handshake_latency, key_generation_time, 
                security_level, qkd_fidelity
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            point.timestamp,
            data.get('handshake_latency'),
            data.get('key_generation_time'),
            data.get('security_level'),
            data.get('qkd_fidelity')
        ))
        
    def _insert_performance_sqlite(self, cursor, point: DataPoint):
        """Insert performance data into SQLite"""
        data = point.data
        cursor.execute('''
            INSERT INTO performance_data (
                timestamp, cpu_usage, memory_usage, simulation_rate, real_time_factor
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            point.timestamp,
            data.get('cpu_usage'),
            data.get('memory_usage'),
            data.get('simulation_rate'),
            data.get('real_time_factor')
        ))
        
    def _store_in_influxdb(self, data_points: List[DataPoint]):
        """Store data points in InfluxDB"""
        if not self.influxdb_client or not self.write_api:
            return
            
        points = []
        
        for point in data_points:
            try:
                # Create InfluxDB point
                influx_point = Point(point.data_type.value)
                
                # Add tags
                if point.tags:
                    for key, value in point.tags.items():
                        influx_point.tag(key, value)
                influx_point.tag("source", point.source)
                
                # Add fields
                for key, value in point.data.items():
                    if isinstance(value, (int, float, bool, str)):
                        influx_point.field(key, value)
                        
                # Set timestamp
                influx_point.time(int(point.timestamp * 1e9))  # Convert to nanoseconds
                
                points.append(influx_point)
                
            except Exception as e:
                logging.error(f"Failed to create InfluxDB point: {e}")
                
        if points:
            try:
                self.write_api.write(bucket="simulation-data", record=points)
            except Exception as e:
                logging.error(f"Failed to write to InfluxDB: {e}")
                
    def _store_as_json(self, data_points: List[DataPoint]):
        """Store data points as JSON files"""
        timestamp = int(time.time())
        
        # Group by data type
        grouped_data = {}
        for point in data_points:
            data_type = point.data_type.value
            if data_type not in grouped_data:
                grouped_data[data_type] = []
            grouped_data[data_type].append(asdict(point))
            
        # Save each data type to separate file
        for data_type, points in grouped_data.items():
            filename = self.output_directory / f"{data_type}_{timestamp}.json"
            try:
                with open(filename, 'w') as f:
                    json.dump(points, f, indent=2, default=str)
            except Exception as e:
                logging.error(f"Failed to save JSON file {filename}: {e}")
                
    def export_to_csv(self, data_type: DataType, filename: str = None):
        """Export data to CSV format"""
        if not filename:
            filename = self.output_directory / f"{data_type.value}_export.csv"
            
        if not self.sqlite_conn:
            logging.error("SQLite connection not available")
            return
            
        try:
            cursor = self.sqlite_conn.cursor()
            
            if data_type == DataType.TELEMETRY:
                cursor.execute('SELECT * FROM telemetry ORDER BY timestamp')
            elif data_type == DataType.NETWORK:
                cursor.execute('SELECT * FROM network_data ORDER BY timestamp')
            elif data_type == DataType.QUANTUM:
                cursor.execute('SELECT * FROM quantum_data ORDER BY timestamp')
            elif data_type == DataType.PERFORMANCE:
                cursor.execute('SELECT * FROM performance_data ORDER BY timestamp')
            else:
                logging.error(f"Unsupported data type for CSV export: {data_type}")
                return
                
            rows = cursor.fetchall()
            
            if rows:
                # Get column names
                columns = [description[0] for description in cursor.description]
                
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(columns)
                    writer.writerows(rows)
                    
                logging.info(f"Exported {len(rows)} rows to {filename}")
            else:
                logging.warning(f"No data found for {data_type}")
                
        except Exception as e:
            logging.error(f"Failed to export CSV: {e}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        return {
            'collected_points': self.collected_points,
            'flushed_points': self.flushed_points,
            'buffer_size': len(self.data_buffer),
            'errors': self.errors,
            'sqlite_available': self.sqlite_conn is not None,
            'influxdb_available': self.influxdb_client is not None,
            'output_directory': str(self.output_directory)
        }
        
    def close(self):
        """Close data collector and flush remaining data"""
        self.flush_buffer()
        
        if self.sqlite_conn:
            self.sqlite_conn.close()
            
        if self.influxdb_client:
            self.influxdb_client.close()
            
        logging.info("Data Collector closed") 