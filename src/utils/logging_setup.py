"""
Logging Setup for Quantum HALE Drone System
=========================================

This module provides logging configuration and setup utilities for
the Quantum HALE Drone System simulation.

Author: Quantum HALE Development Team
License: MIT
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for logging"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "structured",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    structured_output: bool = False
) -> logging.Logger:
    """
    Setup logging configuration for Quantum HALE Drone System
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (structured, colored, simple)
        log_file: Log file path (optional)
        log_dir: Directory for log files
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output to console
        structured_output: Whether to use structured JSON logging
        
    Returns:
        Configured logger instance
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if structured_output or log_format == "structured":
        formatter = StructuredFormatter()
        console_formatter = StructuredFormatter()
    elif log_format == "colored":
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_path = log_path / log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_path / f"quantum_hale_{timestamp}.log"
    
    # Use rotating file handler for automatic log rotation
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create specific loggers for different components
    loggers = {
        'quantum_hale': logging.getLogger('quantum_hale'),
        'flight_sim': logging.getLogger('quantum_hale.flight_sim'),
        'quantum_comms': logging.getLogger('quantum_hale.quantum_comms'),
        'network_sim': logging.getLogger('quantum_hale.network_sim'),
        'integration': logging.getLogger('quantum_hale.integration'),
        'utils': logging.getLogger('quantum_hale.utils')
    }
    
    # Configure component loggers
    for name, component_logger in loggers.items():
        component_logger.setLevel(getattr(logging, log_level.upper()))
        # Don't add handlers to component loggers - they inherit from root
    
    logging.info(f"Logging configured - Level: {log_level}, Format: {log_format}, File: {file_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific component"""
    return logging.getLogger(f"quantum_hale.{name}")


def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, Any]):
    """Log performance metrics in structured format"""
    extra_fields = {
        'metric_type': 'performance',
        'metrics': metrics
    }
    
    logger.info("Performance metrics recorded", extra={'extra_fields': extra_fields})


def log_security_event(logger: logging.Logger, event_type: str, details: Dict[str, Any]):
    """Log security events in structured format"""
    extra_fields = {
        'event_type': 'security',
        'security_event': event_type,
        'details': details
    }
    
    logger.warning(f"Security event: {event_type}", extra={'extra_fields': extra_fields})


def log_network_event(logger: logging.Logger, event_type: str, details: Dict[str, Any]):
    """Log network events in structured format"""
    extra_fields = {
        'event_type': 'network',
        'network_event': event_type,
        'details': details
    }
    
    logger.info(f"Network event: {event_type}", extra={'extra_fields': extra_fields})


def log_quantum_event(logger: logging.Logger, event_type: str, details: Dict[str, Any]):
    """Log quantum communications events in structured format"""
    extra_fields = {
        'event_type': 'quantum',
        'quantum_event': event_type,
        'details': details
    }
    
    logger.info(f"Quantum event: {event_type}", extra={'extra_fields': extra_fields})


def log_flight_event(logger: logging.Logger, event_type: str, details: Dict[str, Any]):
    """Log flight events in structured format"""
    extra_fields = {
        'event_type': 'flight',
        'flight_event': event_type,
        'details': details
    }
    
    logger.info(f"Flight event: {event_type}", extra={'extra_fields': extra_fields})


class LogManager:
    """Log manager for handling multiple log files and rotation"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.loggers = {}
        
    def create_component_logger(
        self,
        component_name: str,
        log_level: str = "INFO",
        max_file_size: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ) -> logging.Logger:
        """Create a dedicated logger for a specific component"""
        
        if component_name in self.loggers:
            return self.loggers[component_name]
        
        # Create logger
        logger = logging.getLogger(f"quantum_hale.{component_name}")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = StructuredFormatter()
        
        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.log_dir / f"{component_name}_{timestamp}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Store logger
        self.loggers[component_name] = logger
        
        logger.info(f"Component logger created for {component_name}")
        return logger
        
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days"""
        import time
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    print(f"Deleted old log file: {log_file}")
                except Exception as e:
                    print(f"Failed to delete log file {log_file}: {e}")
                    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics about log files"""
        stats = {
            'total_files': 0,
            'total_size': 0,
            'components': {},
            'oldest_file': None,
            'newest_file': None
        }
        
        log_files = list(self.log_dir.glob("*.log"))
        stats['total_files'] = len(log_files)
        
        if log_files:
            oldest_time = float('inf')
            newest_time = 0
            
            for log_file in log_files:
                file_size = log_file.stat().st_size
                file_time = log_file.stat().st_mtime
                
                stats['total_size'] += file_size
                
                if file_time < oldest_time:
                    oldest_time = file_time
                    stats['oldest_file'] = log_file.name
                    
                if file_time > newest_time:
                    newest_time = file_time
                    stats['newest_file'] = log_file.name
                    
                # Group by component
                component = log_file.stem.split('_')[0]
                if component not in stats['components']:
                    stats['components'][component] = {
                        'files': 0,
                        'size': 0
                    }
                stats['components'][component]['files'] += 1
                stats['components'][component]['size'] += file_size
                
        return stats


def configure_logging_from_config(config: Dict[str, Any]) -> logging.Logger:
    """Configure logging from configuration dictionary"""
    
    log_config = config.get('logging', {})
    
    return setup_logging(
        log_level=log_config.get('level', 'INFO'),
        log_format=log_config.get('format', 'structured'),
        log_file=log_config.get('file'),
        log_dir=log_config.get('directory', 'logs'),
        max_file_size=log_config.get('max_file_size', 10 * 1024 * 1024),
        backup_count=log_config.get('backup_count', 5),
        console_output=log_config.get('console_output', True),
        structured_output=log_config.get('structured_output', True)
    ) 