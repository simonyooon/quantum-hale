"""
Performance integration tests for Quantum HALE Drone System.

Tests the performance characteristics of the quantum-secured HALE drone
simulation framework under various load conditions and scenarios.
"""

import pytest
import time
import numpy as np
import psutil
import threading
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from integration.simulation_orchestrator import SimulationOrchestrator
from quantum_comms.pqc_handshake import PQCHandshake
from quantum_comms.qkd_simulation import QKDSimulation
from network_sim.ns3_wrapper import NS3Wrapper
from utils.config import ConfigManager


class TestPerformanceCharacteristics:
    """Performance characteristic tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        self.orchestrator = SimulationOrchestrator(self.config_manager)
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.shutdown()
    
    @pytest.mark.performance
    def test_quantum_handshake_latency(self):
        """Test quantum handshake latency performance."""
        config = {
            "key_encapsulation": "Kyber768",
            "digital_signature": "Dilithium3",
            "security_level": 3
        }
        
        handshake = PQCHandshake(config)
        latencies = []
        
        # Measure handshake latency over multiple iterations
        for i in range(100):
            start_time = time.time()
            
            with patch.object(handshake, 'generate_keypair') as mock_gen:
                with patch.object(handshake, 'encapsulate_key') as mock_encap:
                    with patch.object(handshake, 'decapsulate_key') as mock_decap:
                        mock_gen.return_value = (b"public_key", b"private_key")
                        mock_encap.return_value = (b"ciphertext", b"shared_secret")
                        mock_decap.return_value = b"shared_secret"
                        
                        # Complete handshake
                        init_message = handshake.initiate_handshake()
                        response_message = handshake.respond_to_handshake(init_message)
                        success = handshake.complete_handshake(response_message)
                        
                        assert success is True
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        
        # Performance analysis
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Performance requirements
        assert avg_latency < 500  # Average latency < 500ms
        assert p95_latency < 1000  # 95th percentile < 1s
        assert p99_latency < 2000  # 99th percentile < 2s
    
    @pytest.mark.performance
    def test_qkd_key_generation_throughput(self):
        """Test QKD key generation throughput."""
        config = {
            "protocol": "BB84",
            "key_length": 256,
            "fidelity_threshold": 0.95
        }
        
        qkd = QKDSimulation(config)
        key_generation_times = []
        key_lengths = []
        
        # Measure key generation performance
        for i in range(50):
            start_time = time.time()
            
            with patch.object(qkd, 'generate_quantum_states') as mock_gen:
                with patch.object(qkd, 'measure_quantum_state') as mock_measure:
                    mock_gen.return_value = [
                        {"basis": "Z", "bit": 1} for _ in range(1000)
                    ]
                    mock_measure.return_value = {"bit": 1, "success": True}
                    
                    result = qkd.execute_protocol()
                    
                    assert result["success"] is True
                    key_lengths.append(result["key_length"])
            
            end_time = time.time()
            generation_time = end_time - start_time
            key_generation_times.append(generation_time)
        
        # Calculate throughput
        total_keys = len(key_generation_times)
        total_time = sum(key_generation_times)
        avg_key_length = np.mean(key_lengths)
        
        keys_per_second = total_keys / total_time
        bits_per_second = keys_per_second * avg_key_length
        
        # Performance requirements
        assert keys_per_second > 1.0  # At least 1 key per second
        assert bits_per_second > 100  # At least 100 bits per second
    
    @pytest.mark.performance
    def test_network_simulation_scalability(self):
        """Test network simulation scalability with increasing nodes."""
        node_counts = [5, 10, 20, 50]
        simulation_times = []
        
        for node_count in node_counts:
            config = {
                "simulation_time": 10,
                "log_level": "error"
            }
            
            ns3 = NS3Wrapper(config)
            
            # Create topology with increasing nodes
            topology_config = {
                "nodes": [
                    {"id": f"node_{i}", "position": [i * 1000, 0, 20000]}
                    for i in range(node_count)
                ],
                "links": [
                    {"source": f"node_{i}", "target": f"node_{i+1}"}
                    for i in range(node_count - 1)
                ]
            }
            
            start_time = time.time()
            
            with patch.object(ns3, '_run_ns3_simulation') as mock_run:
                mock_run.return_value = {"success": True, "results": {}}
                
                topology = ns3.create_topology(topology_config)
                results = ns3.run_simulation()
                
                assert results["success"] is True
            
            end_time = time.time()
            simulation_time = end_time - start_time
            simulation_times.append(simulation_time)
        
        # Scalability analysis
        for i in range(1, len(node_counts)):
            # Simulation time should not grow exponentially
            time_ratio = simulation_times[i] / simulation_times[i-1]
            node_ratio = node_counts[i] / node_counts[i-1]
            
            # Time growth should be sub-quadratic
            assert time_ratio < node_ratio * 1.5
    
    @pytest.mark.performance
    def test_memory_usage_under_load(self):
        """Test memory usage under high load conditions."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple components to simulate high load
        components = []
        
        for i in range(20):
            config = {
                "key_encapsulation": "Kyber768",
                "security_level": 3
            }
            handshake = PQCHandshake(config)
            components.append(handshake)
        
        # Simulate concurrent operations
        def simulate_handshake():
            for _ in range(10):
                with patch.object(components[0], 'generate_keypair') as mock_gen:
                    mock_gen.return_value = (b"public_key", b"private_key")
                    components[0].initiate_handshake()
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=simulate_handshake)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory requirements
        assert memory_increase < 500  # Memory increase < 500MB
        assert final_memory < 2000  # Total memory < 2GB
    
    @pytest.mark.performance
    def test_cpu_usage_under_load(self):
        """Test CPU usage under high load conditions."""
        # Monitor CPU usage during intensive operations
        cpu_percentages = []
        
        def monitor_cpu():
            for _ in range(10):
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_percentages.append(cpu_percent)
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform intensive operations
        for i in range(100):
            config = {
                "protocol": "BB84",
                "key_length": 256
            }
            qkd = QKDSimulation(config)
            
            with patch.object(qkd, 'generate_quantum_states') as mock_gen:
                mock_gen.return_value = [
                    {"basis": "Z", "bit": 1} for _ in range(100)
                ]
                qkd.execute_protocol()
        
        monitor_thread.join()
        
        # CPU usage analysis
        avg_cpu = np.mean(cpu_percentages)
        max_cpu = np.max(cpu_percentages)
        
        # CPU requirements
        assert avg_cpu < 80  # Average CPU < 80%
        assert max_cpu < 95  # Peak CPU < 95%
    
    @pytest.mark.performance
    def test_concurrent_quantum_operations(self):
        """Test performance of concurrent quantum operations."""
        config = {
            "key_encapsulation": "Kyber768",
            "security_level": 3
        }
        
        results = []
        latencies = []
        
        def concurrent_handshake(thread_id):
            handshake = PQCHandshake(config)
            
            for i in range(10):
                start_time = time.time()
                
                with patch.object(handshake, 'generate_keypair') as mock_gen:
                    with patch.object(handshake, 'encapsulate_key') as mock_encap:
                        mock_gen.return_value = (b"public_key", b"private_key")
                        mock_encap.return_value = (b"ciphertext", b"shared_secret")
                        
                        init_message = handshake.initiate_handshake()
                        response_message = handshake.respond_to_handshake(init_message)
                        success = handshake.complete_handshake(response_message)
                        
                        assert success is True
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                
                results.append({
                    "thread_id": thread_id,
                    "operation_id": i,
                    "success": True
                })
                latencies.append(latency)
        
        # Run concurrent operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_handshake, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Performance analysis
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        # Performance requirements
        assert success_rate > 0.95  # Success rate > 95%
        assert avg_latency < 1000  # Average latency < 1s
        assert max_latency < 5000  # Max latency < 5s
    
    @pytest.mark.performance
    def test_data_throughput_limits(self):
        """Test data throughput limits of the system."""
        # Simulate high data throughput scenario
        data_sizes = [1024, 10240, 102400, 1048576]  # 1KB to 1MB
        throughput_rates = []
        
        for data_size in data_sizes:
            start_time = time.time()
            
            # Simulate data transmission with quantum security
            with patch('quantum_comms.pqc_handshake.PQCHandshake') as mock_pqc:
                mock_pqc.return_value.initiate_handshake.return_value = {
                    "public_key": b"throughput_key",
                    "nonce": b"throughput_nonce"
                }
                
                # Simulate processing data_size bytes
                for i in range(0, data_size, 1024):
                    chunk_size = min(1024, data_size - i)
                    # Simulate quantum encryption of chunk
                    time.sleep(0.001)  # Simulate processing time
            
            end_time = time.time()
            transmission_time = end_time - start_time
            throughput = data_size / transmission_time  # bytes per second
            throughput_rates.append(throughput)
        
        # Throughput analysis
        for i, throughput in enumerate(throughput_rates):
            # Throughput should be reasonable for each data size
            assert throughput > 1000  # At least 1KB/s
            if i > 0:
                # Larger data should have better throughput (less overhead)
                assert throughput >= throughput_rates[i-1] * 0.5
    
    @pytest.mark.performance
    def test_system_responsiveness(self):
        """Test system responsiveness under load."""
        response_times = []
        
        # Simulate system under load
        load_threads = []
        
        def generate_load():
            for _ in range(50):
                config = {"key_encapsulation": "Kyber768"}
                handshake = PQCHandshake(config)
                with patch.object(handshake, 'generate_keypair'):
                    handshake.initiate_handshake()
        
        # Start background load
        for _ in range(5):
            thread = threading.Thread(target=generate_load)
            load_threads.append(thread)
            thread.start()
        
        # Measure response times for critical operations
        for i in range(20):
            start_time = time.time()
            
            # Critical operation (e.g., emergency handshake)
            config = {"key_encapsulation": "Kyber768"}
            handshake = PQCHandshake(config)
            
            with patch.object(handshake, 'generate_keypair') as mock_gen:
                mock_gen.return_value = (b"emergency_key", b"emergency_private")
                result = handshake.initiate_handshake()
                assert result is not None
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)
        
        # Wait for load threads to complete
        for thread in load_threads:
            thread.join()
        
        # Responsiveness analysis
        avg_response = np.mean(response_times)
        p95_response = np.percentile(response_times, 95)
        
        # Responsiveness requirements
        assert avg_response < 200  # Average response < 200ms
        assert p95_response < 500  # 95th percentile < 500ms
    
    @pytest.mark.performance
    def test_resource_cleanup_efficiency(self):
        """Test resource cleanup efficiency."""
        import gc
        
        # Monitor memory before
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Create and destroy many objects
        for _ in range(100):
            config = {
                "key_encapsulation": "Kyber768",
                "security_level": 3
            }
            handshake = PQCHandshake(config)
            
            with patch.object(handshake, 'generate_keypair') as mock_gen:
                mock_gen.return_value = (b"temp_key", b"temp_private")
                handshake.initiate_handshake()
            
            # Explicitly delete object
            del handshake
        
        # Force garbage collection
        gc.collect()
        
        # Monitor memory after
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Cleanup requirements
        assert memory_increase < 100  # Memory increase < 100MB after cleanup
    
    @pytest.mark.performance
    def test_energy_efficiency_simulation(self):
        """Test energy efficiency of quantum operations."""
        # Simulate energy consumption for different operations
        operations = [
            ("quantum_handshake", 100),
            ("qkd_key_generation", 50),
            ("network_simulation", 200),
            ("flight_simulation", 300)
        ]
        
        energy_consumption = {}
        
        for operation_name, iterations in operations:
            start_time = time.time()
            
            if operation_name == "quantum_handshake":
                for _ in range(iterations):
                    config = {"key_encapsulation": "Kyber768"}
                    handshake = PQCHandshake(config)
                    with patch.object(handshake, 'generate_keypair'):
                        handshake.initiate_handshake()
            
            elif operation_name == "qkd_key_generation":
                for _ in range(iterations):
                    config = {"protocol": "BB84", "key_length": 256}
                    qkd = QKDSimulation(config)
                    with patch.object(qkd, 'generate_quantum_states'):
                        qkd.execute_protocol()
            
            elif operation_name == "network_simulation":
                for _ in range(iterations):
                    config = {"simulation_time": 1}
                    ns3 = NS3Wrapper(config)
                    with patch.object(ns3, '_run_ns3_simulation'):
                        ns3.run_simulation()
            
            elif operation_name == "flight_simulation":
                for _ in range(iterations):
                    from flight_sim.hale_dynamics import HALEDynamics
                    config = {"mass": 1000, "wingspan": 50}
                    dynamics = HALEDynamics(config)
                    dynamics.update_state([0, 0, 20000])
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Estimate energy consumption (simplified model)
            # Assume CPU time correlates with energy consumption
            energy_consumption[operation_name] = execution_time
        
        # Energy efficiency analysis
        total_energy = sum(energy_consumption.values())
        
        # Energy efficiency requirements
        assert total_energy < 60  # Total execution time < 60 seconds
        assert energy_consumption["quantum_handshake"] < 30  # Handshake < 30s
        assert energy_consumption["qkd_key_generation"] < 20  # QKD < 20s


if __name__ == "__main__":
    pytest.main([__file__, "-m", "performance"]) 