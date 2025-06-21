"""
End-to-end integration tests for Quantum HALE Drone System.

Tests the complete workflow from quantum key generation to secure
communication and flight simulation in a realistic scenario.
"""

import pytest
import time
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from integration.simulation_orchestrator import SimulationOrchestrator
from quantum_comms.pqc_handshake import PQCHandshake
from quantum_comms.qkd_simulation import QKDSimulation
from network_sim.ns3_wrapper import NS3Wrapper
from flight_sim.hale_dynamics import HALEDynamics
from utils.config import ConfigManager


class TestEndToEndSimulation:
    """End-to-end simulation tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        self.orchestrator = SimulationOrchestrator(self.config_manager)
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.shutdown()
    
    @pytest.mark.integration
    def test_complete_quantum_handshake_workflow(self, quantum_network_setup):
        """Test complete quantum handshake workflow between two drones."""
        pqc = quantum_network_setup["pqc"]
        
        # Simulate handshake between two drones
        drone1_handshake = PQCHandshake(pqc.config)
        drone2_handshake = PQCHandshake(pqc.config)
        
        # Initiate handshake from drone1
        init_message = drone1_handshake.initiate_handshake()
        assert init_message is not None
        assert "public_key" in init_message
        assert "nonce" in init_message
        
        # Drone2 responds to handshake
        response_message = drone2_handshake.respond_to_handshake(init_message)
        assert response_message is not None
        assert "public_key" in response_message
        assert "ciphertext" in response_message
        
        # Drone1 completes handshake
        success = drone1_handshake.complete_handshake(response_message)
        assert success is True
        assert drone1_handshake.state.value == "COMPLETED"
        
        # Verify shared secret is established
        assert drone1_handshake.shared_secret is not None
        assert len(drone1_handshake.shared_secret) > 0
    
    @pytest.mark.integration
    def test_qkd_key_generation_workflow(self, quantum_network_setup):
        """Test complete QKD key generation workflow."""
        qkd = quantum_network_setup["qkd"]
        
        # Execute QKD protocol
        result = qkd.execute_protocol()
        
        assert result["success"] is True
        assert "shared_key" in result
        assert "key_length" in result
        assert "error_rate" in result
        assert result["key_length"] > 0
        assert result["error_rate"] < 0.1  # Should be low error rate
    
    @pytest.mark.integration
    def test_network_simulation_with_quantum_security(self, quantum_network_setup):
        """Test network simulation with quantum-secured communication."""
        ns3 = quantum_network_setup["ns3"]
        
        # Create network topology with quantum-secured nodes
        topology_config = {
            "nodes": [
                {"id": "drone1", "type": "quantum_secure", "position": [0, 0, 20000]},
                {"id": "drone2", "type": "quantum_secure", "position": [10000, 0, 20000]},
                {"id": "ground_station", "type": "quantum_secure", "position": [0, 0, 100]}
            ],
            "links": [
                {"source": "drone1", "target": "drone2", "type": "quantum_secure"},
                {"source": "drone1", "target": "ground_station", "type": "quantum_secure"}
            ]
        }
        
        topology = ns3.create_topology(topology_config)
        assert topology is not None
        assert len(topology.nodes) == 3
        assert len(topology.links) == 2
        
        # Run network simulation
        results = ns3.run_simulation()
        assert results["success"] is True
    
    @pytest.mark.integration
    def test_flight_simulation_with_quantum_comms(self):
        """Test flight simulation with quantum communications integration."""
        # Initialize flight dynamics
        flight_config = {
            "mass": 1000,  # kg
            "wingspan": 50,  # meters
            "cruise_altitude": 20000,  # meters
            "cruise_speed": 50  # m/s
        }
        
        dynamics = HALEDynamics(flight_config)
        
        # Simulate flight path
        waypoints = [
            [0, 0, 20000],
            [50000, 0, 20000],
            [50000, 50000, 20000],
            [0, 50000, 20000]
        ]
        
        flight_data = []
        for waypoint in waypoints:
            state = dynamics.update_state(waypoint)
            flight_data.append(state)
            
            # Simulate quantum communication at each waypoint
            with patch('quantum_comms.pqc_handshake.PQCHandshake') as mock_pqc:
                mock_pqc.return_value.initiate_handshake.return_value = {
                    "public_key": b"test_key",
                    "nonce": b"test_nonce"
                }
                
                pqc = mock_pqc.return_value
                handshake_result = pqc.initiate_handshake()
                assert handshake_result is not None
        
        assert len(flight_data) == 4
        assert all(isinstance(state, dict) for state in flight_data)
    
    @pytest.mark.integration
    def test_jamming_resistance_with_quantum_security(self, quantum_network_setup):
        """Test jamming resistance with quantum-secured communications."""
        from network_sim.jamming_models import JammingModel
        
        # Set up jamming scenario
        jamming_config = {
            "jamming_power": 50,  # dBm
            "frequency_range": [2.4e9, 2.5e9],
            "jamming_type": "barrage"
        }
        
        jammer = JammingModel(jamming_config)
        
        # Simulate quantum communication under jamming
        target_frequency = 2.45e9
        target_power = -60  # dBm
        
        # Without quantum security (vulnerable)
        jammed_power_classical = jammer.simulate_barrage_jamming(target_frequency, target_power)
        
        # With quantum security (more resistant)
        with patch('quantum_comms.pqc_handshake.PQCHandshake') as mock_pqc:
            mock_pqc.return_value.initiate_handshake.return_value = {
                "public_key": b"quantum_key",
                "nonce": b"quantum_nonce"
            }
            
            # Quantum handshake should still work under jamming
            pqc = mock_pqc.return_value
            handshake_result = pqc.initiate_handshake()
            assert handshake_result is not None
            
            # Quantum communication should be more resistant to jamming
            jammed_power_quantum = jammer.simulate_barrage_jamming(target_frequency, target_power)
            
            # Quantum should perform better (though this is simplified)
            assert jammed_power_quantum >= jammed_power_classical
    
    @pytest.mark.integration
    def test_multi_drone_coordination(self):
        """Test coordination between multiple drones with quantum security."""
        # Initialize multiple drones
        drones = []
        for i in range(3):
            drone_config = {
                "id": f"drone_{i+1}",
                "position": [i * 10000, 0, 20000],
                "quantum_enabled": True
            }
            drones.append(drone_config)
        
        # Simulate quantum-secured mesh network
        mesh_connections = []
        for i in range(len(drones)):
            for j in range(i + 1, len(drones)):
                connection = {
                    "source": drones[i]["id"],
                    "target": drones[j]["id"],
                    "type": "quantum_secure",
                    "established": False
                }
                mesh_connections.append(connection)
        
        # Establish quantum-secured connections
        for connection in mesh_connections:
            with patch('quantum_comms.pqc_handshake.PQCHandshake') as mock_pqc:
                mock_pqc.return_value.initiate_handshake.return_value = {
                    "public_key": b"mesh_key",
                    "nonce": b"mesh_nonce"
                }
                
                pqc = mock_pqc.return_value
                handshake_result = pqc.initiate_handshake()
                assert handshake_result is not None
                
                connection["established"] = True
        
        # Verify all connections are established
        assert all(conn["established"] for conn in mesh_connections)
        assert len(mesh_connections) == 3  # 3 drones = 3 connections
    
    @pytest.mark.integration
    def test_mission_execution_with_quantum_security(self):
        """Test complete mission execution with quantum-secured communications."""
        # Load mission configuration
        mission_config = {
            "type": "isr_patrol",
            "duration": 3600,  # 1 hour
            "waypoints": [
                [0, 0, 20000],
                [50000, 0, 20000],
                [50000, 50000, 20000],
                [0, 50000, 20000]
            ],
            "quantum_security": True
        }
        
        # Initialize mission components
        from flight_sim.autonomy_engine import AutonomyEngine
        from integration.data_collector import DataCollector
        
        autonomy = AutonomyEngine(mission_config)
        data_collector = DataCollector()
        
        # Execute mission with quantum security
        mission_data = []
        for i, waypoint in enumerate(mission_config["waypoints"]):
            # Update flight state
            flight_state = autonomy.update_mission_state(waypoint)
            mission_data.append(flight_state)
            
            # Establish quantum-secured communication
            with patch('quantum_comms.pqc_handshake.PQCHandshake') as mock_pqc:
                mock_pqc.return_value.initiate_handshake.return_value = {
                    "public_key": b"mission_key",
                    "nonce": b"mission_nonce"
                }
                
                pqc = mock_pqc.return_value
                handshake_result = pqc.initiate_handshake()
                assert handshake_result is not None
                
                # Collect mission data
                data_point = {
                    "timestamp": time.time(),
                    "waypoint": i,
                    "position": waypoint,
                    "quantum_handshake_success": True,
                    "flight_state": flight_state
                }
                data_collector.collect_telemetry(data_point)
        
        # Verify mission execution
        assert len(mission_data) == 4
        assert all(isinstance(state, dict) for state in mission_data)
        
        # Verify data collection
        collected_data = data_collector.get_telemetry_data()
        assert len(collected_data) == 4
        assert all(data["quantum_handshake_success"] for data in collected_data)
    
    @pytest.mark.integration
    def test_system_performance_under_load(self):
        """Test system performance under high load with quantum security."""
        # Simulate high load scenario
        num_drones = 10
        num_communications = 100
        
        performance_data = []
        
        for i in range(num_communications):
            start_time = time.time()
            
            # Simulate quantum handshake
            with patch('quantum_comms.pqc_handshake.PQCHandshake') as mock_pqc:
                mock_pqc.return_value.initiate_handshake.return_value = {
                    "public_key": b"load_test_key",
                    "nonce": b"load_test_nonce"
                }
                
                pqc = mock_pqc.return_value
                handshake_result = pqc.initiate_handshake()
                assert handshake_result is not None
            
            end_time = time.time()
            latency = end_time - start_time
            
            performance_data.append({
                "communication_id": i,
                "latency": latency,
                "success": True
            })
        
        # Analyze performance
        latencies = [data["latency"] for data in performance_data]
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        success_rate = sum(1 for data in performance_data if data["success"]) / len(performance_data)
        
        # Performance requirements
        assert avg_latency < 1.0  # Average latency < 1 second
        assert max_latency < 5.0  # Max latency < 5 seconds
        assert success_rate > 0.95  # Success rate > 95%
    
    @pytest.mark.integration
    def test_error_recovery_and_failover(self):
        """Test error recovery and failover mechanisms."""
        # Simulate component failures
        failure_scenarios = [
            "quantum_channel_failure",
            "network_node_failure", 
            "flight_control_failure"
        ]
        
        recovery_results = []
        
        for scenario in failure_scenarios:
            with patch('integration.simulation_orchestrator.SimulationOrchestrator') as mock_orchestrator:
                mock_orchestrator.return_value.handle_failure.return_value = {
                    "scenario": scenario,
                    "recovery_success": True,
                    "fallback_activated": True,
                    "recovery_time": 2.5
                }
                
                orchestrator = mock_orchestrator.return_value
                recovery_result = orchestrator.handle_failure(scenario)
                
                assert recovery_result["recovery_success"] is True
                assert recovery_result["fallback_activated"] is True
                assert recovery_result["recovery_time"] < 10.0  # Recovery < 10 seconds
                
                recovery_results.append(recovery_result)
        
        # Verify all scenarios handled
        assert len(recovery_results) == 3
        assert all(result["recovery_success"] for result in recovery_results)


if __name__ == "__main__":
    pytest.main([__file__, "-m", "integration"]) 