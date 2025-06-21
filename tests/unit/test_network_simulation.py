"""
Unit tests for network simulation module.

Tests the network simulation components including NS-3 wrapper,
RF propagation, jamming models, and mesh routing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from network_sim.ns3_wrapper import NS3Wrapper, NetworkTopology
from network_sim.rf_propagation import RFPropagation, PathLossModel
from network_sim.jamming_models import JammingModel, JammerNode
from network_sim.mesh_routing import MeshRouter, RoutingTable


class TestNS3Wrapper:
    """Test cases for NS-3 wrapper implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "simulation_time": 100,
            "log_level": "info",
            "trace_enabled": True
        }
        self.ns3 = NS3Wrapper(self.config)
    
    def test_initialization(self):
        """Test NS-3 wrapper initialization."""
        assert self.ns3.config == self.config
        assert self.ns3.simulation_time == 100
        assert self.ns3.log_level == "info"
    
    def test_topology_creation(self):
        """Test network topology creation."""
        topology_config = {
            "nodes": [
                {"id": "node1", "position": [0, 0, 0]},
                {"id": "node2", "position": [100, 0, 0]}
            ],
            "links": [
                {"source": "node1", "target": "node2", "bandwidth": "1Mbps"}
            ]
        }
        
        topology = self.ns3.create_topology(topology_config)
        
        assert topology is not None
        assert len(topology.nodes) == 2
        assert len(topology.links) == 1
    
    def test_simulation_execution(self):
        """Test simulation execution."""
        with patch.object(self.ns3, '_run_ns3_simulation') as mock_run:
            mock_run.return_value = {"success": True, "results": {}}
            
            results = self.ns3.run_simulation()
            
            assert results["success"] is True
            mock_run.assert_called_once()
    
    def test_node_creation(self):
        """Test node creation in NS-3."""
        node_config = {
            "id": "test_node",
            "type": "wifi",
            "position": [0, 0, 0],
            "antenna_gain": 10
        }
        
        node = self.ns3.create_node(node_config)
        
        assert node is not None
        assert node.id == "test_node"
        assert node.type == "wifi"
    
    def test_link_creation(self):
        """Test link creation in NS-3."""
        link_config = {
            "source": "node1",
            "target": "node2",
            "type": "point_to_point",
            "bandwidth": "1Mbps",
            "delay": "1ms"
        }
        
        link = self.ns3.create_link(link_config)
        
        assert link is not None
        assert link.source == "node1"
        assert link.target == "node2"
    
    def test_trace_collection(self):
        """Test trace data collection."""
        with patch.object(self.ns3, '_collect_traces') as mock_collect:
            mock_collect.return_value = {
                "packet_loss": 0.01,
                "throughput": 950000,
                "latency": 5.2
            }
            
            traces = self.ns3.collect_traces()
            
            assert "packet_loss" in traces
            assert "throughput" in traces
            assert "latency" in traces
    
    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        invalid_config = {"simulation_time": -1}
        
        with pytest.raises(ValueError, match="Invalid simulation time"):
            NS3Wrapper(invalid_config)


class TestNetworkTopology:
    """Test cases for network topology."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.topology = NetworkTopology()
    
    def test_node_addition(self):
        """Test adding nodes to topology."""
        node = {"id": "node1", "position": [0, 0, 0]}
        
        self.topology.add_node(node)
        
        assert len(self.topology.nodes) == 1
        assert self.topology.nodes[0]["id"] == "node1"
    
    def test_link_addition(self):
        """Test adding links to topology."""
        link = {"source": "node1", "target": "node2", "bandwidth": "1Mbps"}
        
        self.topology.add_link(link)
        
        assert len(self.topology.links) == 1
        assert self.topology.links[0]["source"] == "node1"
    
    def test_topology_validation(self):
        """Test topology validation."""
        # Add nodes
        self.topology.add_node({"id": "node1", "position": [0, 0, 0]})
        self.topology.add_node({"id": "node2", "position": [100, 0, 0]})
        
        # Add valid link
        self.topology.add_link({"source": "node1", "target": "node2"})
        
        is_valid = self.topology.validate()
        assert is_valid is True
    
    def test_invalid_link(self):
        """Test invalid link handling."""
        link = {"source": "nonexistent", "target": "node2"}
        
        with pytest.raises(ValueError, match="Invalid link"):
            self.topology.add_link(link)


class TestRFPropagation:
    """Test cases for RF propagation simulation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "frequency": 2.4e9,  # 2.4 GHz
            "tx_power": 30,  # dBm
            "antenna_gain": 10,  # dBi
            "model": "free_space"
        }
        self.rf = RFPropagation(self.config)
    
    def test_initialization(self):
        """Test RF propagation initialization."""
        assert self.rf.frequency == 2.4e9
        assert self.rf.tx_power == 30
        assert self.rf.antenna_gain == 10
    
    def test_free_space_path_loss(self):
        """Test free space path loss calculation."""
        distance = 1000  # meters
        
        path_loss = self.rf.calculate_path_loss(distance)
        
        assert path_loss > 0
        assert isinstance(path_loss, float)
    
    def test_received_power_calculation(self):
        """Test received power calculation."""
        distance = 1000  # meters
        
        rx_power = self.rf.calculate_received_power(distance)
        
        assert rx_power < self.rf.tx_power  # Should be less than tx power
        assert isinstance(rx_power, float)
    
    def test_link_budget_analysis(self):
        """Test link budget analysis."""
        link_config = {
            "distance": 1000,
            "tx_power": 30,
            "rx_sensitivity": -90,
            "fade_margin": 10
        }
        
        budget = self.rf.analyze_link_budget(link_config)
        
        assert "rx_power" in budget
        assert "path_loss" in budget
        assert "margin" in budget
    
    def test_multipath_effects(self):
        """Test multipath effects simulation."""
        distance = 1000
        environment = "urban"
        
        rx_power = self.rf.simulate_multipath(distance, environment)
        
        assert isinstance(rx_power, float)
        assert rx_power < self.rf.tx_power
    
    def test_atmospheric_effects(self):
        """Test atmospheric effects simulation."""
        distance = 10000  # 10 km
        altitude = 20000  # 20 km
        
        rx_power = self.rf.simulate_atmospheric_effects(distance, altitude)
        
        assert isinstance(rx_power, float)
    
    def test_invalid_frequency(self):
        """Test handling of invalid frequency."""
        invalid_config = self.config.copy()
        invalid_config["frequency"] = -1
        
        with pytest.raises(ValueError, match="Invalid frequency"):
            RFPropagation(invalid_config)


class TestPathLossModel:
    """Test cases for path loss models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = PathLossModel()
    
    def test_free_space_model(self):
        """Test free space path loss model."""
        distance = 1000
        frequency = 2.4e9
        
        path_loss = self.model.free_space(distance, frequency)
        
        assert path_loss > 0
        assert isinstance(path_loss, float)
    
    def test_okumura_hata_model(self):
        """Test Okumura-Hata path loss model."""
        distance = 1000
        frequency = 900e6
        height_tx = 30
        height_rx = 1.5
        
        path_loss = self.model.okumura_hata(distance, frequency, height_tx, height_rx)
        
        assert path_loss > 0
        assert isinstance(path_loss, float)
    
    def test_itu_model(self):
        """Test ITU path loss model."""
        distance = 1000
        frequency = 2.4e9
        height_tx = 30
        height_rx = 1.5
        environment = "urban"
        
        path_loss = self.model.itu(distance, frequency, height_tx, height_rx, environment)
        
        assert path_loss > 0
        assert isinstance(path_loss, float)


class TestJammingModel:
    """Test cases for jamming models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "jamming_power": 50,  # dBm
            "frequency_range": [2.4e9, 2.5e9],  # Hz
            "jamming_type": "barrage"
        }
        self.jammer = JammingModel(self.config)
    
    def test_initialization(self):
        """Test jamming model initialization."""
        assert self.jammer.jamming_power == 50
        assert self.jammer.frequency_range == [2.4e9, 2.5e9]
        assert self.jammer.jamming_type == "barrage"
    
    def test_barrage_jamming(self):
        """Test barrage jamming simulation."""
        target_frequency = 2.45e9
        target_power = -60  # dBm
        
        jammed_power = self.jammer.simulate_barrage_jamming(target_frequency, target_power)
        
        assert jammed_power < target_power  # Should reduce received power
        assert isinstance(jammed_power, float)
    
    def test_sweep_jamming(self):
        """Test sweep jamming simulation."""
        target_frequency = 2.45e9
        target_power = -60
        time = 1.0  # seconds
        
        jammed_power = self.jammer.simulate_sweep_jamming(target_frequency, target_power, time)
        
        assert isinstance(jammed_power, float)
    
    def test_reactive_jamming(self):
        """Test reactive jamming simulation."""
        target_frequency = 2.45e9
        target_power = -60
        detection_threshold = -70
        
        jammed_power = self.jammer.simulate_reactive_jamming(
            target_frequency, target_power, detection_threshold
        )
        
        assert isinstance(jammed_power, float)
    
    def test_jamming_effectiveness(self):
        """Test jamming effectiveness calculation."""
        original_snr = 20  # dB
        jammed_snr = 5  # dB
        
        effectiveness = self.jammer.calculate_effectiveness(original_snr, jammed_snr)
        
        assert effectiveness > 0
        assert effectiveness <= 1
    
    def test_invalid_jamming_power(self):
        """Test handling of invalid jamming power."""
        invalid_config = self.config.copy()
        invalid_config["jamming_power"] = -100  # Too low
        
        with pytest.raises(ValueError, match="Invalid jamming power"):
            JammingModel(invalid_config)


class TestJammerNode:
    """Test cases for jammer node simulation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.jammer = JammerNode("jammer1", [0, 0, 0])
    
    def test_jammer_initialization(self):
        """Test jammer node initialization."""
        assert self.jammer.id == "jammer1"
        assert self.jammer.position == [0, 0, 0]
        assert self.jammer.active is False
    
    def test_jammer_activation(self):
        """Test jammer activation."""
        self.jammer.activate()
        
        assert self.jammer.active is True
    
    def test_jammer_deactivation(self):
        """Test jammer deactivation."""
        self.jammer.activate()
        self.jammer.deactivate()
        
        assert self.jammer.active is False
    
    def test_jamming_pattern(self):
        """Test jamming pattern generation."""
        pattern = self.jammer.generate_jamming_pattern()
        
        assert "frequency" in pattern
        assert "power" in pattern
        assert "duration" in pattern


class TestMeshRouter:
    """Test cases for mesh routing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = MeshRouter("router1")
    
    def test_router_initialization(self):
        """Test mesh router initialization."""
        assert self.router.id == "router1"
        assert len(self.router.routing_table) == 0
    
    def test_route_discovery(self):
        """Test route discovery process."""
        destination = "node2"
        
        route = self.router.discover_route(destination)
        
        assert route is not None
        assert "path" in route
        assert "hops" in route
    
    def test_route_maintenance(self):
        """Test route maintenance."""
        route = {
            "destination": "node2",
            "path": ["router1", "node1", "node2"],
            "hops": 2
        }
        
        self.router.maintain_route(route)
        
        assert len(self.router.routing_table) > 0
    
    def test_load_balancing(self):
        """Test load balancing."""
        routes = [
            {"path": ["router1", "node1", "node2"], "cost": 10},
            {"path": ["router1", "node3", "node2"], "cost": 15}
        ]
        
        selected_route = self.router.select_route(routes)
        
        assert selected_route is not None
        assert "path" in selected_route
    
    def test_route_failure_handling(self):
        """Test route failure handling."""
        failed_route = {"destination": "node2", "path": ["router1", "node1", "node2"]}
        
        self.router.handle_route_failure(failed_route)
        
        # Should remove failed route from table
        assert len(self.router.routing_table) == 0


class TestRoutingTable:
    """Test cases for routing table."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.table = RoutingTable()
    
    def test_route_addition(self):
        """Test adding routes to table."""
        route = {
            "destination": "node2",
            "next_hop": "node1",
            "cost": 10,
            "hops": 2
        }
        
        self.table.add_route(route)
        
        assert len(self.table.routes) == 1
        assert self.table.routes[0]["destination"] == "node2"
    
    def test_route_lookup(self):
        """Test route lookup."""
        route = {
            "destination": "node2",
            "next_hop": "node1",
            "cost": 10,
            "hops": 2
        }
        self.table.add_route(route)
        
        found_route = self.table.lookup_route("node2")
        
        assert found_route is not None
        assert found_route["destination"] == "node2"
    
    def test_route_removal(self):
        """Test route removal."""
        route = {
            "destination": "node2",
            "next_hop": "node1",
            "cost": 10,
            "hops": 2
        }
        self.table.add_route(route)
        
        self.table.remove_route("node2")
        
        assert len(self.table.routes) == 0
    
    def test_route_update(self):
        """Test route update."""
        route = {
            "destination": "node2",
            "next_hop": "node1",
            "cost": 10,
            "hops": 2
        }
        self.table.add_route(route)
        
        updated_route = {
            "destination": "node2",
            "next_hop": "node3",
            "cost": 8,
            "hops": 1
        }
        
        self.table.update_route(updated_route)
        
        found_route = self.table.lookup_route("node2")
        assert found_route["next_hop"] == "node3"
        assert found_route["cost"] == 8


if __name__ == "__main__":
    pytest.main([__file__]) 