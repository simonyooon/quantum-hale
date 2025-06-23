import pytest
import numpy as np
from src.network_sim.ns3_wrapper import NS3Wrapper, NodeConfig, LinkConfig, JammingConfig, NetworkType, RoutingProtocol
from src.network_sim.rf_propagation import EnhancedRFPropagation, PropagationModel, FrequencyBand, AtmosphericParams
from src.network_sim.mesh_routing import EnhancedMeshRouting, RoutingAlgorithm, CoordinationProtocol

def test_mesh_topology_creation():
    ns3 = NS3Wrapper()
    ns3.add_node(NodeConfig(node_id="D1", node_type="drone", position=(0,0,20000)))
    ns3.add_node(NodeConfig(node_id="D2", node_type="drone", position=(10000,0,20000)))
    ns3.add_node(NodeConfig(node_id="GS1", node_type="ground_station", position=(0,0,100)))
    ns3.create_topology(NetworkType.MESH)
    assert len(ns3.links) > 0
    for link in ns3.links:
        assert link.source_id in ns3.nodes
        assert link.destination_id in ns3.nodes

def test_jammer_addition_and_effect():
    ns3 = NS3Wrapper()
    ns3.add_node(NodeConfig(node_id="D1", node_type="drone", position=(0,0,20000)))
    jammer = JammingConfig(jammer_id="J1", position=(0,0,21000), power=50, frequency_range=(2.4e9,2.5e9), jamming_type="continuous")
    ns3.add_jammer(jammer)
    assert "J1" in ns3.jammers
    ns3.create_topology(NetworkType.MESH)
    result = ns3.run_simulation(duration=10)
    assert "jamming_effects" in result.__dict__
    assert any(v > 0 for v in result.jamming_effects.values())

def test_save_and_load_configuration(tmp_path):
    ns3 = NS3Wrapper()
    ns3.add_node(NodeConfig(node_id="D1", node_type="drone", position=(0,0,20000)))
    ns3.create_topology(NetworkType.MESH)
    file_path = tmp_path / "config.json"
    ns3.save_configuration(str(file_path))
    ns3_loaded = NS3Wrapper()
    ns3_loaded.load_configuration(str(file_path))
    assert "D1" in ns3_loaded.nodes
    assert len(ns3_loaded.links) == len(ns3.links)

def test_advanced_rf_propagation_models():
    rf = EnhancedRFPropagation(model=PropagationModel.ADVANCED_ITU_R, frequency_band=FrequencyBand.WIFI_2_4)
    loss = rf.calculate_path_loss(10000, 20000, 100)
    assert loss > 0
    result = rf.calculate_link_budget(30, 10, 10, 10000, 20000, 100)
    assert result.snr > 0
    assert result.availability > 0
    stats = rf.get_propagation_statistics([1000, 5000, 10000], 20000, 100)
    assert "average_path_loss" in stats

def test_mesh_routing_algorithms():
    mesh = EnhancedMeshRouting(algorithm=RoutingAlgorithm.AODV)
    mesh.add_node("D1", (0,0,20000))
    mesh.add_node("D2", (10000,0,20000))
    mesh.add_node("GS1", (0,0,100))
    mesh.add_link("D1", "D2", bandwidth=1e6, latency=0.01)
    mesh.add_link("D1", "GS1", bandwidth=1e6, latency=0.02)
    route = mesh.find_route("D1", "GS1")
    assert route is not None
    assert route.path[0] == "D1" and route.path[-1] == "GS1"
    multipath_routes = mesh.find_multipath_routes("D1", "GS1", max_paths=2)
    assert isinstance(multipath_routes, list)
    assert all(r.path[0] == "D1" and r.path[-1] == "GS1" for r in multipath_routes)

def test_coordination_protocol_leader_follower():
    mesh = EnhancedMeshRouting(algorithm=RoutingAlgorithm.AODV, coordination_protocol=CoordinationProtocol.LEADER_FOLLOWER)
    mesh.add_node("D1", (0,0,20000))
    mesh.add_node("D2", (10000,0,20000))
    mesh.add_link("D1", "D2", bandwidth=1e6, latency=0.01)
    mesh.start()
    import time
    time.sleep(2)  # Let the update loop run
    mesh.stop()
    state = mesh.get_network_state()
    assert "leader_node" in state
    assert state["leader_node"] in ["D1", "D2"] 