import pytest
import time
from src.network_sim.ns3_wrapper import NS3Wrapper, NodeConfig, JammingConfig, NetworkType
from src.network_sim.rf_propagation import EnhancedRFPropagation, PropagationModel, FrequencyBand
from src.network_sim.mesh_routing import EnhancedMeshRouting, RoutingAlgorithm, CoordinationProtocol

def test_end_to_end_mesh_with_jamming():
    ns3 = NS3Wrapper()
    # Add drones and ground stations
    ns3.add_node(NodeConfig(node_id="D1", node_type="drone", position=(0,0,20000)))
    ns3.add_node(NodeConfig(node_id="D2", node_type="drone", position=(10000,0,20000)))
    ns3.add_node(NodeConfig(node_id="D3", node_type="drone", position=(5000,10000,20000)))
    ns3.add_node(NodeConfig(node_id="GS1", node_type="ground_station", position=(0,0,100)))
    ns3.create_topology(NetworkType.MESH)
    # Add a jammer
    jammer = JammingConfig(jammer_id="J1", position=(5000,5000,20000), power=60, frequency_range=(2.4e9,2.5e9), jamming_type="continuous")
    ns3.add_jammer(jammer)
    # Run simulation
    result = ns3.run_simulation(duration=20)
    # Check that jamming effects are present and throughput is reduced
    assert any(v > 0 for v in result.jamming_effects.values())
    assert all(result.throughput[n] < 1e6 for n in ["D1","D2","D3"])

def test_dynamic_topology_and_route_adaptation():
    mesh = EnhancedMeshRouting(algorithm=RoutingAlgorithm.AODV)
    mesh.add_node("D1", (0,0,20000))
    mesh.add_node("D2", (10000,0,20000))
    mesh.add_node("D3", (5000,10000,20000))
    mesh.add_node("GS1", (0,0,100))
    mesh.add_link("D1", "D2", bandwidth=1e6, latency=0.01)
    mesh.add_link("D2", "D3", bandwidth=1e6, latency=0.01)
    mesh.add_link("D3", "GS1", bandwidth=1e6, latency=0.01)
    # Initial route
    route1 = mesh.find_route("D1", "GS1")
    assert route1 is not None
    # Remove a link and check route adaptation
    mesh.remove_link("D2", "D3")
    route2 = mesh.find_route("D1", "GS1")
    assert route2 is not None
    assert route2.path != route1.path

def test_high_altitude_rf_propagation_integration():
    rf = EnhancedRFPropagation(model=PropagationModel.ADVANCED_ITU_R, frequency_band=FrequencyBand.WIFI_2_4)
    # Simulate a high-altitude link
    result = rf.calculate_link_budget(30, 10, 10, 50000, 20000, 100)
    assert result.snr > 0
    assert result.availability > 0.5
    # Simulate with rain
    from src.network_sim.rf_propagation import AtmosphericParams, AtmosphericCondition
    rf.set_atmospheric_conditions(AtmosphericParams(condition=AtmosphericCondition.RAIN, rain_rate=10))
    result_rain = rf.calculate_link_budget(30, 10, 10, 50000, 20000, 100)
    assert result_rain.snr < result.snr

def test_multi_drone_coordination_protocol():
    mesh = EnhancedMeshRouting(algorithm=RoutingAlgorithm.AODV, coordination_protocol=CoordinationProtocol.SWARM)
    mesh.add_node("D1", (0,0,20000))
    mesh.add_node("D2", (10000,0,20000))
    mesh.add_node("D3", (5000,10000,20000))
    mesh.add_link("D1", "D2", bandwidth=1e6, latency=0.01)
    mesh.add_link("D2", "D3", bandwidth=1e6, latency=0.01)
    mesh.start()
    time.sleep(2)
    mesh.stop()
    state = mesh.get_network_state()
    assert "coordination_protocol" in state
    assert state["coordination_protocol"] == "swarm"
    assert len(state["nodes"]) == 3 